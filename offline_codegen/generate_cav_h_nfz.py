import casadi as cs
import numpy as np
import os

# ==========================================
# 1. 物理常数与伪谱法参数设置
# ==========================================
R0 = 6378000.0         
g0 = 9.81              
V_ref = np.sqrt(R0 * g0) 
t_ref = np.sqrt(R0 / g0) 
Omega_e = 7.292115e-5 * t_ref 

m = 104305.0           
Aref = 391.22          
rho0 = 1.225           
hs = 7200.0            

Q_dot_max = 1500.0 * 1000.0
n_max = 2.5            
q_max = 18000.0        
k_Q = 1.65e-4
u_max = 10.0 * np.pi / 180.0 * t_ref 

W_penalty = 100000.0
W_NFZ = 4000.0   # 禁飞区软约束的惩罚权重

# --- 新增：禁飞区参数 ---
theta_NFZ1, phi_NFZ1, R_NFZ1 = 5.0 * np.pi / 180.0, 30.0 * np.pi / 180.0, 500000.0 / R0
theta_NFZ2, phi_NFZ2, R_NFZ2 = 2.5 * np.pi / 180.0, 60.0 * np.pi / 180.0, 500000.0 / R0

# --- KRONOS 配置 ---
d = 3       # Radau 多项式阶数
N = 20      # 划分的区间数
K = N + 1

n_x = 8     # 状态量: [r, theta, phi, V, gamma, psi, sigma, tf]
# 控制量从 4 增加到 6，加入禁飞区的松弛变量
n_u = 6     # 控制量: [u_rate, slack_Q, slack_q, slack_n, slack_NFZ1, slack_NFZ2]

tau_root = cs.collocation_points(d, 'radau')
tau = np.append(0, tau_root) 

C = np.zeros((d+1, d+1))
for j in range(d+1):
    p = np.poly1d([1.0])
    for r in range(d+1):
        if r != j:
            p = np.poly1d([1.0, -tau[r]]) * p / (tau[j]-tau[r])
    dp = np.polyder(p)
    for r in range(d+1):
        C[j,r] = dp(tau[r])

# ==========================================
# 2. 连续时间动力学 (带数值安全防护)
# ==========================================
def create_continuous_dynamics():
    states = cs.SX.sym('x', n_x)
    controls = cs.SX.sym('u', n_u)
    
    r, theta, phi, V, gamma, psi, sigma, tf = states[0], states[1], states[2], states[3], states[4], states[5], states[6], states[7]
    u = controls[0] 
    
    # 防止除以0的安全底线
    safe_r = cs.fmax(r, 0.5)          
    safe_V = cs.fmax(V, 1e-4)         
    safe_cos_phi = cs.fmax(cs.cos(phi), 1e-4)
    safe_cos_gamma = cs.fmax(cs.cos(gamma), 1e-4)

    h_dim = (r - 1.0) * R0
    rho = rho0 * cs.exp(-h_dim / hs)
    V_dim = V * V_ref

    alpha_deg = cs.if_else(V_dim > 4570.0, 40.0, 40.0 - 0.20705 * ((V_dim - 4570.0) / 340.0)**2)
    CL = -0.041065 + 0.016292 * alpha_deg + 0.0002602 * alpha_deg**2
    CD = 0.080505 - 0.03026 * CL + 0.86495 * CL**2

    L_acc = (0.5 * rho * V_dim**2 * Aref * CL) / (m * g0)
    D_acc = (0.5 * rho * V_dim**2 * Aref * CD) / (m * g0)

    # 运动学方程
    r_dot = V * cs.sin(gamma)
    theta_dot = V * cs.cos(gamma) * cs.sin(psi) / (safe_r * safe_cos_phi)
    phi_dot = V * cs.cos(gamma) * cs.cos(psi) / safe_r
    V_dot = -D_acc - cs.sin(gamma)/(safe_r**2) + Omega_e**2 * safe_r * cs.cos(phi) * (cs.sin(gamma)*cs.cos(phi) - cs.cos(gamma)*cs.sin(phi)*cs.cos(psi))
    
    gamma_dot = L_acc * cs.cos(sigma) / safe_V + (V**2 - 1/safe_r) * cs.cos(gamma) / (safe_V*safe_r) + 2*Omega_e*cs.cos(phi)*cs.sin(psi) + Omega_e**2 * safe_r * cs.cos(phi)*(cs.cos(gamma)*cs.cos(phi) + cs.sin(gamma)*cs.sin(phi)*cs.cos(psi))/safe_V
    
    psi_dot = L_acc * cs.sin(sigma) / (safe_V*safe_cos_gamma) + V*cs.cos(gamma)*cs.sin(psi)*cs.tan(phi)/safe_r - 2*Omega_e*(cs.tan(gamma)*cs.cos(psi)*cs.cos(phi) - cs.sin(phi)) + Omega_e**2*safe_r*cs.sin(phi)*cs.cos(phi)*cs.sin(psi)/(safe_V*safe_cos_gamma)
    
    sigma_dot = u
    tf_dot = 0.0  
    
    rhs = cs.vertcat(r_dot, theta_dot, phi_dot, V_dot, gamma_dot, psi_dot, sigma_dot, tf_dot)
    
    return cs.Function("f_cont", [states, controls], [rhs, rho, V_dim, L_acc, D_acc])

f_cont = create_continuous_dynamics()

# ==========================================
# 3. 伪谱配点函数
# ==========================================
def create_collocation_function():
    X_k = cs.MX.sym('X_k', n_x)
    W_k = cs.MX.sym('W_k', d*n_x + d*n_u)
    
    Xc = []; Uc = []
    offset = 0
    for j in range(d):
        Xc.append(W_k[offset : offset+n_x])
        offset += n_x
    for j in range(d):
        Uc.append(W_k[offset : offset+n_u])
        offset += n_u

    dt = X_k[7] / N 
    eqs = []
    for j in range(1, d+1):
        dX_dt = C[0, j] * X_k
        for r in range(d):
            dX_dt += C[r+1, j] * Xc[r]
        f_eval, _, _, _, _ = f_cont(Xc[j-1], Uc[j-1])
        eqs.append(dX_dt - dt * f_eval)
        
    eqs_cat = cs.vertcat(*eqs)
    X_end = Xc[-1]
    
    return cs.Function("colloc_fun", [X_k, W_k], [X_end, eqs_cat])

colloc_fun = create_collocation_function()

# ==========================================
# 4. 阶段约束与边界条件
# ==========================================
def path_constraints(W_k, X_k, k):
    cc = []
    if k < K - 1:
        _, eqs = colloc_fun(X_k, W_k)
        cc.append((np.zeros(d*n_x), eqs, np.zeros(d*n_x))) 
        
        offset_x = 0; offset_u = d * n_x
        for j in range(d):
            X_cj = W_k[offset_x + j*n_x : offset_x + (j+1)*n_x]
            U_cj = W_k[offset_u + j*n_u : offset_u + (j+1)*n_u]
            
            u_rate = U_cj[0]
            slack_Q, slack_q, slack_n = U_cj[1], U_cj[2], U_cj[3]
            slack_NFZ1, slack_NFZ2 = U_cj[4], U_cj[5]
            
            # --- 基础控制与松弛变量边界 ---
            cc.append((-u_max, u_rate, u_max))
            cc.append((0.0, slack_Q, cs.inf))
            cc.append((0.0, slack_q, cs.inf))
            cc.append((0.0, slack_n, cs.inf))
            cc.append((0.0, slack_NFZ1, cs.inf))
            cc.append((0.0, slack_NFZ2, cs.inf))
            
            # --- 计算配点气动数据 ---
            _, rho, V_dim, L_acc, D_acc = f_cont(X_cj, U_cj)
            safe_V_dim = cs.fmax(V_dim, 1.0)
            safe_rho = cs.fmax(rho, 1e-8)
            
            Q_dot = k_Q * cs.sqrt(safe_rho) * (safe_V_dim)**3.15
            q_dyn = 0.5 * safe_rho * safe_V_dim**2
            n_load = cs.sqrt(L_acc**2 + D_acc**2 + 1e-8)
            
            # --- 动力学与热力学路径不等式约束 ---
            cc.append((-cs.inf, Q_dot / Q_dot_max - (1.0 + slack_Q), 0.0))
            cc.append((-cs.inf, q_dyn / q_max - (1.0 + slack_q), 0.0))
            cc.append((-cs.inf, n_load / n_max - (1.0 + slack_n), 0.0))
            
            # --- 新增：禁飞区避让约束 ---
            theta_cj, phi_cj = X_cj[1], X_cj[2]
            
            dist_NFZ1_sq = (theta_cj - theta_NFZ1)**2 + (phi_cj - phi_NFZ1)**2
            cc.append((-cs.inf, R_NFZ1**2 - dist_NFZ1_sq - slack_NFZ1, 0.0))
            
            dist_NFZ2_sq = (theta_cj - theta_NFZ2)**2 + (phi_cj - phi_NFZ2)**2
            cc.append((-cs.inf, R_NFZ2**2 - dist_NFZ2_sq - slack_NFZ2, 0.0))
            
    if k == 0:
        x0_val =[1.0 + 100000.0/R0, 0.0, 0.0, 7450.0/V_ref, -0.5*np.pi/180.0, 0.0, 0.0]
        for i in range(7):
            cc.append((x0_val[i], X_k[i], x0_val[i]))
            
    elif k == K - 1:
        xf_target =[1.0 + 25000.0/R0, 12.0*np.pi/180.0, 70.0*np.pi/180.0, -10.0*np.pi/180.0, 90.0*np.pi/180.0]
        cc.append((xf_target[0], X_k[0], xf_target[0]))
        cc.append((xf_target[1], X_k[1], xf_target[1]))
        cc.append((xf_target[2], X_k[2], xf_target[2]))
        cc.append((xf_target[3], X_k[4], xf_target[3])) 
        cc.append((xf_target[4], X_k[5], xf_target[4])) 
        
    return cc

def discrete_dynamics(W_k, X_k, k):
    X_next, _ = colloc_fun(X_k, W_k)
    return X_next

# ==========================================
# 5. 装配 Opti 问题与目标函数
# ==========================================
opti = cs.Opti()

nx_list = [n_x for _ in range(K)]
nu_list = [d*n_x + d*n_u for _ in range(K-1)] + [0] 
ng_list = []

x = []; u = []
for k in range(K):
    x.append(opti.variable(nx_list[k]))
    u.append(opti.variable(nu_list[k]))

slack_sum_path = 0
slack_sum_nfz = 0

for k in range(K):
    if k < K - 1:
        opti.subject_to(x[k+1] == discrete_dynamics(u[k], x[k], k)) 
        offset_u = d * n_x
        for j in range(d):
            U_cj = u[k][offset_u + j*n_u : offset_u + (j+1)*n_u]
            # 分别累加路径松弛和禁飞区松弛
            slack_sum_path += (U_cj[1] + U_cj[2] + U_cj[3])
            slack_sum_nfz += (U_cj[4] + U_cj[5])
        
    path_constr = path_constraints(u[k], x[k], k)
    ng_list.append(0)
    for constr in path_constr:
        ng_list[-1] += constr[1].nnz()
        opti.subject_to((constr[0] <= constr[1]) <= constr[2])

# 更新目标函数：加入对禁飞区软约束的惩罚项
J_obj = -x[N][3] * V_ref + W_penalty * slack_sum_path + W_NFZ * slack_sum_nfz
opti.minimize(J_obj)

# ==========================================
# 6. 初值猜测 (Warm-start)
# ==========================================
tf_guess = 1500.0 / t_ref 

x0_guess = np.array([
    1.0 + 100000.0/R0,       
    0.0,                     
    0.0,                     
    7450.0/V_ref,            
    -0.5*np.pi/180.0,        
    0.0,                     
    0.0,                     
    tf_guess                 
])

xf_guess = np.array([
    1.0 + 25000.0/R0,        
    12.0*np.pi/180.0,        
    70.0*np.pi/180.0,        
    2000.0/V_ref,            
    -10.0*np.pi/180.0,       
    90.0*np.pi/180.0,        
    0.0,                     
    tf_guess                 
])

for k in range(K):
    interp_frac = k / (K - 1) if K > 1 else 0
    x_curr_guess = x0_guess + interp_frac * (xf_guess - x0_guess)
    
    for i in range(n_x):
        opti.set_initial(x[k][i], x_curr_guess[i])
        
    if k < K - 1:
        u_guess = []
        for j in range(d):
            u_guess.extend(x_curr_guess.tolist())
        # [u_rate, slack_Q, slack_q, slack_n, slack_NFZ1, slack_NFZ2] 全赋初值 0
        for j in range(d):
            u_guess.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            
        opti.set_initial(u[k], u_guess)

# ==========================================
# 7. 导出给 fatrop 的 C 代码
# ==========================================
opti.solver('fatrop', {
    'structure_detection': 'manual', 
    'nx': nx_list, 
    'nu': nu_list, 
    'ng': ng_list, 
    'N': N, 
    'expand': True, 
    'fatrop.mu_init': 1e-1
})

current_dir = os.path.dirname(os.path.abspath(__file__))
generated_dir = os.path.join(current_dir, '..', 'generated')
os.makedirs(generated_dir, exist_ok=True)
os.chdir(generated_dir)

print("[*] 开始生成带禁飞区的 KRONOS NLP C代码...")
opti.to_function("kronos_nlp", [], [opti.f, opti.x]).generate('nlp_code.c', {"with_header": True})

# 【新增】：生成用于 C++ 导出 CSV 的维度元数据
with open('problem_metadata.h', 'w') as f:
    f.write("#pragma once\n\n")
    f.write("// CAV_H_NFZ 问题的维度元数据\n")
    f.write(f"#define KRONOS_N  {N}\n")
    f.write(f"#define KRONOS_NX {n_x}\n")
    f.write(f"#define KRONOS_NU {d*n_x + d*n_u} // 伪谱法下 u 包含配点状态\n")
    f.write(f"#define KRONOS_D  {d}\n\n")
    f.write("// 变量名列表\n")
    f.write('#define KRONOS_X_NAMES "r,theta,phi,V,gamma,psi,sigma,tf"\n')
    f.write('#define KRONOS_U_NAMES "u_rate,slack_Q,slack_q,slack_n,slack_NFZ1,slack_NFZ2"\n')

os.chdir(current_dir)
print(f"✅ Code generation and metadata successful! Files automatically saved to: {generated_dir}")