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

# --- KRONOS 配置 ---
d = 4       # Radau 多项式阶数
N = 20      # 划分的区间数
K = N + 1

n_x = 8     # 状态量: [r, theta, phi, V, gamma, psi, sigma, tf] 
n_u = 4     # 控制量: [u_rate, slack_Q, slack_q, slack_n] 

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
# 2. 连续时间动力学 (加入数值安全防护)
# ==========================================
def create_continuous_dynamics():
    states = cs.SX.sym('x', n_x)
    controls = cs.SX.sym('u', n_u)
    
    r, theta, phi, V, gamma, psi, sigma, tf = states[0], states[1], states[2], states[3], states[4], states[5], states[6], states[7]
    u = controls[0] 
    
    # ---------------------------------------------------------
    # 【核心修复】：为分母变量提供安全底线，防止求解器探索时除以0
    safe_r = cs.fmax(r, 0.5)          # 半径无量纲值不可能小于地球的一半
    safe_V = cs.fmax(V, 1e-4)         # 速度不可能真的为0
    safe_cos_phi = cs.fmax(cs.cos(phi), 1e-4)
    safe_cos_gamma = cs.fmax(cs.cos(gamma), 1e-4)
    # ---------------------------------------------------------

    h_dim = (r - 1.0) * R0
    rho = rho0 * cs.exp(-h_dim / hs)
    V_dim = V * V_ref

    alpha_deg = cs.if_else(V_dim > 4570.0, 40.0, 40.0 - 0.20705 * ((V_dim - 4570.0) / 340.0)**2)
    CL = -0.041065 + 0.016292 * alpha_deg + 0.0002602 * alpha_deg**2
    CD = 0.080505 - 0.03026 * CL + 0.86495 * CL**2

    L_acc = (0.5 * rho * V_dim**2 * Aref * CL) / (m * g0)
    D_acc = (0.5 * rho * V_dim**2 * Aref * CD) / (m * g0)

    # 运动学方程 (使用 safe_* 变量替换分母危险项)
    r_dot = V * cs.sin(gamma)
    theta_dot = V * cs.cos(gamma) * cs.sin(psi) / (safe_r * safe_cos_phi)
    phi_dot = V * cs.cos(gamma) * cs.cos(psi) / safe_r
    V_dot = -D_acc - cs.sin(gamma)/(safe_r**2) + Omega_e**2 * safe_r * cs.cos(phi) * (cs.sin(gamma)*cs.cos(phi) - cs.cos(gamma)*cs.sin(phi)*cs.cos(psi))
    
    gamma_dot = L_acc * cs.cos(sigma) / safe_V + (V**2 - 1/safe_r) * cs.cos(gamma) / (safe_V*safe_r) + 2*Omega_e*cs.cos(phi)*cs.sin(psi) + Omega_e**2 * safe_r * cs.cos(phi)*(cs.cos(gamma)*cs.cos(phi) + cs.sin(gamma)*cs.sin(phi)*cs.cos(psi))/safe_V
    
    psi_dot = L_acc * cs.sin(sigma) / (safe_V*safe_cos_gamma) + V*cs.cos(gamma)*cs.sin(psi)*cs.tan(phi)/safe_r - 2*Omega_e*(cs.tan(gamma)*cs.cos(psi)*cs.cos(phi) - cs.sin(phi)) + Omega_e**2*safe_r*cs.sin(phi)*cs.cos(phi)*cs.sin(psi)/(safe_V*safe_cos_gamma)
    
    sigma_dot = u
    tf_dot = 0.0  # 全局时间作为状态量，其导数为0
    
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

    dt = X_k[7] / N # tf 储存在索引 7
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
            
            # 基础控制与松弛变量边界
            cc.append((-u_max, u_rate, u_max))
            cc.append((0.0, slack_Q, cs.inf))
            cc.append((0.0, slack_q, cs.inf))
            cc.append((0.0, slack_n, cs.inf))
            
            # 计算配点气动数据
            _, rho, V_dim, L_acc, D_acc = f_cont(X_cj, U_cj)
            
            safe_V_dim = cs.fmax(V_dim, 1.0)
            safe_rho = cs.fmax(rho, 1e-8)
            
            Q_dot = k_Q * cs.sqrt(safe_rho) * (safe_V_dim)**3.15
            q_dyn = 0.5 * safe_rho * safe_V_dim**2
            n_load = cs.sqrt(L_acc**2 + D_acc**2 + 1e-8)
            
            # 路径不等式约束转换 (带松弛)
            cc.append((-cs.inf, Q_dot / Q_dot_max - (1.0 + slack_Q), 0.0))
            cc.append((-cs.inf, q_dyn / q_max - (1.0 + slack_q), 0.0))
            cc.append((-cs.inf, n_load / n_max - (1.0 + slack_n), 0.0))
            
    if k == 0:
        x0_val =[1.0 + 100000.0/R0, 0.0, 0.0, 7450.0/V_ref, -0.5*np.pi/180.0, 0.0, 0.0]
        for i in range(7):
            cc.append((x0_val[i], X_k[i], x0_val[i]))
            
    elif k == K - 1:
        xf_target =[1.0 + 25000.0/R0, 12.0*np.pi/180.0, 70.0*np.pi/180.0, -10.0*np.pi/180.0, 90.0*np.pi/180.0]
        cc.append((xf_target[0], X_k[0], xf_target[0]))
        cc.append((xf_target[1], X_k[1], xf_target[1]))
        cc.append((xf_target[2], X_k[2], xf_target[2]))
        cc.append((xf_target[3], X_k[4], xf_target[3])) # gamma 是 index 4
        cc.append((xf_target[4], X_k[5], xf_target[4])) # psi 是 index 5
        
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

slack_sum = 0
for k in range(K):
    if k < K - 1:
        opti.subject_to(x[k+1] == discrete_dynamics(u[k], x[k], k)) 
        offset_u = d * n_x
        for j in range(d):
            U_cj = u[k][offset_u + j*n_u : offset_u + (j+1)*n_u]
            slack_sum += (U_cj[1] + U_cj[2] + U_cj[3])
        
    path_constr = path_constraints(u[k], x[k], k)
    ng_list.append(0)
    for constr in path_constr:
        ng_list[-1] += constr[1].nnz()
        opti.subject_to((constr[0] <= constr[1]) <= constr[2])

J_obj = -x[N][3] * V_ref + W_penalty * slack_sum
opti.minimize(J_obj)

# ==========================================
# 6. 初值猜测 (Warm-start) - 彻底解决 0 初始化的崩溃
# ==========================================
tf_guess = 1500.0 / t_ref 

# 状态量起始终点猜测
x0_guess = np.array([
    1.0 + 100000.0/R0,       # r 
    0.0,                     # theta
    0.0,                     # phi
    7450.0/V_ref,            # V
    -0.5*np.pi/180.0,        # gamma
    0.0,                     # psi
    0.0,                     # sigma
    tf_guess                 # tf
])

xf_guess = np.array([
    1.0 + 25000.0/R0,        # r 
    12.0*np.pi/180.0,        # theta
    70.0*np.pi/180.0,        # phi
    2000.0/V_ref,            # V (终点速度猜测)
    -10.0*np.pi/180.0,       # gamma
    90.0*np.pi/180.0,        # psi
    0.0,                     # sigma
    tf_guess                 # tf
])

for k in range(K):
    # 简单的线性插值来猜算中间节点
    interp_frac = k / (K - 1) if K > 1 else 0
    x_curr_guess = x0_guess + interp_frac * (xf_guess - x0_guess)
    
    # 赋予主节点状态初值
    for i in range(n_x):
        opti.set_initial(x[k][i], x_curr_guess[i])
        
    # 赋予内部配点状态及控制量初值
    if k < K - 1:
        u_guess = []
        for j in range(d):
            u_guess.extend(x_curr_guess.tolist())
        # [u_rate, slack_Q, slack_q, slack_n]
        for j in range(d):
            u_guess.extend([0.0, 0.0, 0.0, 0.0])
            
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

print("[*] 开始生成 KRONOS NLP C代码 (需等待数秒进行符号代数化简) ...")
opti.to_function("kronos_nlp", [], [opti.f, opti.x]).generate('nlp_code.c', {"with_header": True})

os.chdir(current_dir)
print(f"✅ Code generation successful! Files automatically saved to: {generated_dir}")