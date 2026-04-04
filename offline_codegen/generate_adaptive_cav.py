# --- START OF FILE generate_adaptive_cav.py ---
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

sigma_rate_max = 10.0 * np.pi / 180.0 * t_ref 
alpha_rate_max = 5.0 * np.pi / 180.0 * t_ref   
alpha_max = 40.0 * np.pi / 180.0
alpha_min = 0.0 * np.pi / 180.0
W_penalty = 100000.0

theta_NFZ1, phi_NFZ1, R_NFZ1 = 5.0 * np.pi / 180.0, 30.0 * np.pi / 180.0, 500000.0 / R0
theta_NFZ2, phi_NFZ2, R_NFZ2 = 2.5 * np.pi / 180.0, 60.0 * np.pi / 180.0, 500000.0 / R0

# --- 自适应 KRONOS 配置 ---
d = 5       
N = 40     
K = N + 1
n_x = 9     # [r, theta, phi, V, gamma, psi, sigma, alpha, tf]
n_u = 5     # [sigma_rate, alpha_rate, slack_Q, slack_q, slack_n] 

tau_root = cs.collocation_points(d, 'radau')
tau = np.append(0, tau_root) 

C_mat = np.zeros((d+1, d+1))
for j in range(d+1):
    p = np.poly1d([1.0])
    for r in range(d+1):
        if r != j: p = np.poly1d([1.0, -tau[r]]) * p / (tau[j]-tau[r])
    dp = np.polyder(p)
    for r in range(d+1):
        C_mat[j,r] = dp(tau[r])

# ==========================================
# 2. 连续时间动力学 
# ==========================================
def create_continuous_dynamics():
    states = cs.SX.sym('x', n_x)
    controls = cs.SX.sym('u', n_u)
    
    r, theta, phi, V, gamma, psi, sigma, alpha, tf = states[0], states[1], states[2], states[3], states[4], states[5], states[6], states[7], states[8]
    sigma_rate, alpha_rate = controls[0], controls[1]
    
    safe_r = cs.fmax(r, 0.5)          
    safe_V = cs.fmax(V, 1e-4)         
    safe_cos_phi = cs.fmax(cs.cos(phi), 1e-4)
    safe_cos_gamma = cs.fmax(cs.cos(gamma), 1e-4)

    h_dim = (r - 1.0) * R0
    rho = rho0 * cs.exp(-h_dim / hs)
    V_dim = V * V_ref

    alpha_deg = alpha * 180.0 / cs.pi
    CL = -0.041065 + 0.016292 * alpha_deg + 0.0002602 * alpha_deg**2
    CD = 0.080505 - 0.03026 * CL + 0.86495 * CL**2

    L_acc = (0.5 * rho * V_dim**2 * Aref * CL) / (m * g0)
    D_acc = (0.5 * rho * V_dim**2 * Aref * CD) / (m * g0)

    r_dot = V * cs.sin(gamma)
    theta_dot = V * cs.cos(gamma) * cs.sin(psi) / (safe_r * safe_cos_phi)
    phi_dot = V * cs.cos(gamma) * cs.cos(psi) / safe_r
    V_dot = -D_acc - cs.sin(gamma)/(safe_r**2) + Omega_e**2 * safe_r * cs.cos(phi) * (cs.sin(gamma)*cs.cos(phi) - cs.cos(gamma)*cs.sin(phi)*cs.cos(psi))
    
    gamma_dot = L_acc * cs.cos(sigma) / safe_V + (V**2 - 1/safe_r) * cs.cos(gamma) / (safe_V*safe_r) + 2*Omega_e*cs.cos(phi)*cs.sin(psi) + Omega_e**2 * safe_r * cs.cos(phi)*(cs.cos(gamma)*cs.cos(phi) + cs.sin(gamma)*cs.sin(phi)*cs.cos(psi))/safe_V
    
    psi_dot = L_acc * cs.sin(sigma) / (safe_V*safe_cos_gamma) + V*cs.cos(gamma)*cs.sin(psi)*cs.tan(phi)/safe_r - 2*Omega_e*(cs.tan(gamma)*cs.cos(psi)*cs.cos(phi) - cs.sin(phi)) + Omega_e**2*safe_r*cs.sin(phi)*cs.cos(phi)*cs.sin(psi)/(safe_V*safe_cos_gamma)
    
    sigma_dot = sigma_rate
    alpha_dot = alpha_rate
    tf_dot = 0.0  
    
    rhs = cs.vertcat(r_dot, theta_dot, phi_dot, V_dot, gamma_dot, psi_dot, sigma_dot, alpha_dot, tf_dot)
    return cs.Function("kronos_f_cont", [states, controls], [rhs])

f_cont = create_continuous_dynamics()

# ==========================================
# 3. 伪谱配点与优化问题装配
# ==========================================
opti = cs.Opti()

p_mesh_fractions = opti.parameter(N)
opti.set_value(p_mesh_fractions, np.ones(N) / N)

nx_list = [n_x for _ in range(K)]
nu_list = [d*n_x + d*n_u for _ in range(K-1)] + [0] 
ng_list = [] 

x = []; u = []
for k in range(K):
    x.append(opti.variable(nx_list[k]))
    u.append(opti.variable(nu_list[k]))

slack_sum_path = 0
u_rate_sq_sum = 0

for k in range(K):
    cc = [] 
    
    cc.append((1.0, x[k][0], 1.0 + 120000.0/R0))               
    cc.append((100.0/V_ref, x[k][3], 8000.0/V_ref))            
    cc.append((-25.0*np.pi/180.0, x[k][4], 15.0*np.pi/180.0))  
    cc.append((-85.0*np.pi/180.0, x[k][6], 85.0*np.pi/180.0))  
    cc.append((alpha_min, x[k][7], alpha_max))                 
    cc.append((100.0/t_ref, x[k][8], 3000.0/t_ref))            

    if k < K - 1:
        tf_var = x[k][8] 
        dt_k = tf_var * p_mesh_fractions[k]
        
        offset_x = 0; offset_u = d * n_x
        Xc = []; Uc = []
        for j in range(d):
            Xc.append(u[k][offset_x : offset_x+n_x])
            offset_x += n_x
        for j in range(d):
            Uc.append(u[k][offset_u : offset_u+n_u])
            offset_u += n_u

        eqs = []
        for j in range(1, d+1):
            dX_dt = C_mat[0, j] * x[k]
            for r in range(d):
                dX_dt += C_mat[r+1, j] * Xc[r]
            f_eval = f_cont(Xc[j-1], Uc[j-1])
            eqs.append(dX_dt - dt_k * f_eval)
        
        cc.append((np.zeros(d*n_x), cs.vertcat(*eqs), np.zeros(d*n_x))) 
        opti.subject_to(x[k+1] == Xc[-1])      

        offset_u = d * n_x
        for j in range(d):
            X_cj = Xc[j]
            U_cj = Uc[j]
            
            sigma_rate, alpha_rate = U_cj[0], U_cj[1]
            slack_Q, slack_q, slack_n = U_cj[2], U_cj[3], U_cj[4]
            
            u_rate_sq_sum += (sigma_rate**2 + alpha_rate**2) * dt_k  
            slack_sum_path += (slack_Q + slack_q + slack_n)
            
            cc.append((-sigma_rate_max, sigma_rate, sigma_rate_max))
            cc.append((-alpha_rate_max, alpha_rate, alpha_rate_max))
            cc.append((0.0, slack_Q, cs.inf))
            cc.append((0.0, slack_q, cs.inf))
            cc.append((0.0, slack_n, cs.inf))
            
            theta_cj, phi_cj = X_cj[1], X_cj[2]
            dist_NFZ1_sq = (theta_cj - theta_NFZ1)**2 + (phi_cj - phi_NFZ1)**2
            cc.append((-cs.inf, R_NFZ1**2 - dist_NFZ1_sq, 0.0))
            dist_NFZ2_sq = (theta_cj - theta_NFZ2)**2 + (phi_cj - phi_NFZ2)**2
            cc.append((-cs.inf, R_NFZ2**2 - dist_NFZ2_sq, 0.0))

    if k == 0:
        x0_val =[1.0 + 100000.0/R0, 0.0, 0.0, 7450.0/V_ref, -0.5*np.pi/180.0, 0.0, 0.0, 40.0*np.pi/180.0]
        for i in range(8):
            cc.append((x0_val[i], x[k][i], x0_val[i]))
    elif k == K - 1:
        xf_target =[1.0 + 25000.0/R0, 12.0*np.pi/180.0, 70.0*np.pi/180.0, -10.0*np.pi/180.0, 90.0*np.pi/180.0]
        cc.append((xf_target[0], x[k][0], xf_target[0]))
        cc.append((xf_target[1], x[k][1], xf_target[1]))
        cc.append((xf_target[2], x[k][2], xf_target[2]))
        cc.append((xf_target[3], x[k][4], xf_target[3])) 
        cc.append((xf_target[4], x[k][5], xf_target[4])) 

    ng_list.append(0)
    for constr in cc:
        ng_list[-1] += constr[1].nnz()
        opti.subject_to( (constr[0] <= constr[1]) <= constr[2] )

# ==========================================
# 4. 目标函数与初值填充
# ==========================================
J_obj = -x[N][3] * V_ref + W_penalty * slack_sum_path + 100.0 * u_rate_sq_sum
opti.minimize(J_obj)

tf_guess = 1500.0 / t_ref 
for k in range(K): opti.set_initial(x[k][8], tf_guess)

# ==========================================
# 5. 导出 C 代码给 KRONOS C++ 框架
# ==========================================
opti.solver('fatrop', {
    'structure_detection': 'manual', 
    'nx': nx_list, 
    'nu': nu_list, 
    'ng': ng_list, 
    'N': N, 
    'expand': False,                       
    'fatrop.mu_init': 1e-2                 
})

current_dir = os.path.dirname(os.path.abspath(__file__))
generated_dir = os.path.join(current_dir, '..', 'generated')
os.makedirs(generated_dir, exist_ok=True)
os.chdir(generated_dir)

print("[*] 正在生成紧凑型 NLP 代码与连续动力学函数...")

opti.to_function("kronos_nlp", 
                 [opti.x, opti.lam_g, p_mesh_fractions],  
                 [opti.f, opti.x, opti.lam_g]             
                 ).generate('nlp_code.c', {"with_header": True})

cg = cs.CodeGenerator('f_cont_code.c', {"with_header": True})
cg.add(f_cont)
cg.generate()

# ==========================================
# 6. 动态生成 C 接口的元数据与初值
# ==========================================
print("[*] 正在生成动态元数据接口 (metadata.c)...")
tau_str = ", ".join([str(t) for t in tau_root])

x0_arr = [1.0 + 100000.0/R0, 0.0, 0.0, 7450.0/V_ref, -0.5*np.pi/180.0, 0.0, 0.0, 40.0*np.pi/180.0, 1500.0/t_ref]
xf_arr = [1.0 + 25000.0/R0, 12.0*np.pi/180.0, 70.0*np.pi/180.0, 2000.0/V_ref, -10.0*np.pi/180.0, 90.0*np.pi/180.0, 0.0, 15.0*np.pi/180.0, 1500.0/t_ref]
x0_str = ", ".join([str(x) for x in x0_arr])
xf_str = ", ".join([str(x) for x in xf_arr])

weights_arr = [0.0] * n_x
weights_arr[3] = V_ref / t_ref
weights_arr[4] = (180.0 / np.pi) / t_ref
weights_str = ", ".join([str(w) for w in weights_arr])

c_metadata = f"""
#include <stddef.h>

void kronos_get_dimensions(int* nx, int* nu_base, int* d, int* N, int* tf_index) {{
    if(nx) *nx = {n_x};
    if(nu_base) *nu_base = {n_u};
    if(d) *d = {d};
    if(N) *N = {N};
    if(tf_index) *tf_index = 8;
}}

void kronos_get_tau_root(double* tau) {{
    double temp[] = {{{tau_str}}};
    for(int i=0; i<{d}; ++i) {{
        tau[i] = temp[i];
    }}
}}

void kronos_get_boundaries(double* x0, double* xf) {{
    double temp_x0[] = {{{x0_str}}};
    double temp_xf[] = {{{xf_str}}};
    for(int i=0; i<{n_x}; ++i) {{
        if(x0) x0[i] = temp_x0[i];
        if(xf) xf[i] = temp_xf[i];
    }}
}}

void kronos_get_error_weights(double* weights) {{
    double temp_w[] = {{{weights_str}}};
    for(int i=0; i<{n_x}; ++i) {{
        weights[i] = temp_w[i];
    }}
}}
"""

with open('metadata.c', 'w') as f:
    f.write(c_metadata)

os.chdir(current_dir)
print(f"✅ 自适应网格生成成功！包含动态元数据接口，输出目录: {generated_dir}")
# --- END OF FILE generate_adaptive_cav.py ---