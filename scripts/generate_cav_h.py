import casadi as ca
import numpy as np
import os
import shutil
import json

def generate_flight_ocp():
    print(">>> 开始生成高超声速飞行器轨迹优化算例 (支持网格自适应与动态障碍物)...")
    
    # ==========================================
    # 1. 物理常数与飞行器参数
    # ==========================================
    R0 = 6378000.0         
    g0 = 9.81              
    V_ref = np.sqrt(R0 * g0) 
    t_ref = np.sqrt(R0 / g0) 
    Omega_e = 7.292115e-5 * t_ref 

    m, Aref, rho0, hs = 104305.0, 391.22, 1.225, 7200.0            
    Q_dot_max, n_max, q_max, k_Q = 1500.0 * 1000.0, 2.5, 18000.0, 1.65e-4
    
    sigma_rate_max = 10.0 * np.pi / 180.0 * t_ref 
    alpha_rate_max = 5.0 * np.pi / 180.0 * t_ref
    alpha_max, alpha_min = 40.0 * np.pi / 180.0, 0.0
    
    theta_NFZ1, phi_NFZ1, R_NFZ1 = 5.0 * np.pi / 180.0, 30.0 * np.pi / 180.0, 500000.0 / R0
    theta_NFZ2, phi_NFZ2, R_NFZ2 = 2.5 * np.pi / 180.0, 60.0 * np.pi / 180.0, 500000.0 / R0

    # ==========================================
    # 2. 维度定义 (纯物理)
    # ==========================================
    nx = 9          
    nu_real = 2     
    d = 5           
    K_intervals = 50 
    
    nu = d * nx + d * nu_real  
    ng = d * nx                
    
    # ==========================================
    # 3. 符号变量与动力学函数 (【新增】：引入运行时参数 sym_p)
    # ==========================================
    sym_x = ca.SX.sym('x', nx)
    sym_u = ca.SX.sym('u', nu)
    sym_p = ca.SX.sym('p', 2) # p[0] = mesh_fraction, p[1] = nfz2_flag
    
    X_inner = [sym_u[i*nx : (i+1)*nx] for i in range(d)]
    U_inner = [sym_u[d*nx + i*nu_real : d*nx + (i+1)*nu_real] for i in range(d)]

    def f_cont(x, u):
        r, theta, phi, V, gamma, psi, sigma, alpha, tf = x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8]
        sigma_rate, alpha_rate = u[0], u[1]
        
        safe_r, safe_V = ca.fmax(r, 0.5), ca.fmax(V, 1e-4)         
        safe_cos_phi, safe_cos_gamma = ca.fmax(ca.cos(phi), 1e-4), ca.fmax(ca.cos(gamma), 1e-4)

        rho = rho0 * ca.exp(-(r - 1.0) * R0 / hs)
        V_dim = V * V_ref

        alpha_deg = alpha * 180.0 / ca.pi
        CL = -0.041065 + 0.016292 * alpha_deg + 0.0002602 * alpha_deg**2
        CD = 0.080505 - 0.03026 * CL + 0.86495 * CL**2

        L_acc = (0.5 * rho * V_dim**2 * Aref * CL) / (m * g0)
        D_acc = (0.5 * rho * V_dim**2 * Aref * CD) / (m * g0)

        r_dot = V * ca.sin(gamma)
        theta_dot = V * ca.cos(gamma) * ca.sin(psi) / (safe_r * safe_cos_phi)
        phi_dot = V * ca.cos(gamma) * ca.cos(psi) / safe_r
        V_dot = -D_acc - ca.sin(gamma)/(safe_r**2) + Omega_e**2 * safe_r * ca.cos(phi) * (ca.sin(gamma)*ca.cos(phi) - ca.cos(gamma)*ca.sin(phi)*ca.cos(psi))
        gamma_dot = L_acc * ca.cos(sigma) / safe_V + (V**2 - 1/safe_r) * ca.cos(gamma) / (safe_V*safe_r) + 2*Omega_e*ca.cos(phi)*ca.sin(psi) + Omega_e**2 * safe_r * ca.cos(phi)*(ca.cos(gamma)*ca.cos(phi) + ca.sin(gamma)*ca.sin(phi)*ca.cos(psi))/safe_V
        psi_dot = L_acc * ca.sin(sigma) / (safe_V*safe_cos_gamma) + V*ca.cos(gamma)*ca.sin(psi)*ca.tan(phi)/safe_r - 2*Omega_e*(ca.tan(gamma)*ca.cos(psi)*ca.cos(phi) - ca.sin(phi)) + Omega_e**2*safe_r*ca.sin(phi)*ca.cos(phi)*ca.sin(psi)/(safe_V*safe_cos_gamma)
        
        rhs = ca.vertcat(r_dot, theta_dot, phi_dot, V_dot, gamma_dot, psi_dot, sigma_rate, alpha_rate, 0.0)
        Q_dot = k_Q * ca.sqrt(rho) * (V_dim)**3.15
        q_dyn = 0.5 * rho * V_dim**2
        n_load = ca.sqrt(L_acc**2 + D_acc**2 + 1e-8)
        
        return rhs, Q_dot, q_dyn, n_load

    # ==========================================
    # 4. 配置离散化与约束生成
    # ==========================================
    tau = ca.collocation_points(d, 'radau')
    tau_root = np.append(0, tau)            
    C = np.zeros((d+1, d+1))
    for j in range(d+1):
        p = np.poly1d([1])
        for r in range(d+1):
            if r != j: p = np.polymul(p, np.poly1d([1, -tau_root[r]])) / (tau_root[j]-tau_root[r])
        dp = np.polyder(p)
        for i in range(d+1): C[j, i] = dp(tau_root[i])

    # 【核心修改】：步长不再除以固定的 K_intervals，而是乘以动态网格比例 sym_p[0]
    h = sym_x[8] * sym_p[0] 
    
    g_eq_list, g_ineq_list = [], []
    ineq_lower, ineq_upper = [], []
    idx_s = []  
    fatrop_inf = 1e20 
    
    # 状态边界
    g_ineq_list.extend([sym_x[0], sym_x[3], sym_x[4], sym_x[6], sym_x[7], sym_x[8]])
    ineq_lower.extend([1.0, 100.0/V_ref, -25.0*np.pi/180.0, -85.0*np.pi/180.0, alpha_min, 100.0/t_ref])
    ineq_upper.extend([1.0 + 120000.0/R0, 8000.0/V_ref, 15.0*np.pi/180.0, 85.0*np.pi/180.0, alpha_max, 3000.0/t_ref])
    current_ineq_idx = 6 

    nfz2_flag = sym_p[1] # 提取环境标志

    for i in range(1, d+1):
        x_dot_i = C[0, i] * sym_x
        for j in range(1, d+1): x_dot_i += C[j, i] * X_inner[j-1]
        rhs, Q_dot, q_dyn, n_load = f_cont(X_inner[i-1], U_inner[i-1])
        g_eq_list.append(h * rhs - x_dot_i)
        
        sigma_rate, alpha_rate = U_inner[i-1][0], U_inner[i-1][1]
        theta_cj, phi_cj = X_inner[i-1][1], X_inner[i-1][2]
        dist1_sq = (theta_cj - theta_NFZ1)**2 + (phi_cj - phi_NFZ1)**2
        dist2_sq = (theta_cj - theta_NFZ2)**2 + (phi_cj - phi_NFZ2)**2
        
        # 【核心修改】：利用 nfz2_flag 实现动态开启禁飞区2的 Big-M 约束
        g_ineq_list.extend([sigma_rate, alpha_rate, Q_dot/Q_dot_max, q_dyn/q_max, n_load/n_max, 
                            R_NFZ1**2 - dist1_sq, 
                            R_NFZ2**2 - dist2_sq - (1.0 - nfz2_flag)*100.0]) 
        
        ineq_lower.extend([-sigma_rate_max, -alpha_rate_max, -fatrop_inf, -fatrop_inf, -fatrop_inf, -fatrop_inf, -fatrop_inf])
        ineq_upper.extend([sigma_rate_max, alpha_rate_max, 1.0, 1.0, 1.0, 0.0, 0.0])
        
        idx_s.extend([current_ineq_idx + 2, current_ineq_idx + 3, current_ineq_idx + 4])
        current_ineq_idx += 7
        
    g_eq, g_ineq, f_dyn = ca.vertcat(*g_eq_list), ca.vertcat(*g_ineq_list), X_inner[-1]
    ng_ineq = current_ineq_idx

    # ==========================================
    # 5. 代码生成 (保持与原版高度一致的命名)
    # ==========================================
    sym_lam_dyn, sym_lam_eq, sym_lam_ineq = ca.SX.sym('ld', nx), ca.SX.sym('le', ng), ca.SX.sym('li', ng_ineq)
    ux = ca.vertcat(sym_u, sym_x)
    J_BAbt = ca.densify(ca.jacobian(f_dyn, ux).T)
    J_Ggt  = ca.densify(ca.jacobian(g_eq, ux).T)
    J_Ggt_ineq = ca.densify(ca.jacobian(g_ineq, ux).T) 
    lagrangian = ca.dot(sym_lam_dyn, f_dyn) + ca.dot(sym_lam_eq, g_eq) + ca.dot(sym_lam_ineq, g_ineq)
    H_RSQrqt = ca.densify(ca.hessian(lagrangian, ux)[0])

    filename = 'casadi_codegen.c'
    cgen = ca.CodeGenerator(filename)
    
    # 【核心修改】：所有导出的函数签名增加 sym_p 参数
    for func in [('eval_f_dyn', f_dyn), ('eval_g_eq', g_eq), ('eval_g_ineq', g_ineq), 
                 ('eval_J_BAbt', J_BAbt), ('eval_J_Ggt', J_Ggt), ('eval_J_Ggt_ineq', J_Ggt_ineq)]:
        cgen.add(ca.Function(func[0], [sym_x, sym_u, sym_p], [func[1]]))
        
    cgen.add(ca.Function('eval_H_RSQrqt',[sym_x, sym_u, sym_p, sym_lam_dyn, sym_lam_eq, sym_lam_ineq],[H_RSQrqt]))
    cgen.generate() 

    # 保持原有的移动逻辑，精准贴合 CMake
    os.makedirs('../src/codegen', exist_ok=True)
    shutil.move(filename, os.path.join('../src/codegen', filename))

    # ==========================================
    # 6. JSON 导出 (保持原版逻辑)
    # ==========================================
    x0_guess = [1.0 + 100000.0/R0, 0.0, 0.0, 7450.0/V_ref, -0.5*np.pi/180.0, 0.0, 0.0, 40.0*np.pi/180.0, 1500.0/t_ref]
    xf_guess = [1.0 + 25000.0/R0, 12.0*np.pi/180.0, 70.0*np.pi/180.0, 2000.0/V_ref, -10.0*np.pi/180.0, 90.0*np.pi/180.0, 0.0, 15.0*np.pi/180.0, 1500.0/t_ref]
    
    u0_guess = []
    for _ in range(d): u0_guess.extend(x0_guess)
    for _ in range(d): u0_guess.extend([0.0, 0.0])
    
    uf_guess = []
    for _ in range(d): uf_guess.extend(xf_guess)
    for _ in range(d): uf_guess.extend([0.0, 0.0])
    
    config_dict = {
        "problem_name": "HypersonicFlight_NativeSlack_Adaptive",
        "K_intervals": K_intervals, "nx": nx, "nu": nu, "ng_defects": ng, "ng_ineq": ng_ineq,
        "init_idx": [0, 1, 2, 3, 4, 5, 6, 7],
        "init_val": x0_guess[:8],
        "term_idx": [0, 1, 2, 4, 5],
        "term_val": [1.0 + 25000.0/R0, 12.0*np.pi/180.0, 70.0*np.pi/180.0, -10.0*np.pi/180.0, 90.0*np.pi/180.0],
        "ineq_lower": ineq_lower,
        "ineq_upper": ineq_upper,
        "obj_state_idx": 3,
        "obj_weight": -V_ref,
        
        "guess_x0": x0_guess, 
        "guess_xf": xf_guess,
        "guess_u0": u0_guess,
        "guess_uf": uf_guess,
        
        "ns": len(idx_s),      
        "idx_s": idx_s,
        "zl": [100000.0] * len(idx_s), 
        "Zl": [0.0] * len(idx_s),     
        "guess_sk": [1e-3] * len(idx_s)
    }

    os.makedirs('../config', exist_ok=True)
    with open('../config/ocp_config.json', 'w') as f:
        json.dump(config_dict, f, indent=4)
        
    print(f"✅ 成功生成飞行器轨迹优化 C 代码 (已输出至 ../src/codegen/) 与 JSON 配置！")

if __name__ == "__main__":
    generate_flight_ocp()