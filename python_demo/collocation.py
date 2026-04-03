# --- START OF FILE collocation.py ---
import casadi as cs
import numpy as np
import config as cfg

def create_collocation_function(f_cont):
    """基于提供的连续时间动力学，创建离散配点函数 (支持可变步长 dt_k)"""
    X_k = cs.MX.sym('X_k', cfg.n_x)
    W_k = cs.MX.sym('W_k', cfg.d*cfg.n_x + cfg.d*cfg.n_u)
    dt_k = cs.MX.sym('dt_k', 1) 
    
    Xc = []; Uc = []
    offset = 0
    for j in range(cfg.d):
        Xc.append(W_k[offset : offset+cfg.n_x])
        offset += cfg.n_x
    for j in range(cfg.d):
        Uc.append(W_k[offset : offset+cfg.n_u])
        offset += cfg.n_u

    eqs = []
    for j in range(1, cfg.d+1):
        dX_dt = cfg.C[0, j] * X_k
        for r in range(cfg.d):
            dX_dt += cfg.C[r+1, j] * Xc[r]
        f_eval, _, _, _, _ = f_cont(Xc[j-1], Uc[j-1])
        eqs.append(dX_dt - dt_k * f_eval)
        
    eqs_cat = cs.vertcat(*eqs)
    X_end = Xc[-1]
    return cs.Function("colloc_fun", [X_k, W_k, dt_k], [X_end, eqs_cat])

def get_path_constraints(W_k, X_k, k, colloc_fun, f_cont, dt_k):
    """生成各个配点上的路径与边界约束"""
    cc = []
    
    # ================= 新增：物理状态硬边界 =================
    # 统一在这里添加，如果是 manual 结构检测模式，也会被自动统计到 ng_list 中
    cc.append((1.0, X_k[0], 1.0 + 120000.0/cfg.R0))               # r: 限制在地面到 120km 之间
    cc.append((100.0/cfg.V_ref, X_k[3], 8000.0/cfg.V_ref))        # V: 速度下限与上限
    cc.append((-25.0*np.pi/180.0, X_k[4], 15.0*np.pi/180.0))      # gamma: 弹道倾角不能大起大落
    cc.append((-85.0*np.pi/180.0, X_k[6], 85.0*np.pi/180.0))      # sigma: 倾侧角限制防翻转
    cc.append((100.0/cfg.t_ref, X_k[7], 3000.0/cfg.t_ref))        # tf: 总时间在合理范围内

    if k < cfg.K - 1:
        _, eqs = colloc_fun(X_k, W_k, dt_k) 
        cc.append((np.zeros(cfg.d*cfg.n_x), eqs, np.zeros(cfg.d*cfg.n_x))) 
        
        offset_x = 0; offset_u = cfg.d * cfg.n_x
        for j in range(cfg.d):
            X_cj = W_k[offset_x + j*cfg.n_x : offset_x + (j+1)*cfg.n_x]
            U_cj = W_k[offset_u + j*cfg.n_u : offset_u + (j+1)*cfg.n_u]
            
            u_rate = U_cj[0]
            slack_Q, slack_q, slack_n = U_cj[1], U_cj[2], U_cj[3]
            
            cc.append((-cfg.u_max, u_rate, cfg.u_max))
            cc.append((0.0, slack_Q, cs.inf))
            cc.append((0.0, slack_q, cs.inf))
            cc.append((0.0, slack_n, cs.inf))
            
            _, rho, V_dim, L_acc, D_acc = f_cont(X_cj, U_cj)
            safe_V_dim = cs.fmax(V_dim, 1.0)
            safe_rho = cs.fmax(rho, 1e-8)
            
            Q_dot = cfg.k_Q * cs.sqrt(safe_rho) * (safe_V_dim)**3.15
            q_dyn = 0.5 * safe_rho * safe_V_dim**2
            n_load = cs.sqrt(L_acc**2 + D_acc**2 + 1e-8)
            
            cc.append((-cs.inf, Q_dot / cfg.Q_dot_max - (1.0 + slack_Q), 0.0))
            cc.append((-cs.inf, q_dyn / cfg.q_max - (1.0 + slack_q), 0.0))
            cc.append((-cs.inf, n_load / cfg.n_max - (1.0 + slack_n), 0.0))
            
            theta_cj, phi_cj = X_cj[1], X_cj[2]
            
            # 禁飞区约束
            dist_NFZ1_sq = (theta_cj - cfg.theta_NFZ1)**2 + (phi_cj - cfg.phi_NFZ1)**2
            cc.append((-cs.inf, cfg.R_NFZ1**2 - dist_NFZ1_sq, 0.0))
            
            dist_NFZ2_sq = (theta_cj - cfg.theta_NFZ2)**2 + (phi_cj - cfg.phi_NFZ2)**2
            cc.append((-cs.inf, cfg.R_NFZ2**2 - dist_NFZ2_sq, 0.0))
            
    if k == 0:
        x0_val =[1.0 + 100000.0/cfg.R0, 0.0, 0.0, 7450.0/cfg.V_ref, -0.5*np.pi/180.0, 0.0, 0.0]
        for i in range(7):
            cc.append((x0_val[i], X_k[i], x0_val[i]))
            
    elif k == cfg.K - 1:
        xf_target =[1.0 + 25000.0/cfg.R0, 12.0*np.pi/180.0, 70.0*np.pi/180.0, -10.0*np.pi/180.0, 90.0*np.pi/180.0]
        cc.append((xf_target[0], X_k[0], xf_target[0]))
        cc.append((xf_target[1], X_k[1], xf_target[1]))
        cc.append((xf_target[2], X_k[2], xf_target[2]))
        cc.append((xf_target[3], X_k[4], xf_target[3])) 
        cc.append((xf_target[4], X_k[5], xf_target[4])) 
        
    return cc
# --- END OF FILE collocation.py ---