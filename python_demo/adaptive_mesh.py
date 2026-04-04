# --- START OF FILE adaptive_mesh.py ---
import numpy as np
import casadi as cs
from scipy.interpolate import interp1d
import config as cfg

def compute_mesh_and_warmstart(sol, x_vars, u_vars, f_cont, old_fractions):
    """
    计算下一次迭代的网格时间占比，并对上一次的最优解插值生成热启动(Warm-Start)初值
    """
    print("[*] 正在评估截断误差并重新划分网格...")
    d = cfg.d
    N = cfg.N
    tau_root = cs.collocation_points(d, 'radau')
    tau = np.append(0, tau_root)
    tau_u = np.array(tau_root)
    
    tf_opt = sol.value(x_vars[0][8]) # 【修改】：tf 提取索引修正为 8
    
    t_nodes_old = [0.0]
    for frac in old_fractions:
        t_nodes_old.append(t_nodes_old[-1] + frac * tf_opt)
    
    t_x_old = []
    x_old = []
    t_u_old = []
    u_old = []
    
    errors = np.zeros(N)
    
    tau_test = 0.5 
    p_X = np.zeros(d+1); dp_X = np.zeros(d+1)
    for j in range(d+1):
        p = np.poly1d([1.0])
        for r in range(d+1):
            if r != j: p = np.poly1d([1.0, -tau[r]]) * p / (tau[j]-tau[r])
        p_X[j] = p(tau_test); dp_X[j] = np.polyder(p)(tau_test)
        
    p_U = np.zeros(d)
    for j in range(d):
        p = np.poly1d([1.0])
        for r in range(d):
            if r != j: p = np.poly1d([1.0, -tau_u[r]]) * p / (tau_u[j]-tau_u[r])
        p_U[j] = p(tau_test)

    for k in range(N):
        dt_k = old_fractions[k] * tf_opt
        x_k = sol.value(x_vars[k])
        w_k = sol.value(u_vars[k])
        
        X_mat = np.zeros((cfg.n_x, d+1))
        X_mat[:, 0] = x_k
        t_x_old.append(t_nodes_old[k])
        x_old.append(x_k)
        
        offset_x = 0
        for j in range(d):
            X_mat[:, j+1] = w_k[offset_x : offset_x + cfg.n_x]
            offset_x += cfg.n_x
            t_x_old.append(t_nodes_old[k] + tau[j+1] * dt_k)
            x_old.append(X_mat[:, j+1])
            
        U_mat = np.zeros((cfg.n_u, d))
        offset_u = cfg.d * cfg.n_x
        for j in range(d):
            U_mat[:, j] = w_k[offset_u : offset_u + cfg.n_u]
            offset_u += cfg.n_u
            t_u_old.append(t_nodes_old[k] + tau_u[j] * dt_k)
            u_old.append(U_mat[:, j])
            
        X_poly = X_mat @ p_X
        U_poly = U_mat @ p_U
        dX_poly_dt = (X_mat @ dp_X) / dt_k
        f_eval = f_cont(X_poly, U_poly)[0].full().flatten()
        
        defect = np.abs(dX_poly_dt - f_eval)
        # alpha虽然插入在7号位置，但3和4仍然是V和gamma的位置，无需修改
        e_V = defect[3] * cfg.V_ref / cfg.t_ref
        e_gamma = defect[4] * 180.0 / np.pi / cfg.t_ref
        errors[k] = e_V + e_gamma 

    max_error = np.max(errors)
    mean_error = np.mean(errors)
    print(f"    -> 当前迭代最大截断误差: {max_error:.6e}")
    print(f"    -> 当前迭代平均截断误差: {mean_error:.6e}")

    t_x_old.append(t_nodes_old[-1])
    x_old.append(sol.value(x_vars[-1]))
    
    t_x_old = np.array(t_x_old)
    x_old = np.array(x_old).T 
    t_u_old = np.array(t_u_old)
    u_old = np.array(u_old).T 
    
    eps = 1e-2 
    monitor = np.sqrt(errors) + eps
    desired_dt = 1.0 / monitor 
    desired_fractions = desired_dt / np.sum(desired_dt)
    
    new_fractions = 0.5 * old_fractions + 0.5 * desired_fractions
    new_fractions = new_fractions / np.sum(new_fractions)

    t_nodes_new = [0.0]
    for frac in new_fractions:
        t_nodes_new.append(t_nodes_new[-1] + frac * tf_opt)
    
    interp_x_fun = interp1d(t_x_old, x_old, kind='linear', bounds_error=False, fill_value=(x_old[:, 0], x_old[:, -1]))
    interp_u_fun = interp1d(t_u_old, u_old, kind='linear', bounds_error=False, fill_value=(u_old[:, 0], u_old[:, -1]))
    
    warm_x = []
    warm_u = []
    
    for k in range(N):
        dt_k = new_fractions[k] * tf_opt
        t_k = t_nodes_new[k]
        
        warm_x.append(interp_x_fun(t_k))
        
        w_k_guess = []
        for j in range(1, d+1):  
            t_xc = t_k + tau[j] * dt_k
            w_k_guess.extend(interp_x_fun(t_xc).tolist())
        for j in range(d):       
            t_uc = t_k + tau_u[j] * dt_k
            w_k_guess.extend(interp_u_fun(t_uc).tolist())
            
        warm_u.append(np.array(w_k_guess))
        
    warm_x.append(interp_x_fun(t_nodes_new[-1]))
    
    return new_fractions, warm_x, warm_u, mean_error
# --- END OF FILE adaptive_mesh.py ---