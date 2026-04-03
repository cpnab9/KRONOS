# --- START OF FILE error_analysis.py ---
import numpy as np
import casadi as cs
import matplotlib.pyplot as plt
import config as cfg

def compute_and_plot_error(sol, x_vars, u_vars, f_cont):
    """
    计算并绘制当前轨迹的动力学截断误差（Defect Error）
    """
    print("[*] 正在计算动力学离散误差...")
    
    d = cfg.d
    tau_root = cs.collocation_points(d, 'radau')
    tau = np.append(0, tau_root)
    tau_u = np.array(tau_root)  # 控制量仅在配点上有定义
    
    # 每个区间内采样 20 个点用于观察连续误差
    num_test = 20
    tau_test = np.linspace(0, 1.0, num_test)
    
    # 1. 预计算拉格朗日插值矩阵和导数矩阵
    Interp_X = np.zeros((num_test, d+1))
    Deriv_X = np.zeros((num_test, d+1))
    for j in range(d+1):
        p = np.poly1d([1.0])
        for r in range(d+1):
            if r != j:
                p = np.poly1d([1.0, -tau[r]]) * p / (tau[j]-tau[r])
        dp = np.polyder(p)
        for i_test, t_val in enumerate(tau_test):
            Interp_X[i_test, j] = p(t_val)
            Deriv_X[i_test, j] = dp(t_val)
            
    Interp_U = np.zeros((num_test, d))
    for j in range(d):
        p = np.poly1d([1.0])
        for r in range(d):
            if r != j:
                p = np.poly1d([1.0, -tau_u[r]]) * p / (tau_u[j]-tau_u[r])
        for i_test, t_val in enumerate(tau_test):
            Interp_U[i_test, j] = p(t_val)

    # 2. 遍历所有网格区间，计算误差
    tf_opt = sol.value(x_vars[0][7])  # 提取优化后的总时间(无量纲)
    dt_nondim = tf_opt / cfg.N        # 均匀网格下的无量纲步长
    
    time_grid = []
    error_V_acc = []     # 速度导数误差 (即加速度误差) m/s^2
    error_gamma_rate = []# 弹道倾角导数误差 deg/s
    error_h_rate = []    # 高度导数误差 (垂直速度误差) m/s

    for k in range(cfg.N):
        x_k = sol.value(x_vars[k])
        w_k = sol.value(u_vars[k])
        
        # 重构区间内配点的矩阵表达
        X_mat = np.zeros((cfg.n_x, d+1))
        X_mat[:, 0] = x_k
        offset_x = 0
        for j in range(d):
            X_mat[:, j+1] = w_k[offset_x : offset_x + cfg.n_x]
            offset_x += cfg.n_x
            
        U_mat = np.zeros((cfg.n_u, d))
        offset_u = cfg.d * cfg.n_x
        for j in range(d):
            U_mat[:, j] = w_k[offset_u : offset_u + cfg.n_u]
            offset_u += cfg.n_u
            
        # 在测试点上计算误差
        for i_test in range(num_test):
            # 获取测试点状态的无量纲多项式评估值
            X_poly = X_mat @ Interp_X[i_test, :]
            U_poly = U_mat @ Interp_U[i_test, :]
            
            # 多项式解析对时间的导数 (无量纲)
            dX_poly_dtau = X_mat @ Deriv_X[i_test, :]
            dX_poly_dt = dX_poly_dtau / dt_nondim 
            
            # 真实动力学给出的导数 (利用已有的连续动力学函数)
            f_eval = f_cont(X_poly, U_poly)[0].full().flatten()
            
            # 导数之差即为动力学违约误差 (Defect)
            defect_nondim = np.abs(dX_poly_dt - f_eval)
            
            # 转换为直观的物理量进行保存
            # r 的导数误差 * (R0/t_ref) => 垂直速度误差 (m/s)
            error_h_rate.append(defect_nondim[0] * (cfg.R0 / cfg.t_ref)) 
            # V 的导数误差 * (V_ref/t_ref) => 加速度误差 (m/s^2)
            error_V_acc.append(defect_nondim[3] * (cfg.V_ref / cfg.t_ref)) 
            # gamma 的导数误差 * (180/pi / t_ref) => 姿态角速率误差 (deg/s)
            error_gamma_rate.append(defect_nondim[4] * (180.0 / np.pi) / cfg.t_ref)
            
            # 记录时间轴
            t_current = (k + tau_test[i_test]) * dt_nondim * cfg.t_ref
            time_grid.append(t_current)

    # 3. 绘制误差图表
    plt.style.use('ggplot')
    fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    
    axs[0].plot(time_grid, error_V_acc, 'b-', linewidth=1.5)
    axs[0].set_ylabel('Acc Error [m/s^2]')
    axs[0].set_title('Dynamics Defect Error Analysis (Polynomial vs True)')
    
    axs[1].plot(time_grid, error_gamma_rate, 'g-', linewidth=1.5)
    axs[1].set_ylabel('Gamma Rate Error [deg/s]')
    
    axs[2].plot(time_grid, error_h_rate, 'r-', linewidth=1.5)
    axs[2].set_ylabel('Vertical Vel Error [m/s]')
    axs[2].set_xlabel('Time [s]')
    
    # 绘制配点区间的网格线，帮助直观观察节点位置
    t_nodes = [k * dt_nondim * cfg.t_ref for k in range(cfg.N + 1)]
    for ax in axs:
        for tn in t_nodes:
            ax.axvline(tn, color='grey', linestyle='--', alpha=0.3)
            
    plt.tight_layout()
    plt.show()
# --- END OF FILE error_analysis.py ---