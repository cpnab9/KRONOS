# --- START OF FILE error_analysis.py ---
import numpy as np
import casadi as cs
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import config as cfg

def compute_and_plot_integration_error(sol, x_vars, u_vars, f_cont, mesh_fractions):
    """
    使用高精度积分法（Integration Error）验证轨迹
    逻辑：提取最优控制量，用 RK45 积分出真实轨迹，对比两者的绝对状态偏差
    """
    print("\n[*] 正在运行高精度积分验证 (Integration Error Analysis)...")
    print("    -> 正在构建连续多项式并提取控制指令...")
    
    d = cfg.d
    tau_root = cs.collocation_points(d, 'radau')
    tau = np.append(0, tau_root)
    tau_u = np.array(tau_root)
    
    # 每个区间内采样 20 个点用于构建平滑曲线
    num_test = 20
    # 注意：使用 endpoint=False 避免相邻区间交界处的时间点重复
    tau_test = np.linspace(0, 1.0, num_test, endpoint=False) 
    
    # 预计算插值矩阵
    Interp_X = np.zeros((num_test, d+1))
    for j in range(d+1):
        p = np.poly1d([1.0])
        for r in range(d+1):
            if r != j: p = np.poly1d([1.0, -tau[r]]) * p / (tau[j]-tau[r])
        for i_test, t_val in enumerate(tau_test):
            Interp_X[i_test, j] = p(t_val)
            
    Interp_U = np.zeros((num_test, d))
    for j in range(d):
        p = np.poly1d([1.0])
        for r in range(d):
            if r != j: p = np.poly1d([1.0, -tau_u[r]]) * p / (tau_u[j]-tau_u[r])
        for i_test, t_val in enumerate(tau_test):
            Interp_U[i_test, j] = p(t_val)

    tf_opt = sol.value(x_vars[0][8])
    t_dense = []
    X_dense = []
    U_dense = []
    
    t_curr = 0.0
    # 重构整个时间轴的致密轨迹
    for k in range(cfg.N):
        dt_k = tf_opt * mesh_fractions[k]
        x_k = sol.value(x_vars[k])
        w_k = sol.value(u_vars[k])
        
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
            
        for i_test in range(num_test):
            t_val = t_curr + tau_test[i_test] * dt_k
            t_dense.append(t_val)
            X_dense.append(X_mat @ Interp_X[i_test, :])
            U_dense.append(U_mat @ Interp_U[i_test, :])
            
        t_curr += dt_k

    # 补齐最后一个节点
    t_dense.append(tf_opt)
    X_dense.append(sol.value(x_vars[-1]))
    # 控制量在 Radau 最后一个节点无定义，沿用上一个计算值即可
    U_dense.append(U_dense[-1]) 
    
    t_dense = np.array(t_dense)
    X_dense = np.array(X_dense).T 
    U_dense = np.array(U_dense).T

    # 构建连续的控制插值函数，供积分器调用
    U_interp = interp1d(t_dense, U_dense, axis=1, bounds_error=False, fill_value="extrapolate")

    # 定义适配 scipy.integrate 的动力学包装器
    def flight_dynamics(t, x_state):
        u_curr = U_interp(t)
        # 调用连续动力学，并将 CasADi 的 DM 对象展平为 numpy array
        dx_dt = f_cont(x_state, u_curr)[0].full().flatten()
        return dx_dt

    print("    -> 正在进行 RK45 数值积分，这可能需要几十秒时间...")
    X0 = sol.value(x_vars[0])
    # 运行高精度积分
    ivp_sol = solve_ivp(
        flight_dynamics, 
        [0, tf_opt], 
        X0, 
        t_eval=t_dense, 
        method='RK45', 
        rtol=1e-8, 
        atol=1e-10
    )

    if not ivp_sol.success:
        print("    ⚠️ 数值积分提前终止:", ivp_sol.message)
        
    X_sim = ivp_sol.y
    t_sec = t_dense * cfg.t_ref

    # 计算模拟轨迹与优化多项式轨迹的物理量绝对偏差
    error_alt = (X_sim[0, :] - X_dense[0, :]) * cfg.R0           # 高度偏差 (m)
    error_vel = (X_sim[3, :] - X_dense[3, :]) * cfg.V_ref        # 速度偏差 (m/s)
    error_gamma = (X_sim[4, :] - X_dense[4, :]) * 180.0 / np.pi  # 弹道倾角偏差 (deg)

    print("    -> 积分完成！正在绘制状态累积偏差...")
    
    plt.style.use('ggplot')
    fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    
    axs[0].plot(t_sec, error_alt, 'r-', linewidth=1.5)
    axs[0].set_ylabel('Altitude Error [m]')
    axs[0].set_title('Integration Verification (Simulation vs Optimization)')
    
    axs[1].plot(t_sec, error_vel, 'b-', linewidth=1.5)
    axs[1].set_ylabel('Velocity Error [m/s]')
    
    axs[2].plot(t_sec, error_gamma, 'g-', linewidth=1.5)
    axs[2].set_ylabel('Gamma Error [deg]')
    axs[2].set_xlabel('Time [s]')

    # === 追加：对比真实轨迹形状（剔除时间相位的影响） ===
    alt_sim = (X_sim[0, :] - 1.0) * cfg.R0 / 1000.0
    alt_opt = (X_dense[0, :] - 1.0) * cfg.R0 / 1000.0
    vel_sim = X_sim[3, :] * cfg.V_ref
    vel_opt = X_dense[3, :] * cfg.V_ref

    fig_shape, ax_shape = plt.subplots(figsize=(8, 5))
    ax_shape.scatter(vel_opt, alt_opt, color='blue', s=20, marker='o', alpha=0.6, label='Optimizer (Discrete Nodes)')
    ax_shape.plot(vel_sim, alt_sim, 'r-', alpha=0.6, linewidth=2, label='RK45 Simulation (True Physics)')
    ax_shape.set_xlabel('Velocity [m/s]')
    ax_shape.set_ylabel('Altitude [km]')
    ax_shape.set_title('Altitude vs Velocity: Shape Comparison')
    ax_shape.legend()
    ax_shape.invert_xaxis()  # 速度从大到小
    ax_shape.grid(True)
    
    # 绘制配点区间的网格线
    t_nodes = [0.0]
    for frac in mesh_fractions:
        t_nodes.append(t_nodes[-1] + frac * tf_opt)
    t_nodes_sec = np.array(t_nodes) * cfg.t_ref
    
    for ax in axs:
        for tn in t_nodes_sec:
            ax.axvline(tn, color='grey', linestyle='--', alpha=0.3)
            
    plt.tight_layout()
    plt.show()
# --- END OF FILE error_analysis.py ---