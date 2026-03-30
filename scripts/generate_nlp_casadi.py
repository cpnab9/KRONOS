import casadi as ca
import numpy as np
import os
from numpy.polynomial.legendre import Legendre

def get_lgl_nodes_and_D(N):
    # 与之前完全一致的 LGL 节点生成代码
    n = N - 1
    tau = np.zeros(N)
    tau[0], tau[-1] = -1.0, 1.0
    c = np.zeros(n + 1)
    c[n] = 1
    P_n = Legendre(c)
    P_n_deriv = P_n.deriv()
    if n > 1:
        tau[1:-1] = np.sort(P_n_deriv.roots())
    c_i = np.ones(N)
    c_i[0], c_i[-1] = 2.0, 2.0
    D = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i != j:
                D[i, j] = (c_i[i] / c_i[j]) * ((-1)**(i+j)) / (tau[i] - tau[j])
    for i in range(N):
        D[i, i] = -np.sum(D[i, :])
    return tau, D

def generate_segment_code():
    print("开始生成最终版 CasADi 分段 C 代码 (含边界与障碍函数)...")
    
    K = 2
    N = 10
    g = 9.81
    n_x, n_u = 3, 1
    tau, D = get_lgl_nodes_and_D(N)
    
    # --- 1. 声明符号变量 ---
    X_local = ca.SX.sym('X_local', n_x * N) 
    U_local = ca.SX.sym('U_local', n_u * N)
    T_global = ca.SX.sym('T_global', 1)
    lam_local = ca.SX.sym('lam_local', n_x * N)
    
    Z_local = ca.vertcat(X_local, U_local)
    X_mat = ca.reshape(X_local, n_x, N)
    U_mat = ca.reshape(U_local, n_u, N)
    
    # 【新增】: 优化与边界参数
    mu = ca.SX.sym('mu', 1)                   # 对数障碍参数
    w_start = ca.SX.sym('w_start', n_x)       # 起点权重开关 (0 或 1)
    x_ref_start = ca.SX.sym('x_ref_start', n_x) # 起点坐标
    w_end = ca.SX.sym('w_end', n_x)           # 终点权重开关 (0 或 1)
    x_ref_end = ca.SX.sym('x_ref_end', n_x)     # 终点坐标
    
    # --- 2. 构建局部动力学约束 (Defects) ---
    dX_dtau = ca.mtimes(X_mat, ca.DM(D).T) 
    defects = []
    
    f_cost_local = 0.0 # 局部代价函数
    
    for i in range(N):
        v_i = X_mat[2, i]
        theta_i = U_mat[0, i]
        
        x_dot = v_i * ca.cos(theta_i)
        y_dot = v_i * ca.sin(theta_i)
        v_dot = -g * ca.sin(theta_i)
        
        f_i = ca.vertcat(x_dot, y_dot, v_dot)
        defect_i = dX_dtau[:, i] - (T_global / (2 * K)) * f_i
        defects.append(defect_i)
        
        # 【新增】：物理限幅的对数障碍函数 (Log-Barrier)
        # 1. 速度限幅: v > 0 (加一个极小的偏移防止初始猜测 log(0) 报错)
        f_cost_local -= mu * ca.log(v_i + 1e-4)
        
        # 2. 控制角限幅: theta 在 [-pi, pi/2] 之间
        f_cost_local -= mu * ca.log(theta_i + ca.pi)
        f_cost_local -= mu * ca.log(ca.pi/2 - theta_i)
        
    g_local = ca.vertcat(*defects) 
    
    # 【新增】：二次惩罚项用于锚定起点和终点 (大权重 1e5 模拟硬约束)
    diff_start = w_start * (X_mat[:, 0] - x_ref_start)
    f_cost_local += 1e5 * ca.dot(diff_start, diff_start)
    
    diff_end = w_end * (X_mat[:, -1] - x_ref_end)
    f_cost_local += 1e5 * ca.dot(diff_end, diff_end)
    
    # --- 3. 符号求导 ---
    jac_g_Z = ca.jacobian(g_local, Z_local)
    jac_g_T = ca.jacobian(g_local, T_global)
    
    L_local = f_cost_local + ca.dot(lam_local, g_local)
    hess_L_Z, _ = ca.hessian(L_local, Z_local)
    grad_L_Z = ca.jacobian(L_local, Z_local).T
    
    # --- 4. 边界与段间连续性约束 (保持不变) ---
    x_end_i = ca.SX.sym('x_end_i', n_x)       
    x_start_ip1 = ca.SX.sym('x_start_ip1', n_x) 
    g_link = x_start_ip1 - x_end_i            
    jac_link_i = ca.jacobian(g_link, x_end_i)
    jac_link_ip1 = ca.jacobian(g_link, x_start_ip1)
    
    # --- 5. 封装导出 C 代码 ---
    opts = {"jit": False}
    
    # 【注意输入参数的变化】加入了 mu, w_start 等
    func_eval_local = ca.Function('eval_segment_local', 
                                  [Z_local, T_global, lam_local, mu, w_start, x_ref_start, w_end, x_ref_end], 
                                  [g_local, jac_g_Z, jac_g_T, hess_L_Z, grad_L_Z],
                                  ['Z', 'T', 'lam', 'mu', 'w_start', 'x_start', 'w_end', 'x_end'], 
                                  ['g_defects', 'jac_g_Z', 'jac_g_T', 'hess_L_Z', 'grad_L_Z'], 
                                  opts)
                                  
    func_eval_link = ca.Function('eval_segment_link',
                                 [x_end_i, x_start_ip1],
                                 [g_link, jac_link_i, jac_link_ip1],
                                 ['x_end', 'x_start'],
                                 ['g_link', 'jac_end', 'jac_start'], 
                                 opts)

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "generated")
    os.makedirs(out_dir, exist_ok=True)
    cg = ca.CodeGenerator('brachistochrone_nlp.c')
    cg.add(func_eval_local)
    cg.add(func_eval_link)
    
    prefix = out_dir + os.sep 
    cg.generate(prefix)
    print(f"C 代码生成成功！")

if __name__ == "__main__":
    generate_segment_code()