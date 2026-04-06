import casadi as ca
import numpy as np
import os
import shutil
import json

def generate_brachistochrone():
    print(">>> 开始生成最速降线 (Brachistochrone) 算例...")
    
    # 1. 定义问题维度
    nx = 4          # 状态: [x, y, v, tf] (水平位置，垂直位置，速度，总时间)
    nu_real = 1     # 控制: [theta] (切线与水平方向的夹角)
    d = 3           # 3阶 Radau 伪谱法
    K_intervals = 50 # 使用 50 个区间以获得更平滑的曲线
    
    nu = d * nx + d * nu_real  # 15
    ng = d * nx                # 12
    # 我们将所有的状态和控制量都放入不等式约束中，以便灵活设置边界
    ng_ineq = nx + nu          # 19

    # 2. 声明 CasADi 符号变量
    sym_x = ca.SX.sym('x', nx)
    sym_u = ca.SX.sym('u', nu)
    sym_lam_dyn = ca.SX.sym('lam_dyn', nx)
    sym_lam_eq = ca.SX.sym('lam_eq', ng)
    sym_lam_ineq = ca.SX.sym('lam_ineq', ng_ineq) 

    X_inner =[sym_u[i*nx : (i+1)*nx] for i in range(d)]
    U_inner = [sym_u[d*nx + i*nu_real : d*nx + (i+1)*nu_real] for i in range(d)]

    # 3. 连续动力学方程
    def f_cont(x, u):
        v, tf, theta = x[2], x[3], u[0]
        g = 9.81
        # dx/dt = v*cos(theta), dy/dt = v*sin(theta), dv/dt = g*sin(theta), d(tf)/dt = 0
        # (假设 y 轴正方向朝下，重力也朝下)
        return ca.vertcat(v * ca.cos(theta), v * ca.sin(theta), g * ca.sin(theta), 0.0)

    # 4. Radau 配点法矩阵计算
    tau = ca.collocation_points(d, 'radau')
    tau_root = np.append(0, tau)            
    
    C = np.zeros((d+1, d+1))
    for j in range(d+1):
        p = np.poly1d([1])
        for r in range(d+1):
            if r != j: p = np.polymul(p, np.poly1d([1, -tau_root[r]])) / (tau_root[j]-tau_root[r])
        dp = np.polyder(p)
        for i in range(d+1): C[j, i] = dp(tau_root[i])

    # 5. 构建等式与不等式缺陷
    h = sym_x[3] / K_intervals  # 步长由自由变量 tf 决定
    g_eq_list = []
    for i in range(1, d+1):
        x_dot_i = C[0, i] * sym_x
        for j in range(1, d+1): x_dot_i += C[j, i] * X_inner[j-1]
        g_eq_list.append(h * f_cont(X_inner[i-1], U_inner[i-1]) - x_dot_i)
        
    g_eq = ca.vertcat(*g_eq_list)
    f_dyn = X_inner[-1] 
    g_ineq = ca.vertcat(sym_x, sym_u)

    # 6. 精确求导与拉格朗日海森矩阵
    ux = ca.vertcat(sym_u, sym_x)
    J_BAbt = ca.densify(ca.jacobian(f_dyn, ux).T)
    J_Ggt  = ca.densify(ca.jacobian(g_eq, ux).T)
    J_Ggt_ineq = ca.densify(ca.jacobian(g_ineq, ux).T) 
    lagrangian = ca.dot(sym_lam_dyn, f_dyn) + ca.dot(sym_lam_eq, g_eq) + ca.dot(sym_lam_ineq, g_ineq)
    H_RSQrqt = ca.densify(ca.hessian(lagrangian, ux)[0])

    # ================== 导出 C 代码 ==================
    filename = 'casadi_codegen.c'
    cgen = ca.CodeGenerator(filename)
    # 不加任何前缀，保持通用接口
    cgen.add(ca.Function('eval_f_dyn',[sym_x, sym_u],[f_dyn]))
    cgen.add(ca.Function('eval_g_eq',[sym_x, sym_u], [g_eq]))
    cgen.add(ca.Function('eval_g_ineq',[sym_x, sym_u], [g_ineq]))
    cgen.add(ca.Function('eval_J_BAbt', [sym_x, sym_u],[J_BAbt]))
    cgen.add(ca.Function('eval_J_Ggt',[sym_x, sym_u],[J_Ggt]))
    cgen.add(ca.Function('eval_J_Ggt_ineq',[sym_x, sym_u],[J_Ggt_ineq]))
    cgen.add(ca.Function('eval_H_RSQrqt',[sym_x, sym_u, sym_lam_dyn, sym_lam_eq, sym_lam_ineq],[H_RSQrqt]))
    cgen.generate() 

    out_dir = '../src/codegen'
    os.makedirs(out_dir, exist_ok=True)
    shutil.move(filename, os.path.join(out_dir, filename))

    # ================== 导出 JSON 配置 ==================
    fatrop_inf = 1e20
    ineq_lower = [-fatrop_inf] * ng_ineq
    ineq_upper = [fatrop_inf] * ng_ineq

    # 关键约束：总时间 tf 必须大于 0 (tf 在状态中的索引为 3)
    ineq_lower[3] = 0.1
    ineq_upper[3] = 20.0

    config_dict = {
        "problem_name": "Brachistochrone",
        "K_intervals": K_intervals,
        "nx": nx, "nu": nu, "ng_defects": ng, "ng_ineq": ng_ineq,
        # 起点: x=0, y=0, v=0
        "init_idx": [0, 1, 2],
        "init_val": [0.0, 0.0, 0.0],
        # 终点: x=10, y=5 (满足你的下降要求)
        "term_idx": [0, 1],
        "term_val": [10.0, 5.0],
        "ineq_lower": ineq_lower,
        "ineq_upper": ineq_upper,
        # 目标函数：最小化时间状态 tf (索引为3)
        "obj_state_idx": 3,
        "obj_weight": 1.0,
        # 良好的初值猜想能帮助内点法更快收敛
        "guess_xk": [5.0, 2.5, 5.0, 2.0], 
        "guess_uk": [0.5] * nu
    }

    config_dir = '../config'
    os.makedirs(config_dir, exist_ok=True)
    with open(os.path.join(config_dir, 'ocp_config.json'), 'w') as f:
        json.dump(config_dict, f, indent=4)
        
    print(f"✅ 成功生成通用 C 代码和 JSON 配置文件！")

if __name__ == "__main__":
    generate_brachistochrone()