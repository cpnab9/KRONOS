import casadi as ca
import numpy as np
import os
import shutil
import json  # 【新增】引入 json 库

def generate_point_mass_2d():
    # ... (前面的物理建模和求导逻辑保持完全不变) ...
    nx, nu_real, d, K_intervals, tf = 5, 2, 3, 100, 5.0
    nu = d * nx + d * nu_real  # 21
    ng = d * nx                # 15
    ng_ineq = d * nu_real      # 6

    sym_x = ca.SX.sym('x', nx)
    sym_u = ca.SX.sym('u', nu)
    sym_lam_dyn = ca.SX.sym('lam_dyn', nx)
    sym_lam_eq = ca.SX.sym('lam_eq', ng)
    sym_lam_ineq = ca.SX.sym('lam_ineq', ng_ineq) 

    X_inner = [sym_u[i*nx : (i+1)*nx] for i in range(d)]
    U_inner = [sym_u[d*nx + i*nu_real : d*nx + (i+1)*nu_real] for i in range(d)]

    def f_cont(x, u):
        vx, vy = x[2], x[3]
        fx, fy = u[0], u[1]
        m = 1.0
        return ca.vertcat(vx, vy, fx/m + 0.5*fy**2/m, fy/m, fx**2 + fy**2)

    tau = ca.collocation_points(d, 'radau')
    tau_root = np.append(0, tau)            
    
    C = np.zeros((d+1, d+1))
    for j in range(d+1):
        p = np.poly1d([1])
        for r in range(d+1):
            if r != j: p = np.polymul(p, np.poly1d([1, -tau_root[r]])) / (tau_root[j]-tau_root[r])
        dp = np.polyder(p)
        for i in range(d+1): C[j, i] = dp(tau_root[i])

    h = tf / K_intervals
    g_eq_list = []
    for i in range(1, d+1):
        x_dot_i = C[0, i] * sym_x
        for j in range(1, d+1): x_dot_i += C[j, i] * X_inner[j-1]
        g_eq_list.append(h * f_cont(X_inner[i-1], U_inner[i-1]) - x_dot_i)
        
    g_eq = ca.vertcat(*g_eq_list)
    f_dyn = X_inner[-1] 
    
    g_ineq_list = [U_inner[i] for i in range(d)]
    g_ineq = ca.vertcat(*g_ineq_list)

    ux = ca.vertcat(sym_u, sym_x)
    J_BAbt = ca.densify(ca.jacobian(f_dyn, ux).T)
    J_Ggt  = ca.densify(ca.jacobian(g_eq, ux).T)
    J_Ggt_ineq = ca.densify(ca.jacobian(g_ineq, ux).T) 
    lagrangian = ca.dot(sym_lam_dyn, f_dyn) + ca.dot(sym_lam_eq, g_eq) + ca.dot(sym_lam_ineq, g_ineq)
    H_RSQrqt = ca.densify(ca.hessian(lagrangian, ux)[0])

    # 1. 导出统一的 C 代码 (不需要任何前缀)
    filename = 'casadi_codegen.c'
    cgen = ca.CodeGenerator(filename)
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

    # 2. 【核心新增】将所有的环境参数打包为 JSON 导出
    # 2. 【核心新增】将所有的环境参数打包为 JSON 导出
    fatrop_inf = 1e20
    config_dict = {
        "problem_name": "PointMass2D",
        "K_intervals": K_intervals,
        "nx": nx, "nu": nu, "ng_defects": ng, "ng_ineq": ng_ineq, # <--- 这里把 ng_defects 改成了 ng
        "init_idx": [0, 1, 2, 3, 4],
        "init_val": [0.0, 0.0, 0.0, 0.0, 0.0],
        "term_idx": [0, 1, 2, 3],
        "term_val": [1.0, 2.0, 3.0, 4.0],
        # 将配置写死在 Python 脚本中，C++ 端只负责执行
        "ineq_lower": [-50.0, -100.0, -50.0, -100.0, -50.0, -100.0],
        "ineq_upper": [ 50.0,  100.0,  50.0,  100.0,  50.0,  100.0],
        "obj_state_idx": 4,
        "obj_weight": 20.0, # 1/dt
        "guess_xk": [0.0] * nx,
        "guess_uk": [0.0] * nu
    }

    config_dir = '../config'
    os.makedirs(config_dir, exist_ok=True)
    with open(os.path.join(config_dir, 'ocp_config.json'), 'w') as f:
        json.dump(config_dict, f, indent=4)
        
    print(f"成功生成 C 代码和 ocp_config.json 配置文件！")

if __name__ == "__main__":
    generate_point_mass_2d()