import casadi as ca
import numpy as np
import os
import shutil  

def generate_casadi_c_code():
    nx = 4          # 状态: [x, y, v, tf]
    nu_real = 1     # 控制: [theta]
    d = 3           # 3阶 Radau
    K_intervals = 20
    
    nu = d * nx + d * nu_real  # 15
    ng = d * nx                # 12

    sym_x = ca.SX.sym('x', nx)
    sym_u = ca.SX.sym('u', nu)
    sym_lam_dyn = ca.SX.sym('lam_dyn', nx)
    sym_lam_eq = ca.SX.sym('lam_eq', ng)

    X_inner =[sym_u[i*nx : (i+1)*nx] for i in range(d)]
    U_inner = [sym_u[d*nx + i*nu_real : d*nx + (i+1)*nu_real] for i in range(d)]

    def f_cont(x, u):
        v, tf, theta = x[2], x[3], u[0]
        g = 9.81
        return ca.vertcat(v * ca.cos(theta), v * ca.sin(theta), g * ca.sin(theta), 0.0)

    tau = ca.collocation_points(d, 'radau')
    tau_root = np.append(0, tau)            
    
    C = np.zeros((d+1, d+1))
    for j in range(d+1):
        p = np.poly1d([1])
        for r in range(d+1):
            if r != j: p = np.polymul(p, np.poly1d([1, -tau_root[r]])) / (tau_root[j]-tau_root[r])
        dp = np.polyder(p)
        for i in range(d+1): C[j, i] = dp(tau_root[i])

    h = sym_x[3] / K_intervals
    g_eq_list =[]
    for i in range(1, d+1):
        x_dot_i = C[0, i] * sym_x
        for j in range(1, d+1): x_dot_i += C[j, i] * X_inner[j-1]
        g_eq_list.append(h * f_cont(X_inner[i-1], U_inner[i-1]) - x_dot_i)
        
    g_eq = ca.vertcat(*g_eq_list)
    f_dyn = X_inner[-1] 

    ux = ca.vertcat(sym_u, sym_x)
    J_BAbt = ca.densify(ca.jacobian(f_dyn, ux).T)
    J_Ggt  = ca.densify(ca.jacobian(g_eq, ux).T)
    lagrangian = ca.dot(sym_lam_dyn, f_dyn) + ca.dot(sym_lam_eq, g_eq)
    H_RSQrqt = ca.densify(ca.hessian(lagrangian, ux)[0])

    # ---- 6. 导出 C 代码 ----
    filename = 'casadi_codegen.c'
    cgen = ca.CodeGenerator(filename) # 只传纯文件名，不带路径
    cgen.add(ca.Function('eval_f_dyn',[sym_x, sym_u],[f_dyn]))
    cgen.add(ca.Function('eval_g_eq',[sym_x, sym_u], [g_eq]))
    cgen.add(ca.Function('eval_J_BAbt', [sym_x, sym_u],[J_BAbt]))
    cgen.add(ca.Function('eval_J_Ggt',[sym_x, sym_u],[J_Ggt]))
    cgen.add(ca.Function('eval_H_RSQrqt',[sym_x, sym_u, sym_lam_dyn, sym_lam_eq],[H_RSQrqt]))
    
    # 这会在当前运行 python 的目录下生成 casadi_codegen.c
    cgen.generate() 

    # ---- 7. 移动文件到 C++ 源码目录 ----
    out_dir = '../src/codegen'
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, filename)
    
    # 移动并覆盖已有文件
    shutil.move(filename, out_file)
    print(f"成功生成并移动代码至: {out_file}")

if __name__ == "__main__":
    generate_casadi_c_code()