import casadi as cs
import numpy as np
import config as cfg
from dynamics import create_continuous_dynamics
from collocation import create_collocation_function, get_path_constraints
from visualization import extract_and_plot

def main():
    # 1. 建立动力学与配点函数
    f_cont = create_continuous_dynamics()
    colloc_fun = create_collocation_function(f_cont)

    # 2. 装配 Opti 问题与目标函数
    opti = cs.Opti()

    nx_list = [cfg.n_x for _ in range(cfg.K)]
    nu_list = [cfg.d*cfg.n_x + cfg.d*cfg.n_u for _ in range(cfg.K-1)] + [0] 
    ng_list = []

    x = []; u = []
    for k in range(cfg.K):
        x.append(opti.variable(nx_list[k]))
        u.append(opti.variable(nu_list[k]))

    slack_sum_path = 0

    for k in range(cfg.K):
        if k < cfg.K - 1:
            X_next, _ = colloc_fun(x[k], u[k])
            opti.subject_to(x[k+1] == X_next) 
            
            offset_u = cfg.d * cfg.n_x
            for j in range(cfg.d):
                U_cj = u[k][offset_u + j*cfg.n_u : offset_u + (j+1)*cfg.n_u]
                slack_sum_path += (U_cj[1] + U_cj[2] + U_cj[3])
            
        path_constr = get_path_constraints(u[k], x[k], k, colloc_fun, f_cont)
        ng_list.append(0)
        for constr in path_constr:
            ng_list[-1] += constr[1].nnz()
            opti.subject_to((constr[0] <= constr[1]) <= constr[2])

    J_obj = -x[cfg.N][3] * cfg.V_ref + cfg.W_penalty * slack_sum_path
    opti.minimize(J_obj)

    # 3. 初值猜测 (Warm-start)
    tf_guess = 1500.0 / cfg.t_ref 
    x0_guess = np.array([
        1.0 + 100000.0/cfg.R0, 0.0, 0.0, 7450.0/cfg.V_ref, -0.5*np.pi/180.0, 0.0, 0.0, tf_guess])
    xf_guess = np.array([
        1.0 + 25000.0/cfg.R0, 12.0*np.pi/180.0, 70.0*np.pi/180.0, 2000.0/cfg.V_ref, -10.0*np.pi/180.0, 90.0*np.pi/180.0, 0.0, tf_guess])

    for k in range(cfg.K):
        interp_frac = k / (cfg.K - 1) if cfg.K > 1 else 0
        x_curr_guess = x0_guess + interp_frac * (xf_guess - x0_guess)
        
        for i in range(cfg.n_x):
            opti.set_initial(x[k][i], x_curr_guess[i])
            
        if k < cfg.K - 1:
            u_guess = []
            for j in range(cfg.d):
                u_guess.extend(x_curr_guess.tolist())
            for j in range(cfg.d):
                u_guess.extend([0.0, 0.0, 0.0, 0.0])
            opti.set_initial(u[k], u_guess)

    # 4. 配置 Fatrop 求解器并执行
    opti.solver('fatrop', {
        'structure_detection': 'manual', 
        'nx': nx_list, 
        'nu': nu_list, 
        'ng': ng_list, 
        'N': cfg.N, 
        'expand': True, 
        'fatrop.mu_init': 1e-1,
        'fatrop.print_level': 5, 
    })

    print("[*] 开始使用 Fatrop 求解 NLP...")
    try:
        sol = opti.solve()
        print("✅ 优化成功！")
        extract_and_plot(sol, x)
    except Exception as e:
        print("❌ 优化失败，提取当前 debug 值：", e)
        # 如果失败，也可以绘制当前 debug 迭代的结果
        extract_and_plot(opti.debug, x)

if __name__ == "__main__":
    main()