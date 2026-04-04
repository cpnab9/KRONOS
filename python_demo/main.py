# --- START OF FILE main.py ---
import casadi as cs
import numpy as np
import config as cfg
from dynamics import create_continuous_dynamics
from collocation import create_collocation_function, get_path_constraints
from visualization import extract_and_plot
from adaptive_mesh import compute_mesh_and_warmstart
from error_analysis import compute_and_plot_integration_error

def main():
    f_cont = create_continuous_dynamics()
    colloc_fun = create_collocation_function(f_cont)

    max_adapt_iters = cfg.max_adapt_iters  
    
    mesh_fractions = np.ones(cfg.N) / cfg.N 
    warm_x = None
    warm_u = None

    for iter_idx in range(max_adapt_iters):
        print(f"\n=============================================")
        print(f"=== 自适应网格迭代 {iter_idx+1}/{max_adapt_iters} ===")
        print(f"=============================================")

        opti = cs.Opti()
        nx_list = [cfg.n_x for _ in range(cfg.K)]
        nu_list = [cfg.d*cfg.n_x + cfg.d*cfg.n_u for _ in range(cfg.K-1)] + [0] 
        ng_list = []

        x = []; u = []
        for k in range(cfg.K):
            x.append(opti.variable(nx_list[k]))
            u.append(opti.variable(nu_list[k]))

        slack_sum_path = 0
        u_rate_sq_sum = 0  

        for k in range(cfg.K):
            if k < cfg.K - 1:
                tf_var = x[k][8] # 【修改1】：tf 的索引修正为 8
                dt_k = tf_var * mesh_fractions[k]
                
                X_next, _ = colloc_fun(x[k], u[k], dt_k)
                opti.subject_to(x[k+1] == X_next) 
                
                offset_u = cfg.d * cfg.n_x
                for j in range(cfg.d):
                    U_cj = u[k][offset_u + j*cfg.n_u : offset_u + (j+1)*cfg.n_u]
                    
                    # 【修改2】：惩罚项中同时加上倾侧角速率和攻角速率，促使气动操作平稳
                    sigma_rate = U_cj[0]
                    alpha_rate = U_cj[1]
                    u_rate_sq_sum += (sigma_rate**2 + alpha_rate**2) * dt_k  
                    
                    slack_sum_path += (U_cj[2] + U_cj[3] + U_cj[4])
                
                path_constr = get_path_constraints(u[k], x[k], k, colloc_fun, f_cont, dt_k)
            else:
                path_constr = get_path_constraints(None, x[k], k, colloc_fun, f_cont, None)
                
            ng_list.append(0)
            for constr in path_constr:
                ng_list[-1] += constr[1].nnz()
                opti.subject_to((constr[0] <= constr[1]) <= constr[2])

        J_obj = -x[cfg.N][3] * cfg.V_ref + cfg.W_penalty * slack_sum_path + 100.0 * u_rate_sq_sum
        opti.minimize(J_obj)

        if iter_idx == 0:
            tf_guess = 1500.0 / cfg.t_ref 
            # 【修改3】：猜测值增加一项 alpha 的过度值
            x0_guess = np.array([1.0 + 100000.0/cfg.R0, 0.0, 0.0, 7450.0/cfg.V_ref, -0.5*np.pi/180.0, 0.0, 0.0, 40.0*np.pi/180.0, tf_guess])
            xf_guess = np.array([1.0 + 25000.0/cfg.R0, 12.0*np.pi/180.0, 70.0*np.pi/180.0, 2000.0/cfg.V_ref, -10.0*np.pi/180.0, 90.0*np.pi/180.0, 0.0, 15.0*np.pi/180.0, tf_guess])

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
                        # 【修改4】：控制初值补齐 0.0 (对应 alpha_rate)
                        u_guess.extend([0.0, 0.0, 1e-3, 1e-3, 1e-3])
                    opti.set_initial(u[k], u_guess)
        else:
            for k in range(cfg.K):
                opti.set_initial(x[k], warm_x[k])
                if k < cfg.K - 1:
                    u_warm_safe = np.array(warm_u[k])
                    offset_u = cfg.d * cfg.n_x
                    for j in range(cfg.d):
                        # 【修改5】：松弛变量的索引自 +2 起算
                        idx_slack_Q = offset_u + j*cfg.n_u + 2
                        idx_slack_q = offset_u + j*cfg.n_u + 3
                        idx_slack_n = offset_u + j*cfg.n_u + 4
                        
                        u_warm_safe[idx_slack_Q] = max(u_warm_safe[idx_slack_Q], 1e-4)
                        u_warm_safe[idx_slack_q] = max(u_warm_safe[idx_slack_q], 1e-4)
                        u_warm_safe[idx_slack_n] = max(u_warm_safe[idx_slack_n], 1e-4)
                        
                    opti.set_initial(u[k], u_warm_safe.tolist())

        opti.solver('fatrop', {
            'structure_detection': 'auto', 
            'expand': True, 
            'fatrop.mu_init': 1e-1 if iter_idx == 0 else 1e-3, 
            'fatrop.print_level': 5, 
        })

        # === 在 main.py 中修改 try...except 代码块 ===
        try:
            sol = opti.solve()
            print(f"✅ 第 {iter_idx+1} 次迭代优化成功！")
            
            # --- 这是上一轮我们讨论的提取收敛误差和更新网格的代码 ---
            mesh_fractions_new, warm_x, warm_u, mean_error = compute_mesh_and_warmstart(sol, x, u, f_cont, mesh_fractions)
            error_tol = 1e-3  
            
            # 判断是否收敛 或 达到最大迭代次数
            if mean_error < error_tol or iter_idx == max_adapt_iters - 1:
                if mean_error < error_tol:
                    print(f"\n🎉 满足收敛准则，提前结束自适应迭代！")
                else:
                    print(f"\n⚠️ 达到最大迭代次数。")
                    
                print("================ 最终结果 ================")
                
                # 【修改处】：调用积分验证函数，传入当前的 mesh_fractions (而不是 _new)
                compute_and_plot_integration_error(sol, x, u, f_cont, mesh_fractions)
                
                extract_and_plot(sol, x)
                break
                
            # 若没收敛，将新的网格比例赋值给下一轮
            mesh_fractions = mesh_fractions_new
            print("新的时间段比例:", np.round(mesh_fractions, 3))

        except Exception as e:
            print("❌ 优化失败：", e)
            extract_and_plot(opti.debug, x)
            break

if __name__ == "__main__":
    main()
# --- END OF FILE main.py ---