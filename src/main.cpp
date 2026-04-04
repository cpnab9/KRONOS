// KRONOS/src/main.cpp
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <Eigen/Dense>

#include "kronos/solver/fatrop_wrapper.hpp"
#include "kronos/utils/library_loader.hpp"
#include "kronos/utils/interpolator.hpp"

// 注意：已经彻底移除了 #include "problem_metadata.h" 和所有物理常数宏

// --- 辅助函数：计算拉格朗日基函数在指定点的值与导数 ---
void compute_lagrange_weights(int d, const std::vector<double>& tau, double tau_test, 
                              Eigen::VectorXd& p, Eigen::VectorXd& dp) {
    p = Eigen::VectorXd::Zero(tau.size());
    dp = Eigen::VectorXd::Zero(tau.size());
    for (size_t j = 0; j < tau.size(); ++j) {
        double L = 1.0;
        double dL = 0.0;
        for (size_t m = 0; m < tau.size(); ++m) {
            if (m != j) {
                double term = 1.0;
                for (size_t k = 0; k < tau.size(); ++k) {
                    if (k != j && k != m) term *= (tau_test - tau[k]) / (tau[j] - tau[k]);
                }
                dL += term / (tau[j] - tau[m]);
                L *= (tau_test - tau[m]) / (tau[j] - tau[m]);
            }
        }
        p(j) = L;
        dp(j) = dL;
    }
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <path_to_shared_library>" << std::endl;
        return -1;
    }

    try {
        // 1. 动态加载求解器、动力学函数，以及元数据信息
        auto funcs = kronos::LibraryLoader::load(argv[1]); 
        auto f_cont = kronos::LibraryLoader::load_f_cont(argv[1]);
        auto prob_info = kronos::LibraryLoader::load_problem_info(argv[1]);
        
        kronos::FatropWrapper solver(funcs); 
        
        // 2. 初始化自适应网格参数 (完全由加载的元数据驱动)
        const int N = prob_info.N;
        const int d = prob_info.d;
        const int nx = prob_info.nx;
        const int nu_base = prob_info.nu_base;
        const int nu_total = d * nx + d * nu_base; 
        const int tf_index = prob_info.tf_index;
        
        std::vector<double> tau = {0.0};
        std::vector<double> tau_u;
        for (int i = 0; i < d; ++i) {
            tau.push_back(prob_info.tau_root[i]);
            tau_u.push_back(prob_info.tau_root[i]);
        }

        // 预计算截断误差评估点 (tau = 0.5) 的插值权重
        Eigen::VectorXd p_X, dp_X, p_U, dp_U_dummy;
        compute_lagrange_weights(d, tau, 0.5, p_X, dp_X);
        compute_lagrange_weights(d - 1, tau_u, 0.5, p_U, dp_U_dummy);

        // 网格初始化：均匀分布
        std::vector<double> mesh_fractions(N, 1.0 / N);
        std::vector<double> initial_guess;

        int max_adapt_iters = 10;
        double error_tol = 1e-3;

        // ==========================================
        // 【动态化】：为第 0 次迭代生成基于元数据的物理插值初值
        // ==========================================
        int K_total = N + 1;
        initial_guess.clear();
        for (int k = 0; k < K_total; ++k) {
            // 在起始终点之间线性插值作为粗糙初始猜测
            double interp_frac = (K_total > 1) ? (double)k / (K_total - 1) : 0.0;
            std::vector<double> x_curr(nx);
            for (int i = 0; i < nx; ++i) {
                x_curr[i] = prob_info.x0_guess[i] + interp_frac * (prob_info.xf_guess[i] - prob_info.x0_guess[i]);
            }
            
            initial_guess.insert(initial_guess.end(), x_curr.begin(), x_curr.end());
            
            if (k < K_total - 1) {
                // 配点状态 (d个，用当前主状态占位)
                for (int j = 0; j < d; ++j) {
                    initial_guess.insert(initial_guess.end(), x_curr.begin(), x_curr.end());
                }
                // 配点控制量 (泛化处理，默认填入安全的小正数)
                for (int j = 0; j < d; ++j) {
                    std::vector<double> u_guess(nu_base, 1e-3); 
                    initial_guess.insert(initial_guess.end(), u_guess.begin(), u_guess.end());
                }
            }
        }

        // 映射获取误差权重向量
        Eigen::Map<Eigen::VectorXd> err_weights(prob_info.error_weights.data(), nx);

        // ==========================================
        // 核心自适应迭代循环
        // ==========================================
        for (int iter = 0; iter < max_adapt_iters; ++iter) {
            std::cout << "\n=============================================\n";
            std::cout << "=== Adaptive Mesh Iteration " << iter + 1 << "/" << max_adapt_iters << " ===\n";
            std::cout << "=============================================\n";

            // (1) 注入动态参数和热启动猜测值
            solver.set_parameters(mesh_fractions);
            if (!initial_guess.empty()) {
                solver.set_initial_guess(initial_guess);
            }

            // (2) 执行求解
            solver.solve();
            const auto& sol = solver.get_solution();
            double tf_opt = sol[tf_index]; // 动态定位 tf 的位置

            // (3) 解析解并提取节点与控制历史
            std::vector<double> t_nodes_old = {0.0};
            for (double frac : mesh_fractions) {
                t_nodes_old.push_back(t_nodes_old.back() + frac * tf_opt);
            }

            std::vector<double> t_x_old, t_u_old;
            std::vector<std::vector<double>> x_old, u_old;
            Eigen::VectorXd errors = Eigen::VectorXd::Zero(N);

            int offset = 0;
            for (int k = 0; k < N; ++k) {
                double dt_k = mesh_fractions[k] * tf_opt;
                
                // 使用 Eigen 映射解向量 (零拷贝)
                Eigen::Map<const Eigen::VectorXd> x_k(sol.data() + offset, nx);
                Eigen::Map<const Eigen::MatrixXd> X_c(sol.data() + offset + nx, nx, d);
                Eigen::Map<const Eigen::MatrixXd> U_c(sol.data() + offset + nx + d * nx, nu_base, d);
                
                Eigen::MatrixXd X_mat(nx, d + 1);
                X_mat.col(0) = x_k;
                X_mat.rightCols(d) = X_c;

                // 记录用于插值的稠密轨迹
                t_x_old.push_back(t_nodes_old[k]);
                x_old.push_back(std::vector<double>(x_k.data(), x_k.data() + nx));
                
                for (int j = 0; j < d; ++j) {
                    t_x_old.push_back(t_nodes_old[k] + tau[j + 1] * dt_k);
                    Eigen::VectorXd X_cj = X_c.col(j);
                    x_old.push_back(std::vector<double>(X_cj.data(), X_cj.data() + nx));
                    
                    t_u_old.push_back(t_nodes_old[k] + tau_u[j] * dt_k);
                    Eigen::VectorXd U_cj = U_c.col(j);
                    u_old.push_back(std::vector<double>(U_cj.data(), U_cj.data() + nu_base));
                }

                // ------------------------------------------------
                // 误差评估: 泛化计算逻辑，利用 Python 端传入的权重
                // ------------------------------------------------
                Eigen::VectorXd X_poly = X_mat * p_X;
                Eigen::VectorXd U_poly = U_c * p_U;
                Eigen::VectorXd dX_poly_dt = (X_mat * dp_X) / dt_k;

                if (f_cont) {
                    const double* arg[2] = {X_poly.data(), U_poly.data()};
                    Eigen::VectorXd f_eval(nx);
                    double* res[1] = {f_eval.data()};
                    long long int iw = 0; double w = 0; 
                    f_cont(arg, res, &iw, &w, 0);

                    Eigen::VectorXd defect = (dX_poly_dt - f_eval).cwiseAbs();
                    
                    // 【消除硬编码】：点乘误差权重向量并求和，取代了之前手动指定 defect(3) 和 defect(4)
                    errors(k) = (defect.array() * err_weights.array()).sum();
                }
                
                offset += nx + nu_total;
            }
            
            // 补齐最后一个状态点
            Eigen::Map<const Eigen::VectorXd> x_final(sol.data() + offset, nx);
            t_x_old.push_back(t_nodes_old.back());
            x_old.push_back(std::vector<double>(x_final.data(), x_final.data() + nx));

            double max_error = errors.maxCoeff();
            double mean_error = errors.mean();
            std::cout << "  > Objective Function: " << std::fixed << std::setprecision(6) << solver.get_objective() << "\n";
            std::cout << "  > Solver Time: " << solver.get_solve_time_ms() << " ms\n";
            std::cout << "  > Mean Truncation Error: " << std::scientific << mean_error << "\n";
            std::cout << "  > Max Truncation Error: " << max_error << "\n";

            if (std::isnan(mean_error)) {
                std::cerr << "\n⚠️ Fatal Error: Truncation error is NaN! Solver encountered singularity.\n";
                break;
            }

            if (mean_error < error_tol || iter == max_adapt_iters - 1) {
                std::cout << "\n🎉 Optimization converged successfully!\n";
                break;
            }

            // (4) 网格重分配策略
            double eps = 1e-2;
            Eigen::VectorXd monitor = errors.cwiseSqrt().array() + eps;
            Eigen::VectorXd desired_dt = 1.0 / monitor.array();
            Eigen::VectorXd desired_fractions = desired_dt / desired_dt.sum();
            
            Eigen::VectorXd new_frac_eigen = 0.5 * Eigen::Map<Eigen::VectorXd>(mesh_fractions.data(), N) + 0.5 * desired_fractions;
            new_frac_eigen /= new_frac_eigen.sum();
            
            std::vector<double> t_nodes_new = {0.0};
            for (int k = 0; k < N; ++k) {
                mesh_fractions[k] = new_frac_eigen(k);
                t_nodes_new.push_back(t_nodes_new.back() + mesh_fractions[k] * tf_opt);
            }

            // (5) 热启动插值 (Warm-start Generation)
            kronos::Interpolator1D interp_x(t_x_old, x_old);
            kronos::Interpolator1D interp_u(t_u_old, u_old);

            initial_guess.clear();
            for (int k = 0; k < N; ++k) {
                double dt_k = mesh_fractions[k] * tf_opt;
                double t_k = t_nodes_new[k];

                auto x_guess = interp_x(t_k);
                initial_guess.insert(initial_guess.end(), x_guess.begin(), x_guess.end());

                for (int j = 1; j <= d; ++j) {
                    auto xc_guess = interp_x(t_k + tau[j] * dt_k);
                    initial_guess.insert(initial_guess.end(), xc_guess.begin(), xc_guess.end());
                }
                
                // 【动态化】：不再硬编码松弛变量索引，给所有的控制分量保底 1e-4 的正值，防止计算奇点
                for (int j = 0; j < d; ++j) {
                    auto uc_guess = interp_u(t_k + tau_u[j] * dt_k);
                    for (auto& u_val : uc_guess) {
                        u_val = std::max(u_val, 1e-4);
                    }
                    initial_guess.insert(initial_guess.end(), uc_guess.begin(), uc_guess.end());
                }
            }
            auto x_final_guess = interp_x(t_nodes_new.back());
            initial_guess.insert(initial_guess.end(), x_final_guess.begin(), x_final_guess.end());
        }

    } catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << std::endl;
        return -1;
    }
    return 0;
}