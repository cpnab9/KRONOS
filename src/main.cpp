#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <Eigen/Dense>

#include "kronos/solver/fatrop_wrapper.hpp"
#include "kronos/utils/library_loader.hpp"
#include "kronos/utils/interpolator.hpp"
#include "problem_metadata.h" // 由 Python 离线生成的宏和常数

// --- 物理与参考常数 ---
const double R0 = 6378000.0;
const double V_ref = std::sqrt(R0 * 9.81);
const double t_ref = std::sqrt(R0 / 9.81);

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
        // 1. 加载求解器和动力学函数
        auto funcs = kronos::LibraryLoader::load(argv[1]); 
        auto f_cont = kronos::LibraryLoader::load_f_cont(argv[1]);
        kronos::FatropWrapper solver(funcs); 
        
        // 2. 初始化自适应网格参数
        const int N = KRONOS_N;
        const int d = KRONOS_D;
        const int nx = KRONOS_NX;
        const int nu_base = KRONOS_NU_BASE;
        const int nu_total = d * nx + d * nu_base; // 伪谱法下每个区间的总控制维数
        
        std::vector<double> tau = {0.0};
        std::vector<double> tau_u;
        for (int i = 0; i < d; ++i) {
            tau.push_back(KRONOS_TAU_ROOT[i]);
            tau_u.push_back(KRONOS_TAU_ROOT[i]);
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
        // 【核心修复】：为第 0 次迭代生成高质量的物理插值初值
        // ==========================================
        double tf_guess = 1500.0 / t_ref;
        std::vector<double> x0_val = {1.0 + 100000.0/R0, 0.0, 0.0, 7450.0/V_ref, -0.5*M_PI/180.0, 0.0, 0.0, 40.0*M_PI/180.0, tf_guess};
        std::vector<double> xf_val = {1.0 + 25000.0/R0, 12.0*M_PI/180.0, 70.0*M_PI/180.0, 2000.0/V_ref, -10.0*M_PI/180.0, 90.0*M_PI/180.0, 0.0, 15.0*M_PI/180.0, tf_guess};

        int K_total = N + 1;
        initial_guess.clear();
        for (int k = 0; k < K_total; ++k) {
            // 在起始终点之间线性插值作为粗糙初始猜测
            double interp_frac = (K_total > 1) ? (double)k / (K_total - 1) : 0.0;
            std::vector<double> x_curr(nx);
            for (int i = 0; i < nx; ++i) {
                x_curr[i] = x0_val[i] + interp_frac * (xf_val[i] - x0_val[i]);
            }
            
            // 压入当前主节点状态
            initial_guess.insert(initial_guess.end(), x_curr.begin(), x_curr.end());
            
            // 如果不是最后一个节点，压入内部配点状态和控制量
            if (k < K_total - 1) {
                // 配点状态 (d个，用当前主状态占位)
                for (int j = 0; j < d; ++j) {
                    initial_guess.insert(initial_guess.end(), x_curr.begin(), x_curr.end());
                }
                // 配点控制量 (d个) [sigma_rate, alpha_rate, slack_Q, slack_q, slack_n]
                for (int j = 0; j < d; ++j) {
                    std::vector<double> u_guess = {0.0, 0.0, 1e-3, 1e-3, 1e-3}; 
                    initial_guess.insert(initial_guess.end(), u_guess.begin(), u_guess.end());
                }
            }
        }

        // ==========================================
        // 核心自适应迭代循环
        // ==========================================
        for (int iter = 0; iter < max_adapt_iters; ++iter) {
            std::cout << "\n=============================================\n";
            std::cout << "=== 自适应网格迭代 " << iter + 1 << "/" << max_adapt_iters << " ===\n";
            std::cout << "=============================================\n";

            // (1) 注入动态参数和热启动猜测值
            solver.set_parameters(mesh_fractions);
            if (!initial_guess.empty()) {
                solver.set_initial_guess(initial_guess);
            }

            // (2) 执行求解
            solver.solve();
            const auto& sol = solver.get_solution();
            double tf_opt = sol[8]; // tf 位于第一个状态的第 8 个索引

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
                // 误差评估: 在中点 (tau=0.5) 比较多项式导数与物理导数
                // ------------------------------------------------
                Eigen::VectorXd X_poly = X_mat * p_X;
                Eigen::VectorXd U_poly = U_c * p_U;
                Eigen::VectorXd dX_poly_dt = (X_mat * dp_X) / dt_k;

                if (f_cont) {
                    const double* arg[2] = {X_poly.data(), U_poly.data()};
                    Eigen::VectorXd f_eval(nx);
                    double* res[1] = {f_eval.data()};
                    long long int iw = 0; double w = 0; // dummy
                    f_cont(arg, res, &iw, &w, 0);

                    Eigen::VectorXd defect = (dX_poly_dt - f_eval).cwiseAbs();
                    double e_V = defect(3) * V_ref / t_ref;
                    double e_gamma = defect(4) * 180.0 / M_PI / t_ref;
                    errors(k) = e_V + e_gamma;
                }
                
                offset += nx + nu_total;
            }
            
            // 补齐最后一个状态点
            Eigen::Map<const Eigen::VectorXd> x_final(sol.data() + offset, nx);
            t_x_old.push_back(t_nodes_old.back());
            x_old.push_back(std::vector<double>(x_final.data(), x_final.data() + nx));

            double max_error = errors.maxCoeff();
            double mean_error = errors.mean();
            std::cout << "  > 目标函数值: " << std::fixed << std::setprecision(6) << solver.get_objective() << "\n";
            std::cout << "  > 耗时: " << solver.get_solve_time_ms() << " ms\n";
            std::cout << "  > 平均截断误差: " << std::scientific << mean_error << "\n";
            std::cout << "  > 最大截断误差: " << max_error << "\n";

            // 【新增】：NaN 安全拦截器
            if (std::isnan(mean_error)) {
                std::cerr << "\n⚠️ 致命错误：截断误差为 NaN！求解器遇到奇点 (可能求解失败且给出了非物理结果)。\n";
                break;
            }

            if (mean_error < error_tol || iter == max_adapt_iters - 1) {
                std::cout << "\n🎉 满足收敛准则，优化完成！\n";
                // 此处可调用 CSV 导出逻辑
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

                // 插入主状态节点
                auto x_guess = interp_x(t_k);
                initial_guess.insert(initial_guess.end(), x_guess.begin(), x_guess.end());

                // 插入配点状态
                for (int j = 1; j <= d; ++j) {
                    auto xc_guess = interp_x(t_k + tau[j] * dt_k);
                    initial_guess.insert(initial_guess.end(), xc_guess.begin(), xc_guess.end());
                }
                
                // 插入控制量，并安全化松弛变量
                for (int j = 0; j < d; ++j) {
                    auto uc_guess = interp_u(t_k + tau_u[j] * dt_k);
                    uc_guess[2] = std::max(uc_guess[2], 1e-4); // slack_Q
                    uc_guess[3] = std::max(uc_guess[3], 1e-4); // slack_q
                    uc_guess[4] = std::max(uc_guess[4], 1e-4); // slack_n
                    initial_guess.insert(initial_guess.end(), uc_guess.begin(), uc_guess.end());
                }
            }
            // 插入最后一个阶段的主状态
            auto x_final_guess = interp_x(t_nodes_new.back());
            initial_guess.insert(initial_guess.end(), x_final_guess.begin(), x_final_guess.end());
        }

    } catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << std::endl;
        return -1;
    }
    return 0;
}