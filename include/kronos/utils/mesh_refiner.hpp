// KRONOS/include/kronos/utils/mesh_refiner.hpp
#pragma once
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <iostream>
#include "kronos/solver/fatrop_wrapper.hpp"

namespace kronos {

class MeshRefiner {
public:
    struct RefineResult {
        std::vector<double> new_fractions;
        std::vector<double> new_initial_guess;
    };

    static constexpr int N = 20; 
    static constexpr int d = 4;
    static constexpr int nx = 9;   // [r, theta, phi, V, gamma, psi, sigma, alpha, tf]
    static constexpr int nu = 5;   // [sigma_rate, alpha_rate, slack_Q, slack_q, slack_n]
    static constexpr int nu_colloc = d * nx + d * nu; 

    // 构建第一轮的物理初值猜测 (严格对齐 Python main.py 里的 1e-3 松弛量)
    static std::vector<double> build_initial_guess() {
        const double PI = 3.14159265358979323846;
        double R0 = 6378000.0;
        double g0 = 9.81;
        double V_ref = std::sqrt(R0 * g0);
        double t_ref = std::sqrt(R0 / g0);
        double tf_guess = 1500.0 / t_ref;

        std::vector<double> x0_val = {
            1.0 + 100000.0/R0, 0.0, 0.0, 7450.0/V_ref, -0.5*PI/180.0, 0.0, 0.0, 40.0*PI/180.0, tf_guess
        };
        std::vector<double> xf_val = {
            1.0 + 25000.0/R0, 12.0*PI/180.0, 70.0*PI/180.0, 2000.0/V_ref, -10.0*PI/180.0, 90.0*PI/180.0, 0.0, 15.0*PI/180.0, tf_guess
        };

        std::vector<double> guess;
        for (int k = 0; k < N + 1; ++k) {
            double frac = static_cast<double>(k) / N;
            std::vector<double> x_curr(nx);
            for (int i = 0; i < nx; ++i) {
                x_curr[i] = x0_val[i] + frac * (xf_val[i] - x0_val[i]);
            }
            
            guess.insert(guess.end(), x_curr.begin(), x_curr.end());

            if (k < N) {
                for (int j = 0; j < d; ++j) {
                    guess.insert(guess.end(), x_curr.begin(), x_curr.end());
                }
                // 【核心修复1】：松弛变量初始化为 1e-3，坚决不能是 0.0！
                for (int j = 0; j < d; ++j) {
                    guess.insert(guess.end(), {0.0, 0.0, 1e-3, 1e-3, 1e-3});
                }
            }
        }
        return guess;
    }

    static std::vector<double> interp1d(const std::vector<double>& t_nodes, const std::vector<std::vector<double>>& y_nodes, double t_query) {
        if (t_query <= t_nodes.front()) return y_nodes.front();
        if (t_query >= t_nodes.back()) return y_nodes.back();

        auto it = std::lower_bound(t_nodes.begin(), t_nodes.end(), t_query);
        int idx = std::distance(t_nodes.begin(), it);
        double t0 = t_nodes[idx - 1];
        double t1 = t_nodes[idx];
        double weight = (t_query - t0) / (t1 - t0);

        std::vector<double> result(y_nodes[0].size());
        for (size_t i = 0; i < result.size(); ++i) {
            result[i] = y_nodes[idx - 1][i] + weight * (y_nodes[idx][i] - y_nodes[idx - 1][i]);
        }
        return result;
    }

    // 网格重构与热启动初值提取
    static RefineResult compute(const std::vector<double>& old_fractions, const std::vector<double>& flat_solution) {
        double tf_opt = flat_solution[8]; 
        std::vector<double> t_nodes_old = {0.0};
        for (double frac : old_fractions) t_nodes_old.push_back(t_nodes_old.back() + frac * tf_opt);

        std::vector<std::vector<double>> x_series;
        std::vector<std::vector<double>> u_series;
        std::vector<double> t_x_series;
        
        int offset = 0;
        for (int k = 0; k < N; ++k) {
            std::vector<double> x_k(flat_solution.begin() + offset, flat_solution.begin() + offset + nx);
            x_series.push_back(x_k);
            t_x_series.push_back(t_nodes_old[k]);
            
            offset += nx;
            offset += d * nx; // 跳过配点状态 Xc
            
            // 提取配点的第一个控制量 U_c0 用于构建控制插值
            std::vector<double> u_k(flat_solution.begin() + offset, flat_solution.begin() + offset + nu);
            u_series.push_back(u_k);
            
            offset += d * nu; 
        }
        x_series.push_back(std::vector<double>(flat_solution.begin() + offset, flat_solution.begin() + offset + nx));
        u_series.push_back(u_series.back()); // 终端补齐
        t_x_series.push_back(t_nodes_old.back());

        // 计算网格误差 Monitor function 
        std::vector<double> desired_dt(N, 0.0);
        double dt_sum = 0.0;
        for (int k = 0; k < N; ++k) {
            double v_k = x_series[k][3];
            double v_k1 = x_series[k+1][3];
            double gamma_k = x_series[k][4];
            double gamma_k1 = x_series[k+1][4];
            
            double err_approx = std::abs(v_k1 - v_k) * 7900.0 + std::abs(gamma_k1 - gamma_k) * 50.0; 
            double monitor = std::sqrt(err_approx) + 0.05; 
            desired_dt[k] = 1.0 / monitor;
            dt_sum += desired_dt[k];
        }

        std::vector<double> new_fractions(N);
        double new_sum = 0.0;
        for (int k = 0; k < N; ++k) {
            new_fractions[k] = 0.5 * old_fractions[k] + 0.5 * (desired_dt[k] / dt_sum);
            new_sum += new_fractions[k];
        }
        for (double& f : new_fractions) f /= new_sum; 

        // 生成网格重分布后的 Warm-Start
        std::vector<double> t_nodes_new = {0.0};
        for (double frac : new_fractions) t_nodes_new.push_back(t_nodes_new.back() + frac * tf_opt);

        std::vector<double> new_flat_x0;
        for (int k = 0; k < N; ++k) {
            // 状态插值
            auto x_interp = interp1d(t_x_series, x_series, t_nodes_new[k]);
            new_flat_x0.insert(new_flat_x0.end(), x_interp.begin(), x_interp.end());

            for (int j = 0; j < d; ++j) {
                new_flat_x0.insert(new_flat_x0.end(), x_interp.begin(), x_interp.end());
            }
            
            // 【核心修复2】：控制量插值与 Slack 边界保护 (对齐 Python main.py)
            auto u_interp = interp1d(t_x_series, u_series, t_nodes_new[k]);
            u_interp[2] = std::max(u_interp[2], 1e-4); // slack_Q
            u_interp[3] = std::max(u_interp[3], 1e-4); // slack_q
            u_interp[4] = std::max(u_interp[4], 1e-4); // slack_n
            
            for (int j = 0; j < d; ++j) {
                new_flat_x0.insert(new_flat_x0.end(), u_interp.begin(), u_interp.end());
            }
        }
        auto x_last = interp1d(t_x_series, x_series, t_nodes_new.back());
        new_flat_x0.insert(new_flat_x0.end(), x_last.begin(), x_last.end());

        return {new_fractions, new_flat_x0};
    }
};

} // namespace kronos