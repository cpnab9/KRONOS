#pragma once

#include "schur_complement.hpp"
#include <vector>
#include <iostream>
#include <cmath>

class IPMCore {
private:
    int K_; // 分段数
    int N_; // 每段节点数
    int n_x_;
    int n_u_;
    
    SchurComplementSolver schur_solver_;
    
    // 存放所有分段的数据
    std::vector<SegmentData> segments_;
    
    // 全局变量
    double T_global_;
    std::vector<Eigen::VectorXd> link_multipliers_; 

    // 辅助函数：计算当前所有局部变量步长的最大范数
    double compute_max_step_norm(const StepData& step) {
        double max_norm = std::abs(step.delta_T);
        for (int i = 0; i < K_; ++i) {
            max_norm = std::max(max_norm, step.delta_Z[i].norm());
        }
        return max_norm;
    }

public:
    // IPMCore(int K, int N, int n_x, int n_u) 
    //     : K_(K), N_(N), n_x_(n_x), n_u_(n_u), 
    //       schur_solver_(K, N, n_x, n_u) {
        
    //     segments_.resize(K_);
    //     link_multipliers_.resize(K_ - 1);
        
    //     int dim_Z = N_ * n_x_ + N_ * n_u_;
    //     int dim_g = N_ * n_x_;
        
    //     T_global_ = 2.0; 
    //     for (int k = 0; k < K_; ++k) {
    //         segments_[k].Z = Eigen::VectorXd::Ones(dim_Z) * 0.1; 
    //         segments_[k].lam = Eigen::VectorXd::Zero(dim_g);
    //     }
    //     for (int k = 0; k < K_ - 1; ++k) {
    //         link_multipliers_[k] = Eigen::VectorXd::Zero(n_x_);
    //     }
    // }

    void solve() {
        std::cout << "==========================================\n";
        std::cout << "      🚀 KRONOS Pseudospectral Planner\n";
        std::cout << "==========================================\n";
        std::cout << "分段数 (K): " << K_ << ", 每段节点数 (N): " << N_ << "\n\n";

        int max_iter = 10; 
        double tol = 1e-6; // 收敛容差
        
        for (int iter = 0; iter < max_iter; ++iter) {
            std::cout << "------------ Iteration " << iter << " ------------\n";
            
            // 步骤 1: 计算搜索方向 (调用舒尔补)
            StepData step = schur_solver_.compute_step(segments_, T_global_, link_multipliers_);
            
            // 步骤 2: 计算步长 (阻尼牛顿法 / 简单的线搜索)
            double alpha_primal = 1.0;
            double alpha_dual = 1.0;
            
            // 如果步长过大，进行截断 (防止非线性发散)
            double max_step = compute_max_step_norm(step);
            if (max_step > 10.0) {
                alpha_primal = 10.0 / max_step;
                alpha_dual = alpha_primal;
                std::cout << "[IPM] 步长过大 (" << max_step << ")，触发截断，alpha = " << alpha_primal << "\n";
            }

            // TODO: 未来在这里加入 Fraction-to-the-boundary 规则
            // 遍历 Z，检查 Z + alpha * delta_Z 是否越界 (如速度 < 0)。若越界，进一步缩小 alpha_primal。

            // 步骤 3: 变量更新 (Update)
            T_global_ += alpha_primal * step.delta_T;
            
            // 防止总时间 T 变为负数 (物理时间不能逆流)
            if (T_global_ < 0.1) T_global_ = 0.1; 

            for (int k = 0; k < K_; ++k) {
                // 原变量 (Primal variables) 更新
                segments_[k].Z += alpha_primal * step.delta_Z[k];
                // 对偶变量 (Dual variables / Multipliers) 更新
                segments_[k].lam += alpha_dual * step.delta_lam[k];
            }

            for (int k = 0; k < K_ - 1; ++k) {
                // Link 乘子更新
                link_multipliers_[k] += alpha_dual * step.delta_link_lam[k];
            }

            if (K_ > 1) {
                std::cout << "      -> Link 乘子步长范数: " << step.delta_link_lam[0].norm() << "\n";
            }
            // ==============================================================

            std::cout << "[IPM] 变量更新完成 | 当前全局时间 T: " << T_global_ << "\n\n";
            
            // 步骤 4: 收敛性检查 (目前仅依靠步长范数作为粗略判据)
            if (max_step < tol) {
                std::cout << "✨ 优化成功收敛！总迭代次数: " << iter + 1 << "\n";
                break;
            }
        }
        
        std::cout << "==========================================\n";
        std::cout << "✅ 优化任务结束！最终时间 T = " << T_global_ << "\n";
    }
};