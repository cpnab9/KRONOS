#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>
#include <iostream>
#include "../utils/casadi_wrapper.hpp"

// ==============================================================================
// 1. 数据结构定义
// ==============================================================================
struct SegmentData {
    Eigen::VectorXd Z;       // 局部变量 [X, U]
    Eigen::VectorXd lam;     // 局部动力学约束乘子
};

struct StepData {
    std::vector<Eigen::VectorXd> delta_Z;   // 局部原变量步长
    std::vector<Eigen::VectorXd> delta_lam; // 局部动力学乘子步长
    double delta_T;                         // 全局时间步长
    std::vector<Eigen::VectorXd> delta_link_lam; // 段间连接点乘子步长
};

// ==============================================================================
// 2. 舒尔补求解器核心类
// ==============================================================================
class SchurComplementSolver {
private:
    int num_segments_;
    int N_;           // 每段节点数
    int n_x_;         // 状态维度
    int dim_Z_;       // 局部变量 Z 维度
    int dim_g_;       // 局部约束维度
    int dim_x_end_;   // 边界状态维度

    CasadiWrapper nlp_;

public:
    SchurComplementSolver(int K, int N, int n_x, int n_u) 
        : num_segments_(K), N_(N), n_x_(n_x) {
        dim_Z_ = N * n_x + N * n_u;
        dim_g_ = N * n_x;
        dim_x_end_ = n_x;
    }

    StepData compute_step(const std::vector<SegmentData>& segments, 
                          double T_global, 
                          const std::vector<Eigen::VectorXd>& x_link_multipliers) {
        
        // ----------------------------------------------------------------------
        // [阶段 1] 全局系统预分配与 Link 缝合初始化
        // ----------------------------------------------------------------------
        int dim_global = 1 + (num_segments_ - 1) * dim_x_end_;
        Eigen::MatrixXd S = Eigen::MatrixXd::Zero(dim_global, dim_global);
        Eigen::VectorXd r_tilde = Eigen::VectorXd::Zero(dim_global);

        // 处理全局时间 T 的对数障碍函数与极小化梯度
        double mu = 0.1; // 障碍参数
        r_tilde(0) = -(1.0 - mu / T_global); 
        S(0, 0) += mu / (T_global * T_global); 

        // 提前评估所有的段间连续性约束，放入全局右端项
        std::vector<LinkEvalResult> link_results(num_segments_ - 1);
        for (int k = 0; k < num_segments_ - 1; ++k) {
            Eigen::VectorXd x_end_k = segments[k].Z.segment((N_ - 1) * n_x_, n_x_);
            Eigen::VectorXd x_start_kp1 = segments[k+1].Z.segment(0, n_x_);
            
            link_results[k] = nlp_.evaluate_link(x_end_k, x_start_kp1);
            
            int global_idx = 1 + k * n_x_;
            r_tilde.segment(global_idx, n_x_) = -link_results[k].g_link;
        }

        std::vector<Eigen::LDLT<Eigen::MatrixXd>> ldlt_cache(num_segments_);
        std::vector<Eigen::MatrixXd> B_cache(num_segments_);
        std::vector<Eigen::VectorXd> r_cache(num_segments_);

        // ----------------------------------------------------------------------
        // [阶段 2] 局部边界设置、KKT 构建与前向消元
        // ----------------------------------------------------------------------
        // 物理边界条件定义
        Eigen::VectorXd w_start_on = Eigen::VectorXd::Ones(n_x_);
        Eigen::VectorXd w_end_on = Eigen::VectorXd::Ones(n_x_);
        w_end_on(2) = 0.0; // 终点速度自由，权重设为 0
        Eigen::VectorXd w_off = Eigen::VectorXd::Zero(n_x_);
        
        Eigen::VectorXd x_start_ref = Eigen::VectorXd::Zero(n_x_); // 起点 (0, 0, 0)
        Eigen::VectorXd x_end_ref = Eigen::VectorXd::Zero(n_x_);
        x_end_ref(0) = 10.0; // 终点 X = 10
        x_end_ref(1) = -5.0; // 终点 Y = -5

        for (int i = 0; i < num_segments_; ++i) {
            // 根据当前段开启或关闭边界锚定
            Eigen::VectorXd w_s = (i == 0) ? w_start_on : w_off;
            Eigen::VectorXd w_e = (i == num_segments_ - 1) ? w_end_on : w_off;

            // 调用 CasADi 获取局部矩阵与完美梯度
            LocalEvalResult local_res = nlp_.evaluate_local(
                segments[i].Z, T_global, segments[i].lam, 
                mu, w_s, x_start_ref, w_e, x_end_ref
            );
            
            // 组装局部 KKT
            int dim_K = dim_Z_ + dim_g_;
            Eigen::MatrixXd K_i = Eigen::MatrixXd::Zero(dim_K, dim_K);
            
            K_i.topLeftCorner(dim_Z_, dim_Z_) = local_res.hess_L_Z; 
            Eigen::MatrixXd J_dense = local_res.jac_g_Z;
            K_i.bottomLeftCorner(dim_g_, dim_Z_) = J_dense;
            K_i.topRightCorner(dim_Z_, dim_g_) = J_dense.transpose();

            // 正则化 (防止纯边界点或不可行初始点的数值奇异)
            K_i.topLeftCorner(dim_Z_, dim_Z_).diagonal().array() += 1e-6;     
            K_i.bottomRightCorner(dim_g_, dim_g_).diagonal().array() -= 1e-6; 

            // 构建局部右端项
            Eigen::VectorXd r_i = Eigen::VectorXd::Zero(dim_K);
            Eigen::VectorXd grad_L_Z = local_res.grad_L_Z; // 提取 CasADi 计算的完美一阶梯度

            // 将段间连接乘子对梯度的拉扯力加进去
            if (i < num_segments_ - 1) { 
                Eigen::MatrixXd J_end = link_results[i].jac_end;
                grad_L_Z.segment((N_ - 1) * n_x_, n_x_) += J_end.transpose() * x_link_multipliers[i];
            }
            if (i > 0) { 
                Eigen::MatrixXd J_start = link_results[i-1].jac_start;
                grad_L_Z.segment(0, n_x_) += J_start.transpose() * x_link_multipliers[i-1];
            }

            r_i.head(dim_Z_) = -grad_L_Z;
            r_i.tail(dim_g_) = -local_res.g_defects; 

            // 分解局部 KKT 矩阵
            ldlt_cache[i].compute(K_i);
            
            // 构建耦合矩阵 B_i
            Eigen::MatrixXd B_i = Eigen::MatrixXd::Zero(dim_K, dim_global);
            Eigen::MatrixXd J_T_dense = local_res.jac_g_T;
            B_i.bottomLeftCorner(dim_g_, 1) = J_T_dense; 

            r_tilde(0) -= (J_T_dense.transpose() * segments[i].lam)(0, 0);

            // 缝合 Link Jacobian
            if (i < num_segments_ - 1) {
                int global_idx = 1 + i * n_x_;
                Eigen::MatrixXd J_end = link_results[i].jac_end;
                B_i.block((N_ - 1) * n_x_, global_idx, n_x_, n_x_) = J_end.transpose();
            }
            if (i > 0) {
                int global_idx = 1 + (i - 1) * n_x_;
                Eigen::MatrixXd J_start = link_results[i-1].jac_start;
                B_i.block(0, global_idx, n_x_, n_x_) = J_start.transpose();
            }

            B_cache[i] = B_i;
            r_cache[i] = r_i;

            // 舒尔补压缩
            Eigen::MatrixXd Kinv_Bi = ldlt_cache[i].solve(B_i);
            S.noalias() -= B_i.transpose() * Kinv_Bi;

            Eigen::VectorXd Kinv_ri = ldlt_cache[i].solve(r_i);
            r_tilde.noalias() -= B_i.transpose() * Kinv_ri;
        }
        
        // ----------------------------------------------------------------------
        // [阶段 3] 求解全局耦合系统
        // ----------------------------------------------------------------------
        S.diagonal().array() += 1e-8; // 全局正则化
        Eigen::VectorXd delta_xg = S.ldlt().solve(r_tilde);
        
        // ----------------------------------------------------------------------
        // [阶段 4] 回代求解局部变量步长
        // ----------------------------------------------------------------------
        StepData step_out;
        step_out.delta_T = delta_xg(0); 
        step_out.delta_Z.resize(num_segments_);
        step_out.delta_lam.resize(num_segments_);
        step_out.delta_link_lam.resize(num_segments_ - 1);
        
        // 分离 Link 乘子的更新步长
        for (int k = 0; k < num_segments_ - 1; ++k) {
            step_out.delta_link_lam[k] = delta_xg.segment(1 + k * n_x_, n_x_);
        }

        // 回代获取局部原对偶变量步长
        for (int i = 0; i < num_segments_; ++i) {
            Eigen::VectorXd rhs_local = r_cache[i] - B_cache[i] * delta_xg;
            Eigen::VectorXd delta_wi = ldlt_cache[i].solve(rhs_local);
            step_out.delta_Z[i] = delta_wi.head(dim_Z_);
            step_out.delta_lam[i] = delta_wi.tail(dim_g_);
        }

        return step_out;
    }
};