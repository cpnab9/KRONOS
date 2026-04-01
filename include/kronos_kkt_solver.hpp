#pragma once
#include "kronos_types.hpp"
#include <Eigen/SparseLU>

namespace kronos {

class SchurKktSolver {
public:
    SchurKktSolver(double H_rho = 1e-6, double S_rho = 1e-8);

    // 全新接口：直接求解扩展 KKT 系统，避免矩阵缩聚导致的 Fill-in 爆炸
    bool solve(const SparseMatrixXd& H, const SparseMatrixXd& A_g, const SparseMatrixXd& A_h,
               const VectorXd& grad_L, const VectorXd& g, const VectorXd& h,
               const VectorXd& s, const VectorXd& z, double mu,
               VectorXd& d_w, VectorXd& d_lam, VectorXd& d_z, VectorXd& d_s);

private:
    double H_rho_;
    double S_rho_;
    
    // 【性能优化核心】：缓存 LU 分解器，避免迭代中重复分析稀疏模式
    Eigen::SparseLU<SparseMatrixXd> solver_;
    bool is_analyzed_ = false;
};

} // namespace kronos