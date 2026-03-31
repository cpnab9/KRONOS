#include "kronos_kkt_solver.hpp"
#include <iostream>

namespace kronos {

SchurKktSolver::SchurKktSolver(double H_rho, double S_rho) 
    : H_rho_(H_rho), S_rho_(S_rho) {}

bool SchurKktSolver::solve(const MatrixXd& H, const MatrixXd& A,
                           const VectorXd& grad_L, const VectorXd& g,
                           VectorXd& d_w, VectorXd& d_lam) {
    int nw = H.rows();
    int ng = A.rows();

    // 1. 正则化 H (Inertia Correction)
    MatrixXd H_reg = H;
    H_reg.diagonal().array() += H_rho_;
    
    // =========================================================
    // 优化 1：彻底抛弃 .inverse()，改用极速且稳定的 LDLT 分解
    // 非线性规划的海森矩阵可能是对称不定矩阵，LDLT 是最优解
    // =========================================================
    Eigen::LDLT<MatrixXd> ldlt_H(H_reg);
    if (ldlt_H.info() != Eigen::Success) {
        std::cerr << "[KRONOS Solver] Error: Hessian factorization failed! Try increasing H_rho." << std::endl;
        return false;
    }

    // =========================================================
    // 优化 2：计算 X = H_reg^{-1} * A^T 
    // 让底层的分解器去解方程，而不是乘以逆矩阵
    // =========================================================
    MatrixXd X = ldlt_H.solve(A.transpose());
    
    // 2. 构造舒尔补系统 S = A * X
    MatrixXd S(ng, ng);
    // 优化 3：使用 .noalias() 阻止 Eigen 在堆栈上生成临时拷贝矩阵
    S.noalias() = A * X;
    S.diagonal().array() += S_rho_;
    
    // 3. 计算乘子步长的右侧 rhs_lam
    // 同理，先解出 y = H_reg^{-1} * grad_L
    VectorXd y = ldlt_H.solve(grad_L);
    VectorXd rhs_lam = g - A * y;
    
    // =========================================================
    // 优化 4：S 是对称的，抛弃偏主元 LU 分解，改用对称矩阵专用的分解
    // =========================================================
    Eigen::LDLT<MatrixXd> ldlt_S(S);
    if (ldlt_S.info() != Eigen::Success) {
        std::cerr << "[KRONOS Solver] Error: Schur complement factorization failed!" << std::endl;
        return false;
    }
    d_lam = ldlt_S.solve(rhs_lam);
    
    // =========================================================
    // 优化 5：数学魔法代换恢复状态步长 d_w
    // 原公式：d_w = -H_inv * (grad_L + A^T * d_lam)
    // 展开后：d_w = -H_inv * grad_L - H_inv * A^T * d_lam
    // 代入前面的 y 和 X：d_w = -y - X * d_lam
    // 结果：彻底省去了最后一次解线性方程组的步骤，变成纯粹的矩阵乘法！
    // =========================================================
    d_w.noalias() = -y - X * d_lam;

    return true;
}

} // namespace kronos