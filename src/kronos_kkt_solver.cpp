#include "kronos_kkt_solver.hpp"

namespace kronos {

SchurKktSolver::SchurKktSolver(double H_rho, double S_rho) 
    : H_rho_(H_rho), S_rho_(S_rho) {}

bool SchurKktSolver::solve(const MatrixXd& H, const MatrixXd& A,
                           const VectorXd& grad_L, const VectorXd& g,
                           VectorXd& d_w, VectorXd& d_lam) {
    int nw = H.rows();
    int ng = A.rows();

    // 1. 正则化 H 并求逆
    MatrixXd H_reg = H + H_rho_ * MatrixXd::Identity(nw, nw);
    MatrixXd H_inv = H_reg.inverse(); // 未来机载化时，此处可替换为稀疏直接求解器
    
    // 2. 构造舒尔补系统 S = A * H_inv * A^T
    MatrixXd S = A * H_inv * A.transpose();
    MatrixXd S_reg = S + S_rho_ * MatrixXd::Identity(ng, ng);
    
    // 3. 计算乘子步长 d_lam
    VectorXd rhs_lam = g - A * H_inv * grad_L;
    d_lam = S_reg.partialPivLu().solve(rhs_lam);
    
    // 4. 恢复状态步长 d_w
    d_w = -H_inv * (grad_L + A.transpose() * d_lam);

    return true; // 未来可以加入矩阵奇异性检查，返回 false 表示求解失败
}

} // namespace kronos