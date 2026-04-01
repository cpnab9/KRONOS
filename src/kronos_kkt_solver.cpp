#include "kronos_kkt_solver.hpp"
#include <iostream>
#include <vector>

namespace kronos {

SchurKktSolver::SchurKktSolver(double H_rho, double S_rho) 
    : H_rho_(H_rho), S_rho_(S_rho), is_analyzed_(false) {}

bool SchurKktSolver::solve(const SparseMatrixXd& H, const SparseMatrixXd& A_g, const SparseMatrixXd& A_h,
                           const VectorXd& grad_L, const VectorXd& g, const VectorXd& h,
                           const VectorXd& s, const VectorXd& z, double mu,
                           VectorXd& d_w, VectorXd& d_lam, VectorXd& d_z, VectorXd& d_s) {
    int nw = H.rows();
    int ng = A_g.rows();
    int nh = A_h.rows();
    int kkt_size = nw + ng + nh;

    // 1. 组装未缩聚的扩展 KKT 矩阵 (Expanded KKT System)
    // 这种做法彻底避免了 A_h^T * Sigma * A_h 造成的稠密 Fill-in 问题
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(H.nonZeros() + 2 * A_g.nonZeros() + 2 * A_h.nonZeros() + nw + ng + nh);

    // 1.1 H 与 正则化
    for (int k = 0; k < H.outerSize(); ++k) {
        for (SparseMatrixXd::InnerIterator it(H, k); it; ++it) {
            triplets.emplace_back(it.row(), it.col(), it.value());
        }
    }
    for (int i = 0; i < nw; ++i) triplets.emplace_back(i, i, H_rho_);

    // 1.2 等式约束 A_g 和 A_g^T
    for (int k = 0; k < A_g.outerSize(); ++k) {
        for (SparseMatrixXd::InnerIterator it(A_g, k); it; ++it) {
            triplets.emplace_back(nw + it.row(), it.col(), it.value());
            triplets.emplace_back(it.col(), nw + it.row(), it.value());
        }
    }
    for (int i = 0; i < ng; ++i) triplets.emplace_back(nw + i, nw + i, -S_rho_);

    // 1.3 不等式约束 A_h 和 A_h^T
    if (nh > 0) {
        for (int k = 0; k < A_h.outerSize(); ++k) {
            for (SparseMatrixXd::InnerIterator it(A_h, k); it; ++it) {
                triplets.emplace_back(nw + ng + it.row(), it.col(), it.value());
                triplets.emplace_back(it.col(), nw + ng + it.row(), it.value());
            }
        }
        // 右下角 -Sigma^-1 块 (Sigma^-1 = S * Z^-1)
        for (int i = 0; i < nh; ++i) {
            double sigma_inv = s(i) / z(i);
            triplets.emplace_back(nw + ng + i, nw + ng + i, -sigma_inv);
        }
    }

    SparseMatrixXd K(kkt_size, kkt_size);
    K.setFromTriplets(triplets.begin(), triplets.end());

    // 2. 组装右侧向量 (RHS)
    VectorXd rhs = VectorXd::Zero(kkt_size);
    rhs.head(nw) = -grad_L;
    if (ng > 0) rhs.segment(nw, ng) = -g;
    if (nh > 0) {
        for (int i = 0; i < nh; ++i) {
            rhs(nw + ng + i) = -h(i) + mu / z(i);
        }
    }

    // 3. 求解线性系统 (缓存分析模式)
    if (!is_analyzed_) {
        solver_.analyzePattern(K);
        is_analyzed_ = true;
    }
    solver_.factorize(K);

    if (solver_.info() != Eigen::Success) {
        std::cerr << "[KRONOS Solver] Error: KKT matrix factorization failed!" << std::endl;
        return false;
    }

    VectorXd sol = solver_.solve(rhs);
    if (solver_.info() != Eigen::Success) {
        std::cerr << "[KRONOS Solver] Error: KKT system solve failed!" << std::endl;
        return false;
    }

    // 4. 提取步长
    d_w = sol.head(nw);
    if (ng > 0) d_lam = sol.segment(nw, ng);
    if (nh > 0) {
        VectorXd d_z_tilde = sol.tail(nh);
        d_z = -d_z_tilde; // 恢复真实的对偶步长
        d_s = A_h * d_w + h - s; // 恢复松弛变量步长
    }

    return true;
}

} // namespace kronos