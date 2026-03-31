#include "kronos_kkt_solver.hpp"
#include <iostream>
#include <vector>
#include <Eigen/SparseLU> // 引入稀疏 LU 分解

namespace kronos {

SchurKktSolver::SchurKktSolver(double H_rho, double S_rho) 
    : H_rho_(H_rho), S_rho_(S_rho) {}

bool SchurKktSolver::solve(const SparseMatrixXd& H, const SparseMatrixXd& A,
                           const VectorXd& grad_L, const VectorXd& g,
                           VectorXd& d_w, VectorXd& d_lam) {
    int nw = H.rows();
    int ng = A.rows();
    int kkt_size = nw + ng;

    // 1. 组装大型稀疏 KKT 矩阵 (Augmented System)
    // K = [ H + H_rho*I    A^T    ]
    //     [      A      -S_rho*I  ]
    SparseMatrixXd K(kkt_size, kkt_size);
    std::vector<Eigen::Triplet<double>> triplets;
    
    // 预估非零元素数量以加速内存分配
    triplets.reserve(H.nonZeros() + 2 * A.nonZeros() + nw + ng);

    // 1.1 插入 H 的非零元素
    for (int k = 0; k < H.outerSize(); ++k) {
        for (SparseMatrixXd::InnerIterator it(H, k); it; ++it) {
            triplets.emplace_back(it.row(), it.col(), it.value());
        }
    }
    // 1.2 插入 H 的对角正则化 (Inertia Correction)
    for (int i = 0; i < nw; ++i) {
        triplets.emplace_back(i, i, H_rho_);
    }

    // 1.3 插入 A 和 A^T 的非零元素
    for (int k = 0; k < A.outerSize(); ++k) {
        for (SparseMatrixXd::InnerIterator it(A, k); it; ++it) {
            triplets.emplace_back(nw + it.row(), it.col(), it.value()); // 左下角 A
            triplets.emplace_back(it.col(), nw + it.row(), it.value()); // 右上角 A^T
        }
    }

    // 1.4 插入右下角的对角正则化
    for (int i = 0; i < ng; ++i) {
        triplets.emplace_back(nw + i, nw + i, -S_rho_);
    }

    K.setFromTriplets(triplets.begin(), triplets.end());

    // 2. 组装右侧向量 (RHS)
    VectorXd rhs(kkt_size);
    rhs.head(nw) = -grad_L;
    rhs.tail(ng) = -g;

    // 3. 求解线性系统
    // 这里使用 Eigen::SparseLU 求解对称不定系统
    Eigen::SparseLU<SparseMatrixXd> solver;
    solver.analyzePattern(K);
    solver.factorize(K);

    if (solver.info() != Eigen::Success) {
        std::cerr << "[KRONOS Solver] Error: KKT matrix factorization failed!" << std::endl;
        return false;
    }

    VectorXd sol = solver.solve(rhs);
    if (solver.info() != Eigen::Success) {
        std::cerr << "[KRONOS Solver] Error: KKT system solve failed!" << std::endl;
        return false;
    }

    // 4. 提取步长
    d_w = sol.head(nw);
    d_lam = sol.tail(ng);

    return true;
}

} // namespace kronos