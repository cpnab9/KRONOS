#pragma once
#include "kronos_types.hpp"

namespace kronos {

// 虽然名字还叫 SchurKktSolver，但我们将把它重构为全稀疏 KKT 系统求解
class SchurKktSolver {
public:
    SchurKktSolver(double H_rho = 1e-5, double S_rho = 1e-8);

    // 修改：将 H 和 A 修改为 SparseMatrixXd
    bool solve(const SparseMatrixXd& H, const SparseMatrixXd& A,
               const VectorXd& grad_L, const VectorXd& g,
               VectorXd& d_w, VectorXd& d_lam);

private:
    double H_rho_;
    double S_rho_;
};

} // namespace kronos