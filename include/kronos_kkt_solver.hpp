#pragma once
#include "kronos_types.hpp"

namespace kronos {

class SchurKktSolver {
public:
    SchurKktSolver(double H_rho = 1e-5, double S_rho = 1e-8);

    // 求解 KKT 线性系统，输出 d_w 和 d_lam
    bool solve(const MatrixXd& H, const MatrixXd& A,
               const VectorXd& grad_L, const VectorXd& g,
               VectorXd& d_w, VectorXd& d_lam);

private:
    double H_rho_;
    double S_rho_;
};

} // namespace kronos