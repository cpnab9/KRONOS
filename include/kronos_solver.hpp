#pragma once
#include "kronos_types.hpp"
#include "kronos_nlp_wrapper.hpp"

namespace kronos {

class NewtonSchurSolver {
public:
    NewtonSchurSolver(NlpWrapper& nlp);

    // 执行优化，传入的 w 和 lam 将在函数内部被更新为最优解
    bool solve(VectorXd& w, VectorXd& lam);

private:
    // 计算价值函数 (Merit Function)
    double compute_merit(const VectorXd& w, const VectorXd& lam, double merit_mu);

    NlpWrapper& nlp_;
    
    // 算法参数
    int max_iter_ = 500;
    double tol_ = 1e-5;
    double H_rho_ = 1e-5;
    double S_rho_ = 1e-8;
};

} // namespace kronos