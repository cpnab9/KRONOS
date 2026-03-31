#pragma once
#include "kronos_types.hpp"
#include "kronos_nlp_wrapper.hpp"
#include "kronos_kkt_solver.hpp"

namespace kronos {

class NewtonOptimizer {
public:
    NewtonOptimizer(NlpWrapper& nlp, SchurKktSolver& kkt_solver);

    // 优化接口保持不变，对外部调用者友好
    bool optimize(VectorXd& w, VectorXd& lam);

private:
    // 替换为更通用的 KKT 残差评估函数（支持有/无不等式）
    double compute_kkt_residual(const VectorXd& w, const VectorXd& lam, 
                                const VectorXd& z, const VectorXd& s, double mu);

    double compute_merit(const VectorXd& w, const VectorXd& s, double mu, double merit_nu);

    NlpWrapper& nlp_;
    SchurKktSolver& kkt_solver_;
    
    int max_iter_ = 500;
    double tol_ = 1e-5;
};

} // namespace kronos    