#pragma once
#include "kronos_types.hpp"
#include "kronos_nlp_wrapper.hpp"
#include "kronos_kkt_solver.hpp"

namespace kronos {

class NewtonOptimizer {
public:
    // 依赖注入：需要一个 NLP 模型和一个 KKT 求解器
    NewtonOptimizer(NlpWrapper& nlp, SchurKktSolver& kkt_solver);

    bool optimize(VectorXd& w, VectorXd& lam);

private:
    double compute_merit(const VectorXd& w, const VectorXd& lam, double merit_mu);

    NlpWrapper& nlp_;
    SchurKktSolver& kkt_solver_;
    
    int max_iter_ = 500;
    double tol_ = 1e-5;
};

} // namespace kronos