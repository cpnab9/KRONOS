#include "kronos_optimizer.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>

using namespace std;

namespace kronos {

NewtonOptimizer::NewtonOptimizer(NlpWrapper& nlp, SchurKktSolver& kkt_solver) 
    : nlp_(nlp), kkt_solver_(kkt_solver) {}

double NewtonOptimizer::compute_merit(const VectorXd& w, const VectorXd& lam, double merit_mu) {
    int nw = nlp_.get_nw();
    int ng = nlp_.get_ng();
    // 修改：替换为稀疏矩阵
    SparseMatrixXd H_dummy(nw, nw), A_dummy(ng, nw);
    VectorXd grad_dummy(nw), g_cand(ng);
    
    nlp_.evaluate(w, lam, H_dummy, A_dummy, grad_dummy, g_cand);
    return w(nw - 1) + merit_mu * g_cand.cwiseAbs().sum();
}

bool NewtonOptimizer::optimize(VectorXd& w_k, VectorXd& lam_k) {
    int nw = nlp_.get_nw();
    int ng = nlp_.get_ng();

    // 修改：替换为稀疏矩阵
    SparseMatrixXd H_val(nw, nw), A_val(ng, nw);
    VectorXd grad_L_val(nw), g_val(ng);
    VectorXd d_w(nw), d_lam(ng);

    cout << left << setw(5) << "Iter" << " | "
         << setw(12) << "Res (Norm)" << " | "
         << setw(10) << "Step Alpha" << " | "
         << setw(8) << "Time (T)" << "\n"
         << "----------------------------------------------\n";

    for (int iter = 0; iter < max_iter_; ++iter) {
        // 1. 评估 NLP 系统
        nlp_.evaluate(w_k, lam_k, H_val, A_val, grad_L_val, g_val);
        
        double res_err = max(grad_L_val.cwiseAbs().maxCoeff(), g_val.cwiseAbs().maxCoeff());
        if (res_err < tol_) {
            cout << "\n✅ Converged at iteration " << iter << "!\n";
            return true;
        }
        
        // 2. 调用底层的求解器
        kkt_solver_.solve(H_val, A_val, grad_L_val, g_val, d_w, d_lam);
        
        // 3. 线搜索 (Line Search)
        double alpha = 1.0;
        double merit_mu = (lam_k + d_lam).cwiseAbs().maxCoeff() + 1.0;
        double phi_0 = compute_merit(w_k, lam_k, merit_mu);
        
        while (alpha > 1e-4) {
            VectorXd w_new = w_k + alpha * d_w;
            VectorXd lam_new = lam_k + alpha * d_lam;
            if (compute_merit(w_new, lam_new, merit_mu) <= phi_0) {
                break;
            }
            alpha *= 0.5;
        }
        
        // 4. 更新变量
        w_k += alpha * d_w;
        lam_k += alpha * d_lam;
        
        cout << left << setw(5) << iter << " | "
             << scientific << setprecision(5) << setw(12) << res_err << " | "
             << fixed << setprecision(4) << setw(10) << alpha << " | "
             << fixed << setprecision(4) << setw(8) << w_k(nw - 1) << "\n";
        cout.unsetf(ios_base::floatfield); 
    }
    cout << "\n❌ Reached maximum iterations.\n";
    return false;
}

} // namespace kronos