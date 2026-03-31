#include "kronos_solver.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>

using namespace std;

namespace kronos {

NewtonSchurSolver::NewtonSchurSolver(NlpWrapper& nlp) : nlp_(nlp) {}

double NewtonSchurSolver::compute_merit(const VectorXd& w, const VectorXd& lam, double merit_mu) {
    int nw = nlp_.get_nw();
    int ng = nlp_.get_ng();
    
    MatrixXd H_dummy(nw, nw), A_dummy(ng, nw);
    VectorXd grad_dummy(nw), g_cand(ng);
    
    nlp_.evaluate(w, lam, H_dummy, A_dummy, grad_dummy, g_cand);
    
    // w 的最后一个元素是时间 T
    return w(nw - 1) + merit_mu * g_cand.cwiseAbs().sum();
}

bool NewtonSchurSolver::solve(VectorXd& w_k, VectorXd& lam_k) {
    int nw = nlp_.get_nw();
    int ng = nlp_.get_ng();

    MatrixXd H_val(nw, nw), A_val(ng, nw);
    VectorXd grad_L_val(nw), g_val(ng);

    cout << left << setw(5) << "Iter" << " | "
         << setw(12) << "Res (Norm)" << " | "
         << setw(10) << "Step Alpha" << " | "
         << setw(8) << "Time (T)" << "\n"
         << "----------------------------------------------\n";

    for (int iter = 0; iter < max_iter_; ++iter) {
        nlp_.evaluate(w_k, lam_k, H_val, A_val, grad_L_val, g_val);
        
        double res_err = max(grad_L_val.cwiseAbs().maxCoeff(), g_val.cwiseAbs().maxCoeff());
        if (res_err < tol_) {
            cout << "\n✅ Converged at iteration " << iter << "!\n";
            return true;
        }
        
        // 1. 正则化与求逆
        MatrixXd H_reg = H_val + H_rho_ * MatrixXd::Identity(nw, nw);
        MatrixXd H_inv = H_reg.inverse();
        
        // 2. 构造与求解舒尔补
        MatrixXd S = A_val * H_inv * A_val.transpose();
        MatrixXd S_reg = S + S_rho_ * MatrixXd::Identity(ng, ng);
        
        VectorXd rhs_lam = g_val - A_val * H_inv * grad_L_val;
        VectorXd d_lam = S_reg.partialPivLu().solve(rhs_lam);
        
        // 3. 恢复状态增量
        VectorXd d_w = -H_inv * (grad_L_val + A_val.transpose() * d_lam);
        
        // 4. 线搜索
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