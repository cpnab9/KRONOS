#include "kronos_optimizer.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <algorithm>

using namespace std;

namespace kronos {

NewtonOptimizer::NewtonOptimizer(NlpWrapper& nlp, SchurKktSolver& kkt_solver) 
    : nlp_(nlp), kkt_solver_(kkt_solver) {}

double NewtonOptimizer::compute_kkt_residual(const VectorXd& w, const VectorXd& lam, 
                                             const VectorXd& z, const VectorXd& s, double mu) {
    int nw = nlp_.get_nw(), ng = nlp_.get_ng(), nh = nlp_.get_nh();
    SparseMatrixXd H_d(nw, nw), Ag_d(ng, nw), Ah_d(nh, nw);
    VectorXd grad_L(nw), g(ng), h(nh);
    double f_dummy; 
    nlp_.evaluate(w, lam, z, H_d, Ag_d, Ah_d, grad_L, g, h, f_dummy);
    
    double err = grad_L.cwiseAbs().maxCoeff();
    if (ng > 0) err = max(err, g.cwiseAbs().maxCoeff());
    if (nh > 0) {
        err = max(err, (h - s).cwiseAbs().maxCoeff());
        err = max(err, (s.cwiseProduct(z) - mu * VectorXd::Ones(nh)).cwiseAbs().maxCoeff());
    }
    return err;
}

double NewtonOptimizer::compute_merit(const VectorXd& w, const VectorXd& s, double mu, double merit_nu) {
    int nw = nlp_.get_nw(), ng = nlp_.get_ng(), nh = nlp_.get_nh();
    SparseMatrixXd H_d(nw, nw), Ag_d(ng, nw), Ah_d(nh, nw);
    VectorXd grad_L(nw), g(ng), h(nh);
    VectorXd lam_dummy(ng), z_dummy(nh);
    double f_val = 0.0;
    
    nlp_.evaluate(w, lam_dummy, z_dummy, H_d, Ag_d, Ah_d, grad_L, g, h, f_val);

    double merit = f_val; 
    if (ng > 0) merit += merit_nu * g.cwiseAbs().sum(); 
    if (nh > 0) {
        for (int i = 0; i < nh; ++i) {
            // 障碍项防止变量越界
            if (s(i) > 0) merit -= mu * std::log(s(i)); 
            else merit += 1e10; 
        }
        merit += merit_nu * (h - s).cwiseAbs().sum(); 
    }
    return merit;
}

bool NewtonOptimizer::optimize(VectorXd& w_k, VectorXd& lam_k) {
    int nw = nlp_.get_nw(), ng = nlp_.get_ng(), nh = nlp_.get_nh();
    VectorXd s_k = VectorXd::Ones(nh), z_k = VectorXd::Ones(nh);

    SparseMatrixXd H_val(nw, nw), A_g_val(ng, nw), A_h_val(nh, nw);
    VectorXd grad_L_val(nw), g_val(ng), h_val(nh);
    VectorXd d_w(nw), d_lam(ng), d_s(nh), d_z(nh);
    double f_val;

    cout << left << setw(5) << "Iter" << " | " << setw(12) << "Res (Norm)" << " | "
         << setw(10) << "Barrier mu" << " | " << setw(10) << "Alpha_P" << "\n"
         << "---------------------------------------------------\n";

    for (int iter = 0; iter < max_iter_; ++iter) {
        nlp_.evaluate(w_k, lam_k, z_k, H_val, A_g_val, A_h_val, grad_L_val, g_val, h_val, f_val);
        
        double mu = 0.0;
        if (nh > 0) mu = std::max(1e-8, (s_k.dot(z_k) / (double)nh) * 0.1);
        
        double res_err = compute_kkt_residual(w_k, lam_k, z_k, s_k, 0.0);
        if (res_err < tol_) {
            cout << "\n✅ Converged at iteration " << iter << "!\n";
            return true;
        }
        
        // --- 矩阵缩聚 ---
        SparseMatrixXd H_mod = H_val;
        VectorXd grad_L_mod = grad_L_val;
        VectorXd S_inv, Sigma;
        if (nh > 0) {
            S_inv = s_k.cwiseInverse();
            Sigma = S_inv.cwiseProduct(z_k);
            H_mod += SparseMatrixXd(A_h_val.transpose() * Sigma.asDiagonal() * A_h_val);
            grad_L_mod += A_h_val.transpose() * (Sigma.cwiseProduct(h_val) - mu * S_inv);
        }

        kkt_solver_.solve(H_mod, A_g_val, grad_L_mod, g_val, d_w, d_lam);
        
        if (nh > 0) {
            d_s = A_h_val * d_w + h_val - s_k;
            d_z = mu * S_inv - z_k - Sigma.cwiseProduct(d_s);
        }

        // --- 1. 计算最大允许步长 (Fraction-to-the-boundary) ---
        double alpha_p_max = 1.0, alpha_d_max = 1.0;
        double tau = 0.995; 
        if (nh > 0) {
            for (int i = 0; i < nh; ++i) {
                if (d_s(i) < 0) alpha_p_max = std::min(alpha_p_max, -tau * s_k(i) / d_s(i));
                if (d_z(i) < 0) alpha_d_max = std::min(alpha_d_max, -tau * z_k(i) / d_z(i));
            }
        }
        
        // --- 2. 确定惩罚参数 nu ---
        double merit_nu = 0.0;
        if (ng > 0) merit_nu = std::max(merit_nu, lam_k.cwiseAbs().maxCoeff());
        if (nh > 0) merit_nu = std::max(merit_nu, z_k.cwiseAbs().maxCoeff());
        merit_nu += 10.0; // 留出足够余量压制残差

        // --- 3. 针对主变量的回溯线搜索 (Backtracking) ---
        double alpha_p = alpha_p_max;
        double phi_0 = compute_merit(w_k, s_k, mu, merit_nu);
        
        while (alpha_p > 1e-4) {
            VectorXd w_next = w_k + alpha_p * d_w;
            VectorXd s_next = s_k;
            if (nh > 0) s_next += alpha_p * d_s;
            
            double phi_next = compute_merit(w_next, s_next, mu, merit_nu);
            // Armijo 条件：允许极小的松弛，防止死锁
            if (phi_next <= phi_0 + 1e-4 * alpha_p) break;
            alpha_p *= 0.5;
        }
        
        // --- 4. 更新变量 (主副变量步长分离) ---
        w_k += alpha_p * d_w;
        if (nh > 0) s_k += alpha_p * d_s;
        
        // 副变量（乘子）通常直接迈出最大可能的对偶步长，有助于 KKT 系统快速平衡
        lam_k += alpha_d_max * d_lam; 
        if (nh > 0) z_k += alpha_d_max * d_z;
        
        cout << left << setw(5) << iter << " | "
             << scientific << setprecision(5) << setw(12) << res_err << " | "
             << fixed << setprecision(6) << setw(10) << mu << " | "
             << fixed << setprecision(4) << setw(10) << alpha_p << "\n";
    }
    return false;
}

} // namespace kronos