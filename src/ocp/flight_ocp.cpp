#include "ocp/flight_ocp.hpp"
#include "ocp/casadi_wrapper.hpp"

namespace aeroplan {

using namespace fatrop;

FlightOCP::FlightOCP(const OCPConfig& config) : cfg_(config) {
    K_total_ = cfg_.K_intervals + 1;
    
    work_J_BAbt_.resize((cfg_.nu + cfg_.nx) * cfg_.nx, 0.0);
    work_H_RSQrqt_.resize((cfg_.nu + cfg_.nx) * (cfg_.nu + cfg_.nx), 0.0);
    work_J_Ggt_.resize((cfg_.nu + cfg_.nx) * cfg_.ng_defects, 0.0);
    work_f_dyn_.resize(cfg_.nx, 0.0);
    work_g_eq_.resize(cfg_.ng_defects, 0.0);
    work_J_Ggt_ineq_.resize((cfg_.nu + cfg_.nx) * cfg_.ng_ineq, 0.0);
}

void FlightOCP::update_initial_state(const std::vector<double>& x0) {
    if (x0.size() == cfg_.init_idx.size()) cfg_.init_val = x0;
}

Index FlightOCP::get_nx(const Index k) const { return cfg_.nx; }
Index FlightOCP::get_nu(const Index k) const { return (k == K_total_ - 1) ? 0 : (cfg_.nu + cfg_.ns); }
Index FlightOCP::get_horizon_length() const { return K_total_; }

Index FlightOCP::get_ng(const Index k) const {
    if (k == 0) return cfg_.ng_defects + cfg_.init_idx.size();          
    else if (k == K_total_ - 1) return cfg_.term_idx.size();         
    else return cfg_.ng_defects;
}

Index FlightOCP::get_ng_ineq(const Index k) const { 
    return (k == K_total_ - 1) ? 0 : (cfg_.ng_ineq + cfg_.ns); 
}

Index FlightOCP::eval_BAbt(const Scalar *states_kp1, const Scalar *inputs_k, const Scalar *states_k, MAT *res, const Index k) {
    blasfeo_gese_wrap(res->m, res->n, 0.0, res, 0, 0);
    call_casadi(eval_J_BAbt, states_k, inputs_k, work_J_BAbt_.data());
    blasfeo_pack_dmat(cfg_.nu, cfg_.nx, work_J_BAbt_.data(), cfg_.nu + cfg_.nx, res, 0, 0);
    blasfeo_pack_dmat(cfg_.nx, cfg_.nx, work_J_BAbt_.data() + cfg_.nu, cfg_.nu + cfg_.nx, res, cfg_.nu + cfg_.ns, 0);
    return 0;
}

Index FlightOCP::eval_RSQrqt(const Scalar *objective_scale, const Scalar *inputs_k, const Scalar *states_k, const Scalar *lam_dyn_k, const Scalar *lam_eq_k, const Scalar *lam_eq_ineq_k, MAT *res, const Index k) {
    blasfeo_gese_wrap(res->m, res->n, 0.0, res, 0, 0);
    if (k < K_total_ - 1) {
        call_casadi_hess(eval_H_RSQrqt, states_k, inputs_k, lam_dyn_k, lam_eq_k, lam_eq_ineq_k, work_H_RSQrqt_.data());
        
        blasfeo_pack_dmat(cfg_.nu, cfg_.nu, work_H_RSQrqt_.data(), cfg_.nu + cfg_.nx, res, 0, 0);
        blasfeo_pack_dmat(cfg_.nx, cfg_.nu, work_H_RSQrqt_.data() + cfg_.nu, cfg_.nu + cfg_.nx, res, cfg_.nu + cfg_.ns, 0);
        blasfeo_pack_dmat(cfg_.nu, cfg_.nx, work_H_RSQrqt_.data() + cfg_.nu * (cfg_.nu + cfg_.nx), cfg_.nu + cfg_.nx, res, 0, cfg_.nu + cfg_.ns);
        blasfeo_pack_dmat(cfg_.nx, cfg_.nx, work_H_RSQrqt_.data() + cfg_.nu * (cfg_.nu + cfg_.nx) + cfg_.nu, cfg_.nu + cfg_.nx, res, cfg_.nu + cfg_.ns, cfg_.nu + cfg_.ns);
        
        if (cfg_.ns > 0) {
            for (int i = 0; i < cfg_.ns; i++) {
                double Zl_val = cfg_.Zl.empty() ? 0.0 : cfg_.Zl[i];
                blasfeo_matel_wrap(res, cfg_.nu + i, cfg_.nu + i) = objective_scale[0] * Zl_val;
            }
        }
    }
    return 0;
}

Index FlightOCP::eval_Ggt(const Scalar *inputs_k, const Scalar *states_k, MAT *res, const Index k) {
    blasfeo_gese_wrap(res->m, res->n, 0.0, res, 0, 0);
    if (k < K_total_ - 1) {
        call_casadi(eval_J_Ggt, states_k, inputs_k, work_J_Ggt_.data());
        
        blasfeo_pack_dmat(cfg_.nu, cfg_.ng_defects, work_J_Ggt_.data(), cfg_.nu + cfg_.nx, res, 0, 0);
        blasfeo_pack_dmat(cfg_.nx, cfg_.ng_defects, work_J_Ggt_.data() + cfg_.nu, cfg_.nu + cfg_.nx, res, cfg_.nu + cfg_.ns, 0);
        
        if (k == 0) {
            for (size_t i = 0; i < cfg_.init_idx.size(); ++i) {
                blasfeo_matel_wrap(res, cfg_.nu + cfg_.ns + cfg_.init_idx[i], cfg_.ng_defects + i) = 1.0;
            }
        }
    } else {
        for (size_t i = 0; i < cfg_.term_idx.size(); ++i) {
            blasfeo_matel_wrap(res, cfg_.term_idx[i], i) = 1.0;
        }
    }
    return 0;
}

Index FlightOCP::eval_b(const Scalar *states_kp1, const Scalar *inputs_k, const Scalar *states_k, Scalar *res, const Index k) {
    call_casadi(eval_f_dyn, states_k, inputs_k, work_f_dyn_.data());
    for(int i=0; i<cfg_.nx; i++) res[i] = -states_kp1[i] + work_f_dyn_[i];
    return 0;
}

Index FlightOCP::eval_g(const Scalar *inputs_k, const Scalar *states_k, Scalar *res, const Index k) {
    if (k < K_total_ - 1) {
        call_casadi(eval_g_eq, states_k, inputs_k, res);
        if (k == 0) {
            for (size_t i = 0; i < cfg_.init_idx.size(); ++i) {
                res[cfg_.ng_defects + i] = states_k[cfg_.init_idx[i]] - cfg_.init_val[i];
            }
        }
    } else {
        for (size_t i = 0; i < cfg_.term_idx.size(); ++i) {
            res[i] = states_k[cfg_.term_idx[i]] - cfg_.term_val[i];
        }
    }
    return 0;
}

Index FlightOCP::eval_L(const Scalar *objective_scale, const Scalar *inputs_k, const Scalar *states_k, Scalar *res, const Index k) {
    *res = 0.0;
    if (k == K_total_ - 1) {
        *res = objective_scale[0] * cfg_.obj_weight * states_k[cfg_.obj_state_idx]; 
    } else if (cfg_.ns > 0) {
        const Scalar* s = inputs_k + cfg_.nu; 
        double slack_cost = 0.0;
        for (int i = 0; i < cfg_.ns; i++) {
            double zl_val = cfg_.zl.empty() ? 0.0 : cfg_.zl[i];
            double Zl_val = cfg_.Zl.empty() ? 0.0 : cfg_.Zl[i];
            slack_cost += 0.5 * Zl_val * s[i] * s[i] + zl_val * s[i];
        }
        *res = objective_scale[0] * slack_cost;
    }
    return 0;
}

Index FlightOCP::eval_rq(const Scalar *objective_scale, const Scalar *inputs_k, const Scalar *states_k, Scalar *res, const Index k) {
    int dim = (k == K_total_ - 1) ? cfg_.nx : (cfg_.nu + cfg_.ns + cfg_.nx);
    for(int i=0; i<dim; i++) res[i] = 0.0;
    if (k == K_total_ - 1) {
        res[cfg_.obj_state_idx] = objective_scale[0] * cfg_.obj_weight;
    } else if (cfg_.ns > 0) {
        const Scalar* s = inputs_k + cfg_.nu;
        for(int i=0; i<cfg_.ns; i++) {
            double zl_val = cfg_.zl.empty() ? 0.0 : cfg_.zl[i];
            double Zl_val = cfg_.Zl.empty() ? 0.0 : cfg_.Zl[i];
            res[cfg_.nu + i] = objective_scale[0] * (Zl_val * s[i] + zl_val); 
        }
    }
    return 0;
}

Index FlightOCP::eval_gineq(const Scalar *inputs_k, const Scalar *states_k, Scalar *res, const Index k) {
    if (k == K_total_ - 1) return 0;
    call_casadi(eval_g_ineq, states_k, inputs_k, res);
    if (cfg_.ns > 0) {
        const Scalar* s = inputs_k + cfg_.nu;
        for (int i = 0; i < cfg_.ns; i++) {
            res[cfg_.idx_s[i]] -= s[i];     
            res[cfg_.ng_ineq + i] = s[i];   
        }
    }
    return 0;
}

Index FlightOCP::eval_Ggt_ineq(const Scalar *inputs_k, const Scalar *states_k, MAT *res, const Index k) {
    if (k == K_total_ - 1) return 0;
    blasfeo_gese_wrap(res->m, res->n, 0.0, res, 0, 0);
    call_casadi(eval_J_Ggt_ineq, states_k, inputs_k, work_J_Ggt_ineq_.data());
    
    blasfeo_pack_dmat(cfg_.nu, cfg_.ng_ineq, work_J_Ggt_ineq_.data(), cfg_.nu + cfg_.nx, res, 0, 0);
    blasfeo_pack_dmat(cfg_.nx, cfg_.ng_ineq, work_J_Ggt_ineq_.data() + cfg_.nu, cfg_.nu + cfg_.nx, res, cfg_.nu + cfg_.ns, 0);

    if (cfg_.ns > 0) {
        for (int i = 0; i < cfg_.ns; i++) {
            blasfeo_matel_wrap(res, cfg_.nu + i, cfg_.idx_s[i]) = -1.0; 
            blasfeo_matel_wrap(res, cfg_.nu + i, cfg_.ng_ineq + i) = 1.0; 
        }
    }
    return 0;
}

Index FlightOCP::get_bounds(Scalar *lower, Scalar *upper, const Index k) const {
    if (k == K_total_ - 1) return 0;
    for (size_t i = 0; i < cfg_.ng_ineq; ++i) {
        lower[i] = cfg_.ineq_lower[i];
        upper[i] = cfg_.ineq_upper[i];
    }
    if (cfg_.ns > 0) {
        for (size_t i = 0; i < cfg_.ns; ++i) {
            lower[cfg_.ng_ineq + i] = 0.0;
            upper[cfg_.ng_ineq + i] = 1e20;
        }
    }
    return 0;
}

// === 核心修改：利用两端点进行线性插值产生高质量初值 ===
Index FlightOCP::get_initial_xk(Scalar *xk, const Index k) const {
    double interp_frac = (K_total_ > 1) ? (double)k / (K_total_ - 1) : 0.0;
    for(int i=0; i<cfg_.nx; i++) {
        double val0 = cfg_.guess_x0.empty() ? 0.0 : cfg_.guess_x0[i];
        double valf = cfg_.guess_xf.empty() ? val0 : cfg_.guess_xf[i];
        xk[i] = val0 + interp_frac * (valf - val0);
    }
    return 0;
}

Index FlightOCP::get_initial_uk(Scalar *uk, const Index k) const {
    if (k < K_total_ - 1) {
        double interp_frac = (K_total_ > 2) ? (double)k / (K_total_ - 2) : 0.0;
        for(int i=0; i<cfg_.nu; i++) {
            double val0 = cfg_.guess_u0.empty() ? 0.0 : cfg_.guess_u0[i];
            double valf = cfg_.guess_uf.empty() ? val0 : cfg_.guess_uf[i];
            uk[i] = val0 + interp_frac * (valf - val0);
        }
        if (cfg_.ns > 0) {
            for(int i=0; i<cfg_.ns; i++) {
                uk[cfg_.nu + i] = cfg_.guess_sk.empty() ? 1e-3 : cfg_.guess_sk[i];
            }
        }
    }
    return 0;
}

} // namespace aeroplan