#include "ocp/flight_ocp.hpp"

extern "C" {
    int eval_f_dyn(const double** arg, double** res, int* iw, double* w, int mem);
    int eval_g_eq(const double** arg, double** res, int* iw, double* w, int mem);
    int eval_g_ineq(const double** arg, double** res, int* iw, double* w, int mem);
    int eval_J_BAbt(const double** arg, double** res, int* iw, double* w, int mem);
    int eval_J_Ggt(const double** arg, double** res, int* iw, double* w, int mem);
    int eval_J_Ggt_ineq(const double** arg, double** res, int* iw, double* w, int mem);
    int eval_H_RSQrqt(const double** arg, double** res, int* iw, double* w, int mem);
}

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

    opt_x_.assign(K_total_, std::vector<double>(cfg_.nx, 0.0));
    opt_u_.assign(cfg_.K_intervals, std::vector<double>(cfg_.nu, 0.0));
}

void FlightOCP::update_initial_state(const std::vector<double>& x0) {
    if (x0.size() == cfg_.init_idx.size()) cfg_.init_val = x0;
}

void FlightOCP::update_config(const OCPConfig& new_cfg) {
    cfg_ = new_cfg;
}

void FlightOCP::get_last_trajectory(std::vector<std::vector<double>>& x_out, std::vector<std::vector<double>>& u_out) const {
    x_out = opt_x_;
    u_out = opt_u_;
}

inline void fill_casadi_params(const OCPConfig& cfg, int k, double* param) {
    param[0] = cfg.mesh_fractions.empty() ? (1.0 / cfg.K_intervals) : cfg.mesh_fractions[k];
    param[1] = cfg.enable_nfz2 ? 1.0 : 0.0;
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
    double param[2]; fill_casadi_params(cfg_, k, param);
    const double* arg[3] = {states_k, inputs_k, param};
    double* res_ptr[1] = {work_J_BAbt_.data()};
    eval_J_BAbt(arg, res_ptr, nullptr, nullptr, 0);
    
    blasfeo_pack_dmat(cfg_.nu, cfg_.nx, work_J_BAbt_.data(), cfg_.nu + cfg_.nx, res, 0, 0);
    blasfeo_pack_dmat(cfg_.nx, cfg_.nx, work_J_BAbt_.data() + cfg_.nu, cfg_.nu + cfg_.nx, res, cfg_.nu + cfg_.ns, 0);
    return 0;
}

Index FlightOCP::eval_RSQrqt(const Scalar *objective_scale, const Scalar *inputs_k, const Scalar *states_k, const Scalar *lam_dyn_k, const Scalar *lam_eq_k, const Scalar *lam_eq_ineq_k, MAT *res, const Index k) {
    blasfeo_gese_wrap(res->m, res->n, 0.0, res, 0, 0);
    if (k < K_total_ - 1) {
        double param[2]; fill_casadi_params(cfg_, k, param);
        const double* arg[6] = {states_k, inputs_k, param, lam_dyn_k, lam_eq_k, lam_eq_ineq_k};
        double* res_ptr[1] = {work_H_RSQrqt_.data()};
        eval_H_RSQrqt(arg, res_ptr, nullptr, nullptr, 0);
        
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
        double param[2]; fill_casadi_params(cfg_, k, param);
        const double* arg[3] = {states_k, inputs_k, param};
        double* res_ptr[1] = {work_J_Ggt_.data()};
        eval_J_Ggt(arg, res_ptr, nullptr, nullptr, 0);
        
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
    if (k == 0) {
        for (int i = 0; i < cfg_.nx; ++i) opt_x_[0][i] = states_k[i];
    }
    for (int i = 0; i < cfg_.nx; ++i) opt_x_[k + 1][i] = states_kp1[i];
    for (int i = 0; i < cfg_.nu; ++i) opt_u_[k][i] = inputs_k[i];

    double param[2]; fill_casadi_params(cfg_, k, param);
    const double* arg[3] = {states_k, inputs_k, param};
    double* res_ptr[1] = {work_f_dyn_.data()};
    eval_f_dyn(arg, res_ptr, nullptr, nullptr, 0);
    
    for(int i=0; i<cfg_.nx; i++) res[i] = -states_kp1[i] + work_f_dyn_[i];
    return 0;
}

Index FlightOCP::eval_g(const Scalar *inputs_k, const Scalar *states_k, Scalar *res, const Index k) {
    if (k < K_total_ - 1) {
        double param[2]; fill_casadi_params(cfg_, k, param);
        const double* arg[3] = {states_k, inputs_k, param};
        double* res_ptr[1] = {res};
        eval_g_eq(arg, res_ptr, nullptr, nullptr, 0);
        
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
    
    double param[2]; fill_casadi_params(cfg_, k, param);
    const double* arg[3] = {states_k, inputs_k, param};
    double* res_ptr[1] = {res};
    eval_g_ineq(arg, res_ptr, nullptr, nullptr, 0);
    
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
    
    double param[2]; fill_casadi_params(cfg_, k, param);
    const double* arg[3] = {states_k, inputs_k, param};
    double* res_ptr[1] = {work_J_Ggt_ineq_.data()};
    eval_J_Ggt_ineq(arg, res_ptr, nullptr, nullptr, 0);
    
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

Index FlightOCP::get_initial_xk(Scalar *xk, const Index k) const {
    if (cfg_.use_warm_start && k < cfg_.warm_x.size()) {
        for(int i=0; i<cfg_.nx; i++) xk[i] = cfg_.warm_x[k][i];
        return 0;
    }
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
        if (cfg_.use_warm_start && k < cfg_.warm_u.size()) {
            for(int i=0; i<cfg_.nu; i++) uk[i] = cfg_.warm_u[k][i];
            if (cfg_.ns > 0) {
                // ========================================================
                // 【修复 1】：Slack 热启动初值修正
                // 突发障碍物导致上一代的轨迹可能在新环境中是严重违规的。
                // 必须赋予 Slack 足够大的初值来“吸收”约束越界，保证 g(x)-s < 0。
                // 如果使用 1e-3，IPM 求解器会认为初始点不可行，导致极端的步长截断！
                // ========================================================
                for(int i=0; i<cfg_.ns; i++) {
                    uk[cfg_.nu + i] = 0.1; // 放大 Slack 初值
                }
            }
            return 0;
        }
        
        // 冷启动逻辑
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