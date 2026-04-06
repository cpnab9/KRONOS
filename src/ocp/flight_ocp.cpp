#include "ocp/flight_ocp.hpp"
#include "ocp/casadi_wrapper.hpp"

namespace aeroplan {

using namespace fatrop;

Index FlightOCP::get_nx(const Index k) const { return nx_; }
Index FlightOCP::get_nu(const Index k) const { return (k == K_total - 1) ? 0 : nu_; }
Index FlightOCP::get_horizon_length() const { return K_total; }

Index FlightOCP::get_ng(const Index k) const {
    if (k == 0) return ng_defects_ + 3;          
    else if (k == K_total - 1) return 2;         
    else return ng_defects_;
}

Index FlightOCP::get_ng_ineq(const Index k) const { return 1; }

Index FlightOCP::eval_BAbt(const Scalar *states_kp1, const Scalar *inputs_k, const Scalar *states_k, MAT *res, const Index k) {
    blasfeo_gese_wrap(res->m, res->n, 0.0, res, 0, 0);
    double J_val[(15 + 4) * 4];
    call_casadi(eval_J_BAbt, states_k, inputs_k, J_val);
    blasfeo_pack_dmat(nu_ + nx_, nx_, J_val, nu_ + nx_, res, 0, 0);
    return 0;
}

Index FlightOCP::eval_RSQrqt(const Scalar *objective_scale, const Scalar *inputs_k, const Scalar *states_k, const Scalar *lam_dyn_k, const Scalar *lam_eq_k, const Scalar *lam_eq_ineq_k, MAT *res, const Index k) {
    blasfeo_gese_wrap(res->m, res->n, 0.0, res, 0, 0);
    if (k < K_total - 1) {
        double H_val[(15 + 4) * (15 + 4)];
        call_casadi_hess(eval_H_RSQrqt, states_k, inputs_k, lam_dyn_k, lam_eq_k, H_val);
        blasfeo_pack_dmat(nu_ + nx_, nu_ + nx_, H_val, nu_ + nx_, res, 0, 0);
    }
    return 0;
}

Index FlightOCP::eval_Ggt(const Scalar *inputs_k, const Scalar *states_k, MAT *res, const Index k) {
    blasfeo_gese_wrap(res->m, res->n, 0.0, res, 0, 0);
    if (k < K_total - 1) {
        double J_val[(15 + 4) * 12];
        call_casadi(eval_J_Ggt, states_k, inputs_k, J_val);
        blasfeo_pack_dmat(nu_ + nx_, ng_defects_, J_val, nu_ + nx_, res, 0, 0);
        if (k == 0) {
            blasfeo_matel_wrap(res, nu_ + 0, 12) = 1.0;
            blasfeo_matel_wrap(res, nu_ + 1, 13) = 1.0;
            blasfeo_matel_wrap(res, nu_ + 2, 14) = 1.0;
        }
    } else {
        blasfeo_matel_wrap(res, 0, 0) = 1.0; 
        blasfeo_matel_wrap(res, 1, 1) = 1.0; 
    }
    return 0;
}

Index FlightOCP::eval_b(const Scalar *states_kp1, const Scalar *inputs_k, const Scalar *states_k, Scalar *res, const Index k) {
    double f_val[4];
    call_casadi(eval_f_dyn, states_k, inputs_k, f_val);
    for(int i=0; i<nx_; i++) res[i] = -states_kp1[i] + f_val[i];
    return 0;
}

Index FlightOCP::eval_g(const Scalar *inputs_k, const Scalar *states_k, Scalar *res, const Index k) {
    if (k < K_total - 1) {
        call_casadi(eval_g_eq, states_k, inputs_k, res);
        if (k == 0) {
            res[12] = states_k[0] - 0.0;
            res[13] = states_k[1] - 0.0;
            res[14] = states_k[2] - 0.0;
        }
    } else {
        res[0] = states_k[0] - 1.0;
        res[1] = states_k[1] - 1.0;
    }
    return 0;
}

Index FlightOCP::eval_L(const Scalar *objective_scale, const Scalar *inputs_k, const Scalar *states_k, Scalar *res, const Index k) {
    if (k == K_total - 1) *res = objective_scale[0] * states_k[3]; 
    else *res = 0.0;
    return 0;
}

Index FlightOCP::eval_rq(const Scalar *objective_scale, const Scalar *inputs_k, const Scalar *states_k, Scalar *res, const Index k) {
    int dim = (k == K_total - 1) ? nx_ : (nu_ + nx_);
    for(int i=0; i<dim; i++) res[i] = 0.0;
    if (k == K_total - 1) res[3] = objective_scale[0] * 1.0;
    return 0;
}

Index FlightOCP::eval_gineq(const Scalar *inputs_k, const Scalar *states_k, Scalar *res, const Index k) {
    res[0] = states_k[3]; 
    return 0;
}

Index FlightOCP::eval_Ggt_ineq(const Scalar *inputs_k, const Scalar *states_k, MAT *res, const Index k) {
    blasfeo_gese_wrap(res->m, res->n, 0.0, res, 0, 0);
    int offset = (k == K_total - 1) ? 0 : nu_; 
    blasfeo_matel_wrap(res, offset + 3, 0) = 1.0;
    return 0;
}

Index FlightOCP::get_bounds(Scalar *lower, Scalar *upper, const Index k) const {
    lower[0] = 0.1;   upper[0] = 10.0;  
    return 0;
}

Index FlightOCP::get_initial_xk(Scalar *xk, const Index k) const {
    xk[0] = k * (1.0 / K_intervals);  
    xk[1] = k * (1.0 / K_intervals);  
    xk[2] = 2.0;                      
    xk[3] = 1.0;                      
    return 0;
}

Index FlightOCP::get_initial_uk(Scalar *uk, const Index k) const {
    if (k < K_total - 1) for(int i=0; i<nu_; i++) uk[i] = 0.5; 
    return 0;
}

} // namespace aeroplan