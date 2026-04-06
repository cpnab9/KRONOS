#pragma once
#include <fatrop/fatrop.hpp>
#include <vector>

namespace aeroplan {

struct OCPConfig {
    fatrop::Index K_intervals = 100;
    fatrop::Index nx = 5;
    fatrop::Index nu = 21;
    fatrop::Index ng_defects = 15;
    fatrop::Index ng_ineq = 6;

    std::vector<int> init_idx;
    std::vector<double> init_val;

    std::vector<int> term_idx;
    std::vector<double> term_val;

    std::vector<double> ineq_lower;
    std::vector<double> ineq_upper;

    int obj_state_idx = 4;
    double obj_weight = 1.0;

    std::vector<double> guess_xk;
    std::vector<double> guess_uk;
};

class FlightOCP : public fatrop::OcpAbstract {
private:
    OCPConfig cfg_;
    fatrop::Index K_total_;

    std::vector<double> work_J_BAbt_;
    std::vector<double> work_H_RSQrqt_;
    std::vector<double> work_J_Ggt_;
    std::vector<double> work_f_dyn_;
    std::vector<double> work_g_eq_;
    std::vector<double> work_J_Ggt_ineq_; 

public:
    explicit FlightOCP(const OCPConfig& config);
    
    virtual fatrop::Index get_nx(const fatrop::Index k) const override;
    virtual fatrop::Index get_nu(const fatrop::Index k) const override;
    virtual fatrop::Index get_horizon_length() const override;
    virtual fatrop::Index get_ng(const fatrop::Index k) const override;
    virtual fatrop::Index get_ng_ineq(const fatrop::Index k) const override;

    virtual fatrop::Index eval_BAbt(const fatrop::Scalar *states_kp1, const fatrop::Scalar *inputs_k, const fatrop::Scalar *states_k, MAT *res, const fatrop::Index k) override;
    virtual fatrop::Index eval_RSQrqt(const fatrop::Scalar *objective_scale, const fatrop::Scalar *inputs_k, const fatrop::Scalar *states_k, const fatrop::Scalar *lam_dyn_k, const fatrop::Scalar *lam_eq_k, const fatrop::Scalar *lam_eq_ineq_k, MAT *res, const fatrop::Index k) override;
    virtual fatrop::Index eval_Ggt(const fatrop::Scalar *inputs_k, const fatrop::Scalar *states_k, MAT *res, const fatrop::Index k) override;
    
    virtual fatrop::Index eval_b(const fatrop::Scalar *states_kp1, const fatrop::Scalar *inputs_k, const fatrop::Scalar *states_k, fatrop::Scalar *res, const fatrop::Index k) override;
    virtual fatrop::Index eval_g(const fatrop::Scalar *inputs_k, const fatrop::Scalar *states_k, fatrop::Scalar *res, const fatrop::Index k) override;
    virtual fatrop::Index eval_L(const fatrop::Scalar *objective_scale, const fatrop::Scalar *inputs_k, const fatrop::Scalar *states_k, fatrop::Scalar *res, const fatrop::Index k) override;
    virtual fatrop::Index eval_rq(const fatrop::Scalar *objective_scale, const fatrop::Scalar *inputs_k, const fatrop::Scalar *states_k, fatrop::Scalar *res, const fatrop::Index k) override;
    
    virtual fatrop::Index eval_gineq(const fatrop::Scalar *inputs_k, const fatrop::Scalar *states_k, fatrop::Scalar *res, const fatrop::Index k) override;
    virtual fatrop::Index eval_Ggt_ineq(const fatrop::Scalar *inputs_k, const fatrop::Scalar *states_k, MAT *res, const fatrop::Index k) override;
    
    virtual fatrop::Index get_bounds(fatrop::Scalar *lower, fatrop::Scalar *upper, const fatrop::Index k) const override;
    virtual fatrop::Index get_initial_xk(fatrop::Scalar *xk, const fatrop::Index k) const override;
    virtual fatrop::Index get_initial_uk(fatrop::Scalar *uk, const fatrop::Index k) const override;
    
    void update_initial_state(const std::vector<double>& x0);
};

} // namespace aeroplan