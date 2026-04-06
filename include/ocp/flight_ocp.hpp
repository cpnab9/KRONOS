#pragma once
#include <fatrop/fatrop.hpp>

namespace aeroplan {

class FlightOCP : public fatrop::OcpAbstract {
private:
    const fatrop::Index K_intervals = 20;
    const fatrop::Index K_total = 21; 
    const fatrop::Index nx_ = 4;
    const fatrop::Index nu_ = 15;
    const fatrop::Index ng_defects_ = 12;

public:
    virtual fatrop::Index get_nx(const fatrop::Index k) const override;
    virtual fatrop::Index get_nu(const fatrop::Index k) const override;
    virtual fatrop::Index get_horizon_length() const override;
    virtual fatrop::Index get_ng(const fatrop::Index k) const override;
    virtual fatrop::Index get_ng_ineq(const fatrop::Index k) const override;

    // 注意：这里的 MAT 不加 fatrop:: 前缀，因为它是全局 BLASFEO 宏
    virtual fatrop::Index eval_BAbt(const fatrop::Scalar *states_kp1, const fatrop::Scalar *inputs_k, const fatrop::Scalar *states_k, MAT *res, const fatrop::Index k) override;
    virtual fatrop::Index eval_RSQrqt(const fatrop::Scalar *objective_scale, const fatrop::Scalar *inputs_k, const fatrop::Scalar *states_k, const fatrop::Scalar *lam_dyn_k, const fatrop::Scalar *lam_eq_k, const fatrop::Scalar *lam_eq_ineq_k, MAT *res, const fatrop::Index k) override;
    virtual fatrop::Index eval_Ggt(const fatrop::Scalar *inputs_k, const fatrop::Scalar *states_k, MAT *res, const fatrop::Index k) override;
    
    virtual fatrop::Index eval_b(const fatrop::Scalar *states_kp1, const fatrop::Scalar *inputs_k, const fatrop::Scalar *states_k, fatrop::Scalar *res, const fatrop::Index k) override;
    virtual fatrop::Index eval_g(const fatrop::Scalar *inputs_k, const fatrop::Scalar *states_k, fatrop::Scalar *res, const fatrop::Index k) override;
    virtual fatrop::Index eval_L(const fatrop::Scalar *objective_scale, const fatrop::Scalar *inputs_k, const fatrop::Scalar *states_k, fatrop::Scalar *res, const fatrop::Index k) override;
    virtual fatrop::Index eval_rq(const fatrop::Scalar *objective_scale, const fatrop::Scalar *inputs_k, const fatrop::Scalar *states_k, fatrop::Scalar *res, const fatrop::Index k) override;
    virtual fatrop::Index eval_gineq(const fatrop::Scalar *inputs_k, const fatrop::Scalar *states_k, fatrop::Scalar *res, const fatrop::Index k) override;
    
    // 同样，这里的 MAT 也不加 fatrop:: 前缀
    virtual fatrop::Index eval_Ggt_ineq(const fatrop::Scalar *inputs_k, const fatrop::Scalar *states_k, MAT *res, const fatrop::Index k) override;
    
    virtual fatrop::Index get_bounds(fatrop::Scalar *lower, fatrop::Scalar *upper, const fatrop::Index k) const override;
    virtual fatrop::Index get_initial_xk(fatrop::Scalar *xk, const fatrop::Index k) const override;
    virtual fatrop::Index get_initial_uk(fatrop::Scalar *uk, const fatrop::Index k) const override;
};

} // namespace aeroplan