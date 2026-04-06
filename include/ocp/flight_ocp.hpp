#pragma once
#include <fatrop/fatrop.hpp>
#include <vector>

namespace aeroplan {

// 通用 OCP 配置文件
struct OCPConfig {
    fatrop::Index K_intervals = 20;
    fatrop::Index nx = 4;
    fatrop::Index nu = 15;
    fatrop::Index ng_defects = 12;

    // 初始状态约束 (绑定哪些状态的索引及对应的值)
    std::vector<int> init_idx;
    std::vector<double> init_val;

    // 终端状态约束
    std::vector<int> term_idx;
    std::vector<double> term_val;

    // 状态不等式约束边界
    std::vector<int> ineq_state_idx;
    std::vector<double> ineq_lower;
    std::vector<double> ineq_upper;

    // 目标函数: 指定优化的状态索引与权重
    int obj_state_idx = 3;
    double obj_weight = 1.0;

    // 热启动基础猜想
    std::vector<double> guess_xk;
    std::vector<double> guess_uk;
};

class FlightOCP : public fatrop::OcpAbstract {
private:
    OCPConfig cfg_;
    fatrop::Index K_total_;

    // CasADi 工作空间，预分配在堆上，避免在栈上使用变长数组(VLA)导致的内存溢出
    std::vector<double> work_J_BAbt_;
    std::vector<double> work_H_RSQrqt_;
    std::vector<double> work_J_Ggt_;
    std::vector<double> work_f_dyn_;
    std::vector<double> work_g_eq_;

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
    
    // 支持闭环在线规划时，动态更新初始反馈状态
    void update_initial_state(const std::vector<double>& x0);
};

} // namespace aeroplan