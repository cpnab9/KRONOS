#pragma once
#include "kronos_types.hpp"
#include <vector>

namespace kronos {

class NlpWrapper {
public:
    NlpWrapper();
    
    // 评估 KKT 系统
    void evaluate(const VectorXd& w, const VectorXd& lam,
                  MatrixXd& H, MatrixXd& A, VectorXd& grad_L, VectorXd& g);

    int get_nw() const { return n_w_; }
    int get_ng() const { return n_g_; }

private:
    int n_w_;
    int n_g_;
    // 预先分配 CasADi 计算所需的内存工作区，避免循环中重复 new/delete
    std::vector<long long> iw_;
    std::vector<double> work_;
};

} // namespace kronos