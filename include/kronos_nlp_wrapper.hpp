#ifndef KRONOS_NLP_WRAPPER_HPP
#define KRONOS_NLP_WRAPPER_HPP

#include "kronos_types.hpp"
#include "../generated/kronos_config.h"

namespace kronos {

class NlpWrapper {
public:
    NlpWrapper();

    // 评估 KKT 系统，新增了输入 z，以及输出 A_h 和 h
    void evaluate(const VectorXd& w, const VectorXd& lam, const VectorXd& z, 
                  SparseMatrixXd& H, SparseMatrixXd& A_g, SparseMatrixXd& A_h, 
                  VectorXd& grad_L, VectorXd& g, VectorXd& h, double& f_val); 

    // 获取维度接口
    int get_nw() const { return KRONOS_N_W; }
    int get_ng() const { return KRONOS_N_G; }
    int get_nh() const { return KRONOS_N_H; } // 新增：获取不等式数量

private:
    VectorXd H_nonzeros_;
    VectorXd A_nonzeros_;
    VectorXd Ah_nonzeros_; // 新增：用于接收 A_h 的非零元素
};

} // namespace kronos

#endif // KRONOS_NLP_WRAPPER_HPP