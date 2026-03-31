#ifndef KRONOS_NLP_WRAPPER_HPP
#define KRONOS_NLP_WRAPPER_HPP

#include "kronos_types.hpp"
#include "../generated/kronos_config.h"

namespace kronos {

class NlpWrapper {
public:
    NlpWrapper();

    // 评估 KKT 系统 (使用稀疏矩阵)
    void evaluate(const VectorXd& w, const VectorXd& lam, 
                  SparseMatrixXd& H, SparseMatrixXd& A, 
                  VectorXd& grad_L, VectorXd& g);

    // 恢复缺失的维度获取接口，直接返回配置文件中的宏
    int get_nw() const { return KRONOS_N_W; }
    int get_ng() const { return KRONOS_N_G; }

private:
    // 用于接收 CasADi 输出的非零元素一维数组
    VectorXd H_nonzeros_;
    VectorXd A_nonzeros_;
};

} // namespace kronos

#endif // KRONOS_NLP_WRAPPER_HPP