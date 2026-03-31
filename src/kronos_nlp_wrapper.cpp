#include "kronos_nlp_wrapper.hpp"

// 引入生成的 C 代码和宏
extern "C" {
    #include "kkt_funcs.h"
}
#include "kronos_config.h"

namespace kronos {

NlpWrapper::NlpWrapper() : n_w_(KRONOS_N_W), n_g_(KRONOS_N_G) {
    // 初始化时获取 CasADi 需要的内存大小
    casadi_int sz_arg, sz_res, sz_iw, sz_w;
    kkt_func_work(&sz_arg, &sz_res, &sz_iw, &sz_w);
    
    // 预分配内存空间
    iw_.resize(sz_iw);
    work_.resize(sz_w);
}

void NlpWrapper::evaluate(const VectorXd& w, const VectorXd& lam,
                          MatrixXd& H, MatrixXd& A, VectorXd& grad_L, VectorXd& g) {
    const double* arg[2] = {w.data(), lam.data()};
    double* res[4] = {H.data(), A.data(), grad_L.data(), g.data()};
    
    // 调用实际的 C 函数
    ::kkt_func(arg, res, iw_.data(), work_.data(), 0);
}

} // namespace kronos