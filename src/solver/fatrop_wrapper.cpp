// KRONOS/src/solver/fatrop_wrapper.cpp
#include "kronos/solver/fatrop_wrapper.hpp"

namespace kronos {

FatropWrapper::FatropWrapper() {
    // 1. 获取模型所需的参数、结果、整型和浮点型工作空间的维度
    kronos_nlp_work(&sz_arg_, &sz_res_, &sz_iw_, &sz_w_);
    
    // 2. 预分配内存 (在机载系统启动时完成)
    arg_.resize(sz_arg_, nullptr);
    res_.resize(sz_res_, nullptr);
    iw_.resize(sz_iw_, 0);
    w_.resize(sz_w_, 0.0);

    // 3. 获取输出解向量的大小 (即 Python 脚本中 opti.x 的总长度)
    const casadi_int* sp = kronos_nlp_sparsity_out(0);
    casadi_int nrow = sp[0]; 
    res_buffer_.resize(nrow, 0.0);
    
    // 将结果缓冲区的地址绑定到 res 指针数组
    res_[0] = res_buffer_.data();

    // 4. Checkout 线程本地内存
    kronos_nlp_incref();
    mem_ = kronos_nlp_checkout();
}

FatropWrapper::~FatropWrapper() {
    // 资源释放
    kronos_nlp_release(mem_);
    kronos_nlp_decref();
}

void FatropWrapper::solve() {
    // 执行求解
    // 这里是对 fatrop 求解器的隐式调用
    if (kronos_nlp(arg_.data(), res_.data(), iw_.data(), w_.data(), mem_)) {
        throw std::runtime_error("Fatrop solver returned an error during execution.");
    }
}

const std::vector<double>& FatropWrapper::get_solution() const {
    return res_buffer_;
}

} // namespace kronos