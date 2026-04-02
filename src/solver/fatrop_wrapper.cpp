// KRONOS/src/solver/fatrop_wrapper.cpp
#include "kronos/solver/fatrop_wrapper.hpp"

namespace kronos {

FatropWrapper::FatropWrapper(const CasadiSolverFunctions& funcs) : funcs_(funcs) {
    // 1. 获取工作空间维度
    funcs_.work(&sz_arg_, &sz_res_, &sz_iw_, &sz_w_);
    
    // 2. 预分配内存 (保持零动态内存分配)
    arg_.resize(sz_arg_, nullptr);
    res_.resize(sz_res_, nullptr);
    iw_.resize(sz_iw_, 0);
    w_.resize(sz_w_, 0.0);

    // 3. 映射输出缓冲区
    // 根据 Python 导出的 [opti.f, opti.x]，res_ 指针数组应如下排列：
    
    // [输出 0]: 目标函数值 (Scalar)
    if (sz_res_ > 0) {
        res_[0] = &obj_value_;
    }

    // [输出 1]: 设计变量向量 (Vector)
    if (sz_res_ > 1) {
        const casadi_int* sp_x = funcs_.sparsity_out(1); 
        casadi_int nrow_x = sp_x[0]; 
        res_buffer_.resize(nrow_x, 0.0);
        res_[1] = res_buffer_.data();
    }

    // 4. 初始化 CasADi 线程内存句柄
    funcs_.incref();
    mem_ = funcs_.checkout();
}

FatropWrapper::~FatropWrapper() {
    if (funcs_.release) {
        funcs_.release(mem_);
    }
    if (funcs_.decref) {
        funcs_.decref();
    }
}

void FatropWrapper::solve() {
    // 调用底层的 fatrop 求解逻辑
    if (funcs_.eval(arg_.data(), res_.data(), iw_.data(), w_.data(), mem_)) {
        throw std::runtime_error("Fatrop solver returned an error during execution.");
    }
}

double FatropWrapper::get_objective() const {
    return obj_value_;
}

const std::vector<double>& FatropWrapper::get_solution() const {
    return res_buffer_;
}

} // namespace kronos