// KRONOS/src/solver/fatrop_wrapper.cpp
#include "kronos/solver/fatrop_wrapper.hpp"
#include <chrono>

namespace kronos {

FatropWrapper::FatropWrapper(const CasadiSolverFunctions& funcs) : funcs_(funcs) {
    funcs_.work(&sz_arg_, &sz_res_, &sz_iw_, &sz_w_);
    
    arg_.resize(sz_arg_, nullptr);
    res_.resize(sz_res_, nullptr);
    iw_.resize(sz_iw_, 0);
    w_.resize(sz_w_, 0.0);

    if (sz_res_ > 0) {
        res_[0] = &obj_value_;
    }

    if (sz_res_ > 1) {
        const casadi_int* sp_x = funcs_.sparsity_out(1); 
        casadi_int nrow_x = sp_x[0]; 
        res_buffer_.resize(nrow_x, 0.0);
        res_[1] = res_buffer_.data();
    }

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

// 【新增】设置初始猜测值 (映射到 arg[0])
void FatropWrapper::set_initial_guess(const std::vector<double>& x0) {
    if (x0.empty()) return;
    initial_guess_ = x0; 
    if (sz_arg_ > 0) {
        arg_[0] = initial_guess_.data(); 
    }
}

// 【新增】设置外部参数 (映射到 arg[1])
void FatropWrapper::set_parameters(const std::vector<double>& p) {
    if (p.empty()) return;
    parameters_ = p;
    if (sz_arg_ > 1) {
        arg_[1] = parameters_.data();
    }
}

void FatropWrapper::solve() {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // 调用生成的纯 C 函数，此时内部会读取被赋值的 arg_[0] 和 arg_[1]
    if (funcs_.eval(arg_.data(), res_.data(), iw_.data(), w_.data(), mem_)) {
        throw std::runtime_error("Fatrop solver returned an error during execution.");
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    solve_time_ms_ = std::chrono::duration<double, std::milli>(end_time - start_time).count();
}

double FatropWrapper::get_objective() const {
    return obj_value_;
}

const std::vector<double>& FatropWrapper::get_solution() const {
    return res_buffer_;
}

double FatropWrapper::get_solve_time_ms() const {
    return solve_time_ms_;
}

} // namespace kronos