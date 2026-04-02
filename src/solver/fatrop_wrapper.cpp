// KRONOS/src/solver/fatrop_wrapper.cpp
#include "kronos/solver/fatrop_wrapper.hpp"
#include <chrono> // 【新增】时间库

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

void FatropWrapper::solve() {
    // 【修改】包裹 eval 进行耗时统计
    auto start_time = std::chrono::high_resolution_clock::now();
    
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

// 【新增】返回统计的时间
double FatropWrapper::get_solve_time_ms() const {
    return solve_time_ms_;
}

} // namespace kronos