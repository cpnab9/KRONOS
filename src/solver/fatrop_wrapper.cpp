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

    // 0: Objective value
    if (sz_res_ > 0) {
        res_[0] = &obj_value_;
    }

    // 1: Primal solution (x)
    if (sz_res_ > 1) {
        const casadi_int* sp_x = funcs_.sparsity_out(1); 
        res_buffer_.resize(sp_x[0], 0.0);
        res_[1] = res_buffer_.data();
    }
    
    // 2: Dual solution (lam_g)
    if (sz_res_ > 2) {
        const casadi_int* sp_lam_g = funcs_.sparsity_out(2); 
        res_lam_g_.resize(sp_lam_g[0], 0.0);
        res_[2] = res_lam_g_.data();
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

void FatropWrapper::set_initial_guess(const std::vector<double>& x0, 
                                      const std::vector<double>& lam_g0) {
    if (!x0.empty()) {
        initial_guess_ = x0; 
        if (sz_arg_ > 0) arg_[0] = initial_guess_.data(); 
    }
    if (!lam_g0.empty()) {
        lam_g_guess_ = lam_g0; 
        if (sz_arg_ > 1) arg_[1] = lam_g_guess_.data(); 
    }
}

void FatropWrapper::set_parameters(const std::vector<double>& p) {
    if (!p.empty()) {
        parameters_ = p;
        // 核心修正：由于 lam_x 被移除，动态网格参数的索引前移到 2
        if (sz_arg_ > 2) {
            arg_[2] = parameters_.data();
        }
    }
}

void FatropWrapper::solve() {
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

const std::vector<double>& FatropWrapper::get_lam_g() const {
    return res_lam_g_;
}

double FatropWrapper::get_solve_time_ms() const {
    return solve_time_ms_;
}

} // namespace kronos