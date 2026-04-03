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

    // 【新增】：根据导出的参数维度分配输入内存
    // 在 Python 端我们定义的签名是: p (参数/网格), x0 (初值猜测)
    if (sz_arg_ > 0) {
        const casadi_int* sp_p = funcs_.sparsity_in(0);
        param_buffer_.resize(sp_p[0], 1.0); // 初始化为全 1（外部应该立刻用 set_mesh_fractions 覆盖）
        arg_[0] = param_buffer_.data();
    }
    if (sz_arg_ > 1) {
        const casadi_int* sp_x0 = funcs_.sparsity_in(1);
        x0_buffer_.resize(sp_x0[0], 0.0);   // 初始化为全 0
        arg_[1] = x0_buffer_.data();
    }

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

double FatropWrapper::get_solve_time_ms() const {
    return solve_time_ms_;
}

// 【新增】：设置网格占比参数
void FatropWrapper::set_mesh_fractions(const std::vector<double>& fractions) {
    if (fractions.size() == param_buffer_.size()) {
        param_buffer_ = fractions;
    } else {
        throw std::runtime_error("Mesh fractions size mismatch.");
    }
}

// 【新增】：设置热启动初值猜测
void FatropWrapper::set_initial_guess(const std::vector<double>& x0) {
    if (x0.size() == x0_buffer_.size()) {
        x0_buffer_ = x0;
    } else {
        throw std::runtime_error("Initial guess size mismatch.");
    }
}

} // namespace kronos