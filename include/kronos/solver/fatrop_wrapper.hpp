// KRONOS/include/kronos/solver/fatrop_wrapper.hpp
#pragma once

#include <vector>
#include <string>
#include <stdexcept>

#ifndef casadi_int
#define casadi_int long long int
#endif

#ifndef casadi_real
#define casadi_real double
#endif

namespace kronos {

struct CasadiSolverFunctions {
    int (*work)(casadi_int* sz_arg, casadi_int* sz_res, casadi_int* sz_iw, casadi_int* sz_w);
    const casadi_int* (*sparsity_out)(casadi_int i);
    void (*incref)(void);
    void (*decref)(void);
    int (*checkout)(void);
    void (*release)(int mem);
    int (*eval)(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem);
};

class FatropWrapper {
public:
    explicit FatropWrapper(const CasadiSolverFunctions& funcs);
    ~FatropWrapper();

    FatropWrapper(const FatropWrapper&) = delete;
    FatropWrapper& operator=(const FatropWrapper&) = delete;

    void solve();

    double get_objective() const;
    const std::vector<double>& get_solution() const;
    
    // 【新增】获取求解器纯计算的耗时 (单位：毫秒)
    double get_solve_time_ms() const;

private:
    CasadiSolverFunctions funcs_; 
    
    int mem_; 
    casadi_int sz_arg_, sz_res_, sz_iw_, sz_w_;
    
    std::vector<const double*> arg_;
    std::vector<double*> res_;
    std::vector<casadi_int> iw_;
    std::vector<double> w_;
    
    double obj_value_ = 0.0;          
    std::vector<double> res_buffer_; 
    
    // 【新增】用于内部记录时间的变量
    double solve_time_ms_ = 0.0;
};

} // namespace kronos