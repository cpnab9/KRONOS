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

    // 仅保留一般约束乘子 lam_g 的热启动接口
    void set_initial_guess(const std::vector<double>& x0, 
                           const std::vector<double>& lam_g0 = {});
    
    void set_parameters(const std::vector<double>& p);

    void solve();

    double get_objective() const;
    const std::vector<double>& get_solution() const;
    
    // 仅保留获取 lam_g 的接口
    const std::vector<double>& get_lam_g() const;
    
    double get_solve_time_ms() const;

private:
    CasadiSolverFunctions funcs_; 
    int mem_; 
    casadi_int sz_arg_, sz_res_, sz_iw_, sz_w_;
    
    std::vector<const double*> arg_;
    std::vector<double*> res_;
    std::vector<casadi_int> iw_;
    std::vector<double> w_;
    
    std::vector<double> initial_guess_;
    std::vector<double> lam_g_guess_;
    std::vector<double> parameters_;

    double obj_value_ = 0.0;          
    std::vector<double> res_buffer_; 
    std::vector<double> res_lam_g_; 
    
    double solve_time_ms_ = 0.0;
};

} // namespace kronos