// KRONOS/include/kronos/solver/fatrop_wrapper.hpp
#pragma once

#include <vector>
#include <stdexcept>

// 适配 CasADi 生成的默认数据类型
#ifndef casadi_int
#define casadi_int long long int
#endif

#ifndef casadi_real
#define casadi_real double
#endif

namespace kronos {

// 【新增】：CasADi 生成的 C 函数签名接口，用于彻底解耦具体算例
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
    // 构造函数现在接收一组函数指针，变成通用求解器
    explicit FatropWrapper(const CasadiSolverFunctions& funcs);
    ~FatropWrapper();

    void solve();
    const std::vector<double>& get_solution() const;

private:
    CasadiSolverFunctions funcs_; // 保存当前算例的函数指针
    casadi_int sz_arg_, sz_res_, sz_iw_, sz_w_;
    
    std::vector<const double*> arg_;
    std::vector<double*> res_;
    std::vector<casadi_int> iw_;
    std::vector<double> w_;
    std::vector<double> res_buffer_;
    int mem_;
};

} // namespace kronos