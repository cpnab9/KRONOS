// KRONOS/include/kronos/solver/fatrop_wrapper.hpp
#pragma once

#include <vector>
#include <string>
#include <stdexcept>

// 适配 CasADi 生成代码的基础数据类型
#ifndef casadi_int
#define casadi_int long long int
#endif

#ifndef casadi_real
#define casadi_real double
#endif

namespace kronos {

/**
 * @brief CasADi 生成的 C API 函数指针包装
 * 用于将具体的算例逻辑（.so）注入到通用的求解器中
 */
struct CasadiSolverFunctions {
    int (*work)(casadi_int* sz_arg, casadi_int* sz_res, casadi_int* sz_iw, casadi_int* sz_w);
    const casadi_int* (*sparsity_out)(casadi_int i);
    void (*incref)(void);
    void (*decref)(void);
    int (*checkout)(void);
    void (*release)(int mem);
    int (*eval)(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem);
};

/**
 * @brief 通用的 fatrop 求解器包装类
 * 负责内存管理、算例加载后的接口绑定以及求解触发
 */
class FatropWrapper {
public:
    /**
     * @brief 构造函数：接收算例函数指针并预分配所有必要内存
     * @param funcs 包含算例接口的结构体
     */
    explicit FatropWrapper(const CasadiSolverFunctions& funcs);
    
    ~FatropWrapper();

    // 禁用拷贝以保护内存安全
    FatropWrapper(const FatropWrapper&) = delete;
    FatropWrapper& operator=(const FatropWrapper&) = delete;

    /**
     * @brief 触发单次 NLP 求解
     */
    void solve();

    /**
     * @brief 获取优化目标函数的值 (f)
     * 对应 Python 导出时 to_function 的第一个输出
     */
    double get_objective() const;

    /**
     * @brief 获取所有设计变量的最优解向量 (x)
     * 对应 Python 导出时 to_function 的第二个输出
     */
    const std::vector<double>& get_solution() const;

private:
    CasadiSolverFunctions funcs_; // 当前加载的算例接口
    
    int mem_; // 线程本地内存句柄
    casadi_int sz_arg_, sz_res_, sz_iw_, sz_w_;
    
    // 求解器工作空间缓冲区（零动态内存分配运行模式）
    std::vector<const double*> arg_;
    std::vector<double*> res_;
    std::vector<casadi_int> iw_;
    std::vector<double> w_;
    
    // 结果存储
    double obj_value_ = 0.0;          // 目标函数值
    std::vector<double> res_buffer_; // 设计变量向量
};

} // namespace kronos