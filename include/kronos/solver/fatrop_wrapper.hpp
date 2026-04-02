// KRONOS/include/kronos/solver/fatrop_wrapper.hpp
#pragma once

#include <vector>
#include <stdexcept>
#include "kronos_nlp_functions.h" // 包含由 CasADi 生成的 C API 头文件

namespace kronos {

class FatropWrapper {
public:
    FatropWrapper();
    ~FatropWrapper();

    // 禁用拷贝构造和赋值操作，防止底层的内存引用计数混乱
    FatropWrapper(const FatropWrapper&) = delete;
    FatropWrapper& operator=(const FatropWrapper&) = delete;

    // 触发单次求解
    void solve();

    // 获取最优状态和控制量序列
    const std::vector<double>& get_solution() const;

private:
    int mem_; // 线程本地内存句柄
    casadi_int sz_arg_, sz_res_, sz_iw_, sz_w_;
    
    // 工作空间内存缓冲区
    std::vector<const double*> arg_;
    std::vector<double*> res_;
    std::vector<casadi_int> iw_;
    std::vector<double> w_;
    
    // 存储最终 opti.x 的结果缓冲区
    std::vector<double> res_buffer_;
};

} // namespace kronos