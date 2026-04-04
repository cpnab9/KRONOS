// KRONOS/include/kronos/utils/library_loader.hpp
#pragma once
#include <dlfcn.h>
#include <stdexcept>
#include <iostream>
#include <vector>
#include "kronos/solver/fatrop_wrapper.hpp"

namespace kronos {

// 定义连续动力学函数的函数指针类型
typedef int (*CasadiContDynamicsFunc)(const double**, double**, long long int*, double*, int);

// ==========================================
// 【新增】定义元数据 C 接口的函数指针类型
// ==========================================
typedef void (*GetDimsFunc)(int*, int*, int*, int*, int*);
typedef void (*GetTauFunc)(double*);
typedef void (*GetBoundsFunc)(double*, double*);
typedef void (*GetWeightsFunc)(double*);

// 【新增】用于在 C++ 中保存算例信息的结构体
struct ProblemInfo {
    int nx;
    int nu_base;
    int d;
    int N;
    int tf_index;
    std::vector<double> tau_root;
    std::vector<double> x0_guess;
    std::vector<double> xf_guess;
    std::vector<double> error_weights;
};

class LibraryLoader {
public:
    static CasadiSolverFunctions load(const std::string& so_path) {
        void* handle = dlopen(so_path.c_str(), RTLD_LAZY);
        if (!handle) throw std::runtime_error("Cannot open library: " + std::string(dlerror()));

        CasadiSolverFunctions funcs;
        funcs.work = (int (*)(casadi_int*, casadi_int*, casadi_int*, casadi_int*))dlsym(handle, "kronos_nlp_work");
        funcs.eval = (int (*)(const casadi_real**, casadi_real**, casadi_int*, casadi_real*, int))dlsym(handle, "kronos_nlp");
        funcs.sparsity_out = (const casadi_int* (*)(casadi_int))dlsym(handle, "kronos_nlp_sparsity_out");
        funcs.incref = (void (*)(void))dlsym(handle, "kronos_nlp_incref");
        funcs.decref = (void (*)(void))dlsym(handle, "kronos_nlp_decref");
        funcs.checkout = (int (*)(void))dlsym(handle, "kronos_nlp_checkout");
        funcs.release = (void (*)(int))dlsym(handle, "kronos_nlp_release");

        if (!funcs.work || !funcs.eval) throw std::runtime_error("Required NLP symbols not found in " + so_path);
        
        return funcs;
    }

    static CasadiContDynamicsFunc load_f_cont(const std::string& so_path) {
        void* handle = dlopen(so_path.c_str(), RTLD_LAZY); 
        if (!handle) throw std::runtime_error("Cannot open library: " + std::string(dlerror()));

        auto f_cont = (CasadiContDynamicsFunc)dlsym(handle, "kronos_f_cont");
        if (!f_cont) {
            std::cerr << "[WARNING] kronos_f_cont not found in " << so_path << ". Error analysis will be disabled." << std::endl;
        }
        return f_cont;
    }

    // ==========================================
    // 【新增】单独加载并解析动态元数据
    // ==========================================
    static ProblemInfo load_problem_info(const std::string& so_path) {
        void* handle = dlopen(so_path.c_str(), RTLD_LAZY);
        if (!handle) throw std::runtime_error("Cannot open library: " + std::string(dlerror()));

        // 获取 metadata.c 中的函数符号
        GetDimsFunc get_dims = (GetDimsFunc)dlsym(handle, "kronos_get_dimensions");
        GetTauFunc get_tau = (GetTauFunc)dlsym(handle, "kronos_get_tau_root");
        GetBoundsFunc get_bounds = (GetBoundsFunc)dlsym(handle, "kronos_get_boundaries");
        GetWeightsFunc get_weights = (GetWeightsFunc)dlsym(handle, "kronos_get_error_weights");

        if (!get_dims || !get_tau || !get_bounds || !get_weights) {
            throw std::runtime_error("Metadata functions not found in library! Did you compile metadata.c?");
        }

        ProblemInfo info;
        // 1. 获取基础维度
        get_dims(&info.nx, &info.nu_base, &info.d, &info.N, &info.tf_index);

        // 2. 获取配点数组
        info.tau_root.resize(info.d);
        get_tau(info.tau_root.data());

        // 3. 获取初值猜测边界
        info.x0_guess.resize(info.nx);
        info.xf_guess.resize(info.nx);
        get_bounds(info.x0_guess.data(), info.xf_guess.data());

        // 4. 获取误差评估权重
        info.error_weights.resize(info.nx);
        get_weights(info.error_weights.data());

        return info;
    }
};

} // namespace kronos