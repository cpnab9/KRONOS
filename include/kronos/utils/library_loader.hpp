// KRONOS/include/kronos/utils/library_loader.hpp
#pragma once
#include <dlfcn.h>
#include <stdexcept>
#include <iostream>
#include "kronos/solver/fatrop_wrapper.hpp"

namespace kronos {

// 定义连续动力学函数的函数指针类型
// 签名遵循 CasADi 规范: int f(const double** arg, double** res, int* iw, double* w, int mem)
typedef int (*CasadiContDynamicsFunc)(const double**, double**, long long int*, double*, int);

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

    // 【新增】单独加载连续动力学函数
    static CasadiContDynamicsFunc load_f_cont(const std::string& so_path) {
        void* handle = dlopen(so_path.c_str(), RTLD_LAZY); // 同一库，引用计数增加
        if (!handle) throw std::runtime_error("Cannot open library: " + std::string(dlerror()));

        auto f_cont = (CasadiContDynamicsFunc)dlsym(handle, "kronos_f_cont");
        if (!f_cont) {
            std::cerr << "[WARNING] kronos_f_cont not found in " << so_path << ". Error analysis will be disabled." << std::endl;
        }
        return f_cont;
    }
};

} // namespace kronos