#pragma once
#include <dlfcn.h>
#include <string>
#include <stdexcept>
#include "kronos/solver/fatrop_wrapper.hpp"

namespace kronos {
class LibraryLoader {
public:
    static CasadiSolverFunctions load(const std::string& so_path) {
        void* handle = dlopen(so_path.c_str(), RTLD_LAZY);
        if (!handle) throw std::runtime_error("Cannot open library: " + std::string(dlerror()));

        CasadiSolverFunctions funcs;
        funcs.work = (int (*)(casadi_int*, casadi_int*, casadi_int*, casadi_int*))dlsym(handle, "kronos_nlp_work");
        funcs.eval = (int (*)(const casadi_real**, casadi_real**, casadi_int*, casadi_real*, int))dlsym(handle, "kronos_nlp");
        // 【新增】：加载输入端稀疏矩阵信息（用于获取 p 和 x0 的维度）
        funcs.sparsity_in = (const casadi_int* (*)(casadi_int))dlsym(handle, "kronos_nlp_sparsity_in");
        funcs.sparsity_out = (const casadi_int* (*)(casadi_int))dlsym(handle, "kronos_nlp_sparsity_out");
        funcs.incref = (void (*)(void))dlsym(handle, "kronos_nlp_incref");
        funcs.decref = (void (*)(void))dlsym(handle, "kronos_nlp_decref");
        funcs.checkout = (int (*)(void))dlsym(handle, "kronos_nlp_checkout");
        funcs.release = (void (*)(int))dlsym(handle, "kronos_nlp_release");

        // 【新增】：加载连续动力学函数 f_cont
        funcs.f_cont_eval = (int (*)(const casadi_real**, casadi_real**, casadi_int*, casadi_real*, int))dlsym(handle, "f_cont");

        if (!funcs.work || !funcs.eval || !funcs.f_cont_eval || !funcs.sparsity_in) {
            throw std::runtime_error("Required symbols (kronos_nlp, sparsity_in, or f_cont) not found in " + so_path);
        }
        return funcs;
    }
};
}