#pragma once
#include <dlfcn.h>
#include "kronos/solver/fatrop_wrapper.hpp"

namespace kronos {
class LibraryLoader {
public:
    static CasadiSolverFunctions load(const std::string& so_path) {
        void* handle = dlopen(so_path.c_str(), RTLD_LAZY);
        if (!handle) throw std::runtime_error("Cannot open library: " + std::string(dlerror()));

        CasadiSolverFunctions funcs;
        // 使用 dlsym 查找统一的函数名
        funcs.work = (int (*)(casadi_int*, casadi_int*, casadi_int*, casadi_int*))dlsym(handle, "kronos_nlp_work");
        funcs.eval = (int (*)(const casadi_real**, casadi_real**, casadi_int*, casadi_real*, int))dlsym(handle, "kronos_nlp");
        funcs.sparsity_out = (const casadi_int* (*)(casadi_int))dlsym(handle, "kronos_nlp_sparsity_out");
        funcs.incref = (void (*)(void))dlsym(handle, "kronos_nlp_incref");
        funcs.decref = (void (*)(void))dlsym(handle, "kronos_nlp_decref");
        funcs.checkout = (int (*)(void))dlsym(handle, "kronos_nlp_checkout");
        funcs.release = (void (*)(int))dlsym(handle, "kronos_nlp_release");

        if (!funcs.work || !funcs.eval) throw std::runtime_error("Required symbols not found in " + so_path);
        return funcs;
    }
};
}