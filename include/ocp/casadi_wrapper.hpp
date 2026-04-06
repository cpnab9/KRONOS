#pragma once

extern "C" {
    int eval_f_dyn(const double** arg, double** res, int* iw, double* w, int mem);
    int eval_g_eq(const double** arg, double** res, int* iw, double* w, int mem);
    int eval_J_BAbt(const double** arg, double** res, int* iw, double* w, int mem);
    int eval_J_Ggt(const double** arg, double** res, int* iw, double* w, int mem);
    int eval_H_RSQrqt(const double** arg, double** res, int* iw, double* w, int mem);
}

inline void call_casadi(int (*f)(const double**, double**, int*, double*, int), 
                        const double* a1, const double* a2, double* r1) {
    const double* args[2] = {a1, a2};
    double* res[1] = {r1};
    f(args, res, nullptr, nullptr, 0);
}

inline void call_casadi_hess(int (*f)(const double**, double**, int*, double*, int), 
                             const double* a1, const double* a2, const double* a3, const double* a4, double* r1) {
    const double* args[4] = {a1, a2, a3, a4};
    double* res[1] = {r1};
    f(args, res, nullptr, nullptr, 0);
}