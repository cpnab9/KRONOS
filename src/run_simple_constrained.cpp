// KRONOS/src/run_simple_constrained.cpp
#include <iostream>
#include <iomanip>
#include <vector>
#include "kronos/solver/fatrop_wrapper.hpp"

// 引入新生成的约束问题 C 函数
extern "C" {
    #include "constrained_nlp_functions.h"
}

int main() {
    std::cout << "[KRONOS] Initializing Simple Constrained Optimizer..." << std::endl;
    
    try {
        // 1. 打包函数指针
        kronos::CasadiSolverFunctions constrained_funcs;
        constrained_funcs.work = constrained_nlp_work;
        constrained_funcs.sparsity_out = constrained_nlp_sparsity_out;
        constrained_funcs.incref = constrained_nlp_incref;
        constrained_funcs.decref = constrained_nlp_decref;
        constrained_funcs.checkout = constrained_nlp_checkout;
        constrained_funcs.release = constrained_nlp_release;
        constrained_funcs.eval = constrained_nlp;

        // 2. 实例化求解器
        kronos::FatropWrapper solver(constrained_funcs);
        std::cout << "[KRONOS] Solver memory allocated. Launching fatrop..." << std::endl;

        // 3. 求解
        solver.solve();
        std::cout << "[KRONOS] Optimization converged!" << std::endl;

        // 4. 解析结果
        const auto& sol = solver.get_solution();
        
        std::cout << "\n--- Optimized Trajectory (Position, Velocity & Control) ---" << std::endl;
        std::cout << std::setw(5) << "Step" 
                  << std::setw(15) << "Position (x)" 
                  << std::setw(15) << "Velocity (v)" 
                  << std::setw(15) << "Control (a)" << std::endl;
        
        for (int k = 0; k <= 20; ++k) {
            // 由于我们在 Python 端使用了交替定义，
            // 每一个步长包含 nx(2) 和 nu(1)，所以 offset = k * 3
            int offset = k * 3; 
            
            double pos = sol[offset + 0];
            double vel = sol[offset + 1];
            // 最后一个节点 (k=20) 没有控制输入，手动补零打印
            double acc = (k < 20) ? sol[offset + 2] : 0.0;
            
            std::cout << std::setw(5) << k 
                      << std::setw(15) << std::fixed << std::setprecision(4) << pos 
                      << std::setw(15) << std::fixed << std::setprecision(4) << vel 
                      << std::setw(15) << std::fixed << std::setprecision(4) << acc << std::endl;
        }

        std::cout << "-----------------------------------------------------------\n" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "[FATAL ERROR] " << e.what() << std::endl;
        return -1;
    }

    return 0;
}