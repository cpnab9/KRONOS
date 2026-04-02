// KRONOS/src/main.cpp
#include <iostream>
#include <iomanip>
#include "kronos/solver/fatrop_wrapper.hpp"

// 引入自动生成的最速降线算例底层的 C 函数声明
extern "C" {
    #include "kronos_nlp_functions.h"
}

int main() {
    std::cout << "[KRONOS] Initializing Flight Trajectory Optimizer..." << std::endl;
    
    try {
        // ==========================================
        // 【算例定义区】：配置最速降线问题的求解器函数
        // ==========================================
        kronos::CasadiSolverFunctions brachio_funcs;
        brachio_funcs.work = kronos_nlp_work;
        brachio_funcs.sparsity_out = kronos_nlp_sparsity_out;
        brachio_funcs.incref = kronos_nlp_incref;
        brachio_funcs.decref = kronos_nlp_decref;
        brachio_funcs.checkout = kronos_nlp_checkout;
        brachio_funcs.release = kronos_nlp_release;
        brachio_funcs.eval = kronos_nlp;

        // ==========================================
        // 【求解器执行区】：实例化通用求解器并传入算例
        // ==========================================
        kronos::FatropWrapper mpc_solver(brachio_funcs);
        std::cout << "[KRONOS] Solver memory allocated successfully. Zero allocation mode active." << std::endl;

        std::cout << "[KRONOS] Launching fatrop solver for Brachistochrone problem..." << std::endl;
        mpc_solver.solve();
        std::cout << "[KRONOS] Optimization converged!" << std::endl;

        // ==========================================
        // 【结果解析区】：目前暂留在此处，后续可进一步封装
        // ==========================================
        const auto& sol = mpc_solver.get_solution();
        
        if (sol.size() < 4) {
            throw std::runtime_error("Solution vector is too small!");
        }

        int final_node_idx = sol.size() - 4; 
        
        double x_end = sol[final_node_idx + 0];
        double y_end = sol[final_node_idx + 1];
        double v_end = sol[final_node_idx + 2];
        double tf    = sol[final_node_idx + 3];
        
        std::cout << "\n------------------------------------------" << std::endl;
        std::cout << " Optimal Descent Time (tf) : " << std::fixed << std::setprecision(4) << tf << " seconds" << std::endl;
        std::cout << " Target Reached (x, y)     : (" << x_end << ", " << y_end << ")" << std::endl;
        std::cout << " Terminal Velocity (v)     : " << v_end << " m/s" << std::endl;
        std::cout << " Expected Target (x, y)    : (10.0000, -5.0000)" << std::endl;
        std::cout << "------------------------------------------\n" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "[FATAL ERROR] " << e.what() << std::endl;
        return -1;
    }

    return 0;
}