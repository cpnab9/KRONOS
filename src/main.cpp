// KRONOS/src/main.cpp
#include <iostream>
#include <iomanip>
#include "kronos/solver/fatrop_wrapper.hpp"

int main() {
    std::cout << "[KRONOS] Initializing Flight Trajectory Optimizer..." << std::endl;
    
    try {
        kronos::FatropWrapper mpc_solver;
        std::cout << "[KRONOS] Solver memory allocated successfully. Zero allocation mode active." << std::endl;

        std::cout << "[KRONOS] Launching fatrop solver for Brachistochrone problem..." << std::endl;
        mpc_solver.solve();
        std::cout << "[KRONOS] Optimization converged!" << std::endl;

        // 获取最优解结果
        const auto& sol = mpc_solver.get_solution();
        
        // ==========================================
        // 【核心修复】：动态解析，不再硬编码 N！
        // 数组的最后 4 个元素永远是终点状态: [x, y, v, tf]
        // ==========================================
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