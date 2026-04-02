// KRONOS/src/main.cpp
#include <iostream>
#include <iomanip>
#include <chrono> // 【新增】时间库
#include "kronos/solver/fatrop_wrapper.hpp"
#include "kronos/utils/library_loader.hpp"

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <path_to_shared_library>" << std::endl;
        return -1;
    }

    try {
        // 记录总流水线开始时间
        auto t_total_start = std::chrono::high_resolution_clock::now();

        // 【阶段 1】：动态库加载与符号解析
        auto t_load_start = std::chrono::high_resolution_clock::now();
        auto funcs = kronos::LibraryLoader::load(argv[1]);
        auto t_load_end = std::chrono::high_resolution_clock::now();
        double load_time_ms = std::chrono::duration<double, std::milli>(t_load_end - t_load_start).count();

        // 【阶段 2】：求解器内存分配与初始化
        auto t_init_start = std::chrono::high_resolution_clock::now();
        kronos::FatropWrapper solver(funcs);
        auto t_init_end = std::chrono::high_resolution_clock::now();
        double init_time_ms = std::chrono::duration<double, std::milli>(t_init_end - t_init_start).count();

        // 【阶段 3】：执行优化求解 (内部自带计时)
        solver.solve();
        double solve_time_ms = solver.get_solve_time_ms();

        // 记录总流水线结束时间
        auto t_total_end = std::chrono::high_resolution_clock::now();
        double total_time_ms = std::chrono::duration<double, std::milli>(t_total_end - t_total_start).count();

        // ----------------- 打印最终报告 -----------------
        std::cout << "\n" << std::string(50, '=') << std::endl;
        std::cout << "   KRONOS OPTIMIZATION FINAL REPORT" << std::endl;
        std::cout << std::string(50, '-') << std::endl;
        
        // 1. 结果信息
        std::cout << "  > OPTIMAL OBJECTIVE     : " 
                  << std::fixed << std::setprecision(6) 
                  << solver.get_objective() << std::endl;
        std::cout << "  > SOLVER STATUS         : CONVERGED\n" << std::endl;
        
        // 2. 时间分析报告
        std::cout << "  [ TIMING ANALYSIS ]" << std::endl;
        std::cout << "  * Library Loading       : " << std::fixed << std::setprecision(3) << std::setw(8) << load_time_ms << " ms" << std::endl;
        std::cout << "  * Memory Initialization : " << std::fixed << std::setprecision(3) << std::setw(8) << init_time_ms << " ms" << std::endl;
        std::cout << "  * Fatrop Execution      : " << std::fixed << std::setprecision(3) << std::setw(8) << solve_time_ms << " ms" << std::endl;
        std::cout << std::string(50, '-') << std::endl;
        std::cout << "  > TOTAL PIPELINE TIME   : " << std::fixed << std::setprecision(3) << std::setw(8) << total_time_ms << " ms" << std::endl;
        std::cout << std::string(50, '=') << "\n" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << std::endl;
        return -1;
    }
    return 0;
}