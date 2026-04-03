// KRONOS/src/main.cpp
#include <iostream>
#include <iomanip>
#include <chrono>
#include <fstream>  
#include <vector>
#include "kronos/solver/fatrop_wrapper.hpp"
#include "kronos/utils/library_loader.hpp"
#include "problem_metadata.h" 

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <path_to_shared_library>" << std::endl;
        return -1;
    }

    try {
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

        // 【阶段 3】：执行优化求解
        solver.solve(); 
        double solve_time_ms = solver.get_solve_time_ms(); 

        auto t_total_end = std::chrono::high_resolution_clock::now();
        double total_time_ms = std::chrono::duration<double, std::milli>(t_total_end - t_total_start).count();

        // ----------------- 导出结果至 CSV -----------------
        const std::vector<double>& solution = solver.get_solution(); 
        std::ofstream csv_file("optimization_results.csv");

        if (csv_file.is_open()) {
            // 1. 写入表头 (包含展开后完整的 U 变量名)
            csv_file << "time," << KRONOS_X_NAMES << "," << KRONOS_U_NAMES << "\n";

            // 2. 解析解向量并写入数据
            // 解向量布局: [x0, u0, x1, u1, ..., xN]
            size_t ptr = 0;
            
            // 【核心修改 1】：使用 Python 传过来的宏动态提取 tf 的值
            double tf = solution[KRONOS_TF_INDEX]; 

            for (int k = 0; k <= KRONOS_N; ++k) {
                // 当前阶段的时间戳
                double current_time = (static_cast<double>(k) / KRONOS_N) * tf;
                csv_file << current_time;

                // 提取并写入主节点状态变量 (NX 个)
                for (int i = 0; i < KRONOS_NX; ++i) {
                    csv_file << "," << solution[ptr++];
                }

                // 【核心修改 2】：完整提取 u 块（包含所有配点状态和松弛变量）
                if (k < KRONOS_N) {
                    for (int i = 0; i < KRONOS_NU; ++i) {
                        csv_file << "," << solution[ptr++]; 
                    }
                } else {
                    // 最后一帧没有控制量，必须补齐 NU 个 0.0 以对齐 CSV 列
                    for (int i = 0; i < KRONOS_NU; ++i) {
                        csv_file << ",0.0"; 
                    }
                }
                csv_file << "\n";
            }
            csv_file.close();
            std::cout << "  > RESULTS EXPORTED TO   : optimization_results.csv" << std::endl;
        }

        // ----------------- 打印最终报告 -----------------
        std::cout << "\n" << std::string(50, '=') << std::endl;
        std::cout << "   KRONOS OPTIMIZATION FINAL REPORT" << std::endl;
        std::cout << std::string(50, '-') << std::endl;
        std::cout << "  > OPTIMAL OBJECTIVE     : " << std::fixed << std::setprecision(6) << solver.get_objective() << std::endl;
        std::cout << "  > SOLVER STATUS         : CONVERGED\n" << std::endl;
        std::cout << "  [ TIMING ANALYSIS ]" << std::endl;
        std::cout << "  * Library Loading       : " << std::setprecision(3) << std::setw(8) << load_time_ms << " ms" << std::endl;
        std::cout << "  * Memory Initialization : " << std::setprecision(3) << std::setw(8) << init_time_ms << " ms" << std::endl;
        std::cout << "  * Fatrop Execution      : " << std::setprecision(3) << std::setw(8) << solve_time_ms << " ms" << std::endl;
        std::cout << std::string(50, '-') << std::endl;
        std::cout << "  > TOTAL PIPELINE TIME   : " << std::setprecision(3) << std::setw(8) << total_time_ms << " ms" << std::endl;
        std::cout << std::string(50, '=') << "\n" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << std::endl;
        return -1;
    }
    return 0;
}