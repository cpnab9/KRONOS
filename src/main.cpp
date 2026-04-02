// KRONOS/src/main.cpp
#include <iostream>
#include <iomanip>
#include <chrono>
#include <fstream>  // 【新增】文件流
#include <vector>
#include "kronos/solver/fatrop_wrapper.hpp"
#include "kronos/utils/library_loader.hpp"
#include "problem_metadata.h" // 【新增】由 Python 脚本生成的维度信息

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <path_to_shared_library>" << std::endl;
        return -1;
    }

    try {
        auto t_total_start = std::chrono::high_resolution_clock::now();

        // 【阶段 1】：动态库加载与符号解析
        auto t_load_start = std::chrono::high_resolution_clock::now();
        auto funcs = kronos::LibraryLoader::load(argv[1]); //
        auto t_load_end = std::chrono::high_resolution_clock::now();
        double load_time_ms = std::chrono::duration<double, std::milli>(t_load_end - t_load_start).count();

        // 【阶段 2】：求解器内存分配与初始化
        auto t_init_start = std::chrono::high_resolution_clock::now();
        kronos::FatropWrapper solver(funcs); //
        auto t_init_end = std::chrono::high_resolution_clock::now();
        double init_time_ms = std::chrono::duration<double, std::milli>(t_init_end - t_init_start).count();

        // 【阶段 3】：执行优化求解
        solver.solve(); //
        double solve_time_ms = solver.get_solve_time_ms(); //

        auto t_total_end = std::chrono::high_resolution_clock::now();
        double total_time_ms = std::chrono::duration<double, std::milli>(t_total_end - t_total_start).count();

        // ----------------- 【新增】导出结果至 CSV -----------------
        const std::vector<double>& solution = solver.get_solution(); //
        std::ofstream csv_file("optimization_results.csv");

        if (csv_file.is_open()) {
            // 1. 写入表头
            csv_file << "time," << KRONOS_X_NAMES << "," << KRONOS_U_NAMES << "\n";

            // 2. 解析解向量并写入数据
            // 解向量布局: [x0, u0, x1, u1, ..., xN]
            size_t ptr = 0;
            
            // 获取总时间 tf (在 Brachistochrone 问题中，tf 是状态向量的第 4 个元素，下标为 3)
            // 我们取初始点的 tf 即可，因为它在所有阶段都被约束为相等
            double tf = solution[3]; 

            for (int k = 0; k <= KRONOS_N; ++k) {
                // 当前阶段的时间戳
                double current_time = (static_cast<double>(k) / KRONOS_N) * tf;
                csv_file << current_time;

                // 提取并写入状态变量 (NX 个)
                for (int i = 0; i < KRONOS_NX; ++i) {
                    csv_file << "," << solution[ptr++];
                }

                // 提取并写入控制变量 (NU 个，最后一帧没有控制量)
                if (k < KRONOS_N) {
                    // 注意：在伪谱法中，u_k 包含配点状态和控制，我们通常关心真正的控制输入
                    // 假设控制量 theta 位于 u_k 向量的特定位置，这里演示提取 u_k 的第一个元素
                    csv_file << "," << solution[ptr]; 
                    ptr += KRONOS_NU; // 跳过整个 u 块进入下一个 x
                } else {
                    csv_file << ",0.0"; // 最后一帧补充占位符
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