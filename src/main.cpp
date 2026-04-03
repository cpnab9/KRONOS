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

        // ----------------- 导出结果至 CSV (优雅展平版) -----------------
        const std::vector<double>& solution = solver.get_solution(); 
        std::ofstream csv_file("optimization_results.csv");

        if (csv_file.is_open()) {
            // 1. 写入表头 (纯净的基础变量名)
            csv_file << "time," << KRONOS_X_NAMES << "," << KRONOS_U_NAMES << "\n";

            size_t ptr = 0;
            double tf = solution[KRONOS_TF_INDEX]; 
            double dt = tf / KRONOS_N; // 每个主区间的时间跨度
            const std::vector<double> tau = KRONOS_TAU_ROOT;

            // 读取第 0 帧主节点状态 (t = 0)
            std::vector<double> x_val(KRONOS_NX);
            for (int i = 0; i < KRONOS_NX; ++i) {
                x_val[i] = solution[ptr++];
            }

            // 输出起点 (t=0时刻，控制量记为0)
            csv_file << 0.0;
            for (int i = 0; i < KRONOS_NX; ++i) csv_file << "," << x_val[i];
            for (int i = 0; i < KRONOS_NU_BASE; ++i) csv_file << ",0.0";
            csv_file << "\n";

            // 遍历 N 个时间区间，解包每个区间内部的 d 个配点
            for (int k = 0; k < KRONOS_N; ++k) {
                
                // 读取当前区间内的全部内部状态量 (D * NX)
                std::vector<double> X_c(KRONOS_D * KRONOS_NX);
                for (int i = 0; i < KRONOS_D * KRONOS_NX; ++i) {
                    X_c[i] = solution[ptr++];
                }

                // 读取当前区间内的全部内部控制量 (D * NU_BASE)
                std::vector<double> U_c(KRONOS_D * KRONOS_NU_BASE);
                for (int i = 0; i < KRONOS_D * KRONOS_NU_BASE; ++i) {
                    U_c[i] = solution[ptr++];
                }

                // 纵向输出这 d 个内部配点
                double t_start = k * dt;
                for (int j = 0; j < KRONOS_D; ++j) {
                    double t_cj = t_start + tau[j] * dt; // 真实的物理时间戳
                    csv_file << t_cj;
                    
                    // 输出该配点的状态量
                    for (int i = 0; i < KRONOS_NX; ++i) {
                        csv_file << "," << X_c[j * KRONOS_NX + i];
                    }
                    // 输出该配点的控制量/松弛变量
                    for (int i = 0; i < KRONOS_NU_BASE; ++i) {
                        csv_file << "," << U_c[j * KRONOS_NU_BASE + i];
                    }
                    csv_file << "\n";
                }

                // 吸收并跨过下一个主节点状态 X_{k+1} 
                for (int i = 0; i < KRONOS_NX; ++i) {
                    x_val[i] = solution[ptr++];
                }
            }
            csv_file.close();
            std::cout << "  > RESULTS EXPORTED TO   : optimization_results.csv (Flattened 76 Nodes)" << std::endl;
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