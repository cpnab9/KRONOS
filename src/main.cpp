#include <iostream>
#include <iomanip>
#include <chrono> // 引入高精度计时库

#include "kronos_types.hpp"
#include "kronos_nlp_wrapper.hpp"
#include "kronos_kkt_solver.hpp"
#include "kronos_optimizer.hpp"
#include "kronos_config.h"

using namespace std;
using namespace kronos;

int main() {
    cout << "========================================\n";
    cout << "  KRONOS Trajectory Optimizer v2.0\n";
    cout << "  Architecture: NLP -> Newton -> Schur\n";
    cout << "========================================\n";

    // ==========================================
    // ⏱️ 阶段一：初始化与内存分配计时
    // ==========================================
    auto t_init_start = std::chrono::steady_clock::now();

    // 1. 初始化模型
    NlpWrapper nlp;

    // 2. 初始化底层线性求解器
    SchurKktSolver linear_solver(1e-5, 1e-8);

    // 3. 初始化优化算法（将模型和求解器注入进去）
    NewtonOptimizer optimizer(nlp, linear_solver);

    // 4. 准备初值 (Eigen::Map 是一种零拷贝操作，非常快)
    VectorXd w = Eigen::Map<const VectorXd>(w_init, nlp.get_nw());
    VectorXd lam = Eigen::Map<const VectorXd>(lam_init, nlp.get_ng());

    auto t_init_end = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> t_init = t_init_end - t_init_start;


    // ==========================================
    // ⏱️ 阶段二：核心数学优化过程计时 (最关键)
    // ==========================================
    cout << "[KRONOS] Starting optimization loop...\n";
    auto t_solve_start = std::chrono::steady_clock::now();

    // 5. 求解
    bool success = optimizer.optimize(w, lam);

    auto t_solve_end = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> t_solve = t_solve_end - t_solve_start;


    // ==========================================
    // 📊 打印结果与性能报告
    // ==========================================
    cout << "\n========================================\n";
    if (success) {
        // 假设最后一个状态变量是时间 T
        double T_opt = w(nlp.get_nw() - 1);
        cout << "🎯 Optimization Success!\n";
        cout << "   Optimal Time T = " << fixed << setprecision(4) << T_opt << " s\n";
    } else {
        cout << "⚠️ Optimization Failed to converge.\n";
    }

    // 打印耗时报告 (精确到小数点后 3 位毫秒)
    cout << "\n--- ⏱️ Performance Profiling ---\n";
    cout << "   Initialization : " << fixed << setprecision(3) << t_init.count() << " ms\n";
    cout << "   Optimization   : " << fixed << setprecision(3) << t_solve.count() << " ms\n";
    cout << "   Total Time     : " << fixed << setprecision(3) << (t_init + t_solve).count() << " ms\n";
    cout << "========================================\n";

    return 0;
}