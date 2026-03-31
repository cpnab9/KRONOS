#include <iostream>
#include <iomanip>
#include <chrono>

#include "kronos_types.hpp"
#include "kronos_nlp_wrapper.hpp"
#include "kronos_kkt_solver.hpp"
#include "kronos_optimizer.hpp"
#include "kronos_config.h"

using namespace std;
using namespace kronos;

int main() {
    cout << "========================================\n";
    cout << "  KRONOS Trajectory Optimizer v2.0 (IPM)\n";
    cout << "  Architecture: NLP -> Newton -> Schur\n";
    cout << "========================================\n";

    // ==========================================
    // ⏱️ 阶段一：初始化与内存分配计时
    // ==========================================
    auto t_init_start = std::chrono::steady_clock::now();

    NlpWrapper nlp;
    SchurKktSolver linear_solver(1e-5, 1e-8);
    NewtonOptimizer optimizer(nlp, linear_solver);

    VectorXd w = Eigen::Map<const VectorXd>(w_init, nlp.get_nw());
    VectorXd lam = Eigen::Map<const VectorXd>(lam_init, nlp.get_ng());

    auto t_init_end = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> t_init = t_init_end - t_init_start;

    // ==========================================
    // ⏱️ 阶段二：核心数学优化过程计时
    // ==========================================
    cout << "[KRONOS] Starting optimization loop...\n";
    auto t_solve_start = std::chrono::steady_clock::now();

    bool success = optimizer.optimize(w, lam);

    auto t_solve_end = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> t_solve = t_solve_end - t_solve_start;

    // ==========================================
    // 📊 打印结果与性能报告
    // ==========================================
    cout << "\n========================================\n";
    if (success) {
        cout << "🎯 Optimization Success!\n";
        
        int nw = nlp.get_nw();
        // 【修改点】：根据变量维度决定打印方式
        if (nw <= 10) {
            // 如果是简单算例（比如我们的 2D 问题），直接打印所有变量
            cout << "   Optimal w = [" << w.transpose() << "]\n";
        } else {
            // 如果是大型轨迹优化问题，打印首尾几个关键变量
            double T_opt = w(nw - 1);
            cout << "   Optimal Time T = " << fixed << setprecision(4) << T_opt << " s\n";
            cout << "   First 5 states = [" << w.head(5).transpose() << " ...]\n";
        }
    } else {
        cout << "⚠️ Optimization Failed to converge.\n";
    }

    // 打印耗时报告
    cout << "\n--- ⏱️ Performance Profiling ---\n";
    cout << "   Initialization : " << fixed << setprecision(3) << t_init.count() << " ms\n";
    cout << "   Optimization   : " << fixed << setprecision(3) << t_solve.count() << " ms\n";
    cout << "   Total Time     : " << fixed << setprecision(3) << (t_init + t_solve).count() << " ms\n";
    cout << "========================================\n";

    return 0;
}