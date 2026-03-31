#include <iostream>
#include <iomanip>
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

    // 1. 初始化模型
    NlpWrapper nlp;

    // 2. 初始化底层线性求解器
    SchurKktSolver linear_solver(1e-5, 1e-8);

    // 3. 初始化优化算法（将模型和求解器注入进去）
    NewtonOptimizer optimizer(nlp, linear_solver);

    // 4. 准备初值
    VectorXd w = Eigen::Map<const VectorXd>(w_init, nlp.get_nw());
    VectorXd lam = Eigen::Map<const VectorXd>(lam_init, nlp.get_ng());

    // 5. 求解
    if (optimizer.optimize(w, lam)) {
        double T_opt = w(nlp.get_nw() - 1);
        cout << "🎯 Optimization Success! Optimal Time T = " 
             << fixed << setprecision(4) << T_opt << " s\n";
    } else {
        cout << "⚠️ Optimization Failed to converge.\n";
    }

    return 0;
}