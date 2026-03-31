#include <iostream>
#include <iomanip>
#include "kronos_types.hpp"
#include "kronos_nlp_wrapper.hpp"
#include "kronos_solver.hpp"
#include "kronos_config.h"

using namespace std;
using namespace kronos;

int main() {
    cout << "========================================\n";
    cout << "  KRONOS Trajectory Optimizer v1.0\n";
    cout << "========================================\n";

    // 1. 实例化 NLP 包装器
    NlpWrapper nlp;

    // 2. 实例化求解器
    NewtonSchurSolver solver(nlp);

    // 3. 初始化变量 (从 python 生成的 kronos_config.h 中读取)
    VectorXd w = Eigen::Map<const VectorXd>(w_init, nlp.get_nw());
    VectorXd lam = Eigen::Map<const VectorXd>(lam_init, nlp.get_ng());

    // 4. 执行求解
    if (solver.solve(w, lam)) {
        double T_opt = w(nlp.get_nw() - 1);
        cout << "🎯 Optimization Success! Optimal Time T = " 
             << fixed << setprecision(4) << T_opt << " s\n";
    } else {
        cout << "⚠️ Optimization Failed to converge.\n";
    }

    return 0;
}