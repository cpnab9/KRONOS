#include "../include/solver/ipm_core.hpp"

int main() {
    // 对应我们最速降线问题中的参数
    int K = 2;       // 2 个分段
    int N = 10;      // 每段 10 个配点
    int n_x = 3;     // 状态维度 [x, y, v]
    int n_u = 1;     // 控制维度 [theta]

    // 实例化内点法核心求解器
    IPMCore planner(K, N, n_x, n_u);

    // 启动求解
    planner.solve();

    return 0;
}