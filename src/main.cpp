#include "planner/trajectory_planner.hpp"
#include <iostream>

int main() {
    std::cout << "启动飞行器伪谱法轨迹规划器 (V1.0)..." << std::endl;
    
    // 初始化规划器 (内部完成问题内存的预分配和初始化)
    aeroplan::TrajectoryPlanner planner;
    
    // 执行一次在线规划
    planner.plan();
    
    return 0;
}