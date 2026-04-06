#include "planner/trajectory_planner.hpp"

int main() {
    // 读取 Python 自动生成的配置文件
    aeroplan::TrajectoryPlanner planner("../config/ocp_config.json");
    planner.plan();
    return 0;
}