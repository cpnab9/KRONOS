#include <iostream>
#include <vector>
#include "planner/trajectory_planner.hpp"

using namespace aeroplan;

int main() {
    try {
        std::cout << ">>> 启动 KRONOS 飞行器轨迹优化系统 <<<" << std::endl;

        // 1. 初始化规划器 (加载初始 JSON 配置)
        TrajectoryPlanner planner("../config/ocp_config.json");

        // ---------------------------------------------------------
        // 阶段 1：基准离线规划 (冷启动)
        // 此时环境中可能没有突发障碍物 (NFZ2 关闭)
        // ---------------------------------------------------------
        std::cout << "\n======================================" << std::endl;
        std::cout << "=== 阶段 1：基准离线规划 (冷启动) ===" << std::endl;
        std::cout << "======================================" << std::endl;
        
        bool success_phase1 = planner.plan();
        if (!success_phase1) {
            std::cerr << "[错误] 阶段 1 基准规划失败！" << std::endl;
            return -1;
        }

        // ---------------------------------------------------------
        // 模拟环境变化：飞行过程中探测到前方突发禁飞区
        // ---------------------------------------------------------
        std::cout << "\n[!] 传感器警告：检测到前方出现突发威胁(NFZ2)！" << std::endl;
        std::cout << "[!] 正在触发重规划机制..." << std::endl;

        // 此处你可以添加代码修改底层配置，例如开启 NFZ2 标志位
        // planner.enable_nfz2(); // 假设你提供了一个动态开启约束的接口

        // 提取上一阶段的最优解，进行网格自适应分配并执行一维稠密插值
        bool map_success = planner.update_mesh_and_warmstart();
        if (!map_success) {
            std::cerr << "[错误] 热启动数据映射失败！" << std::endl;
            return -1;
        }

        // ---------------------------------------------------------
        // 阶段 2：突发威胁规避 (热启动)
        // 此时系统继承了上一阶段的高质量插值猜想，并使用极小的 mu_init
        // ---------------------------------------------------------
        std::cout << "\n======================================" << std::endl;
        std::cout << "=== 阶段 2：突发威胁规避 (热启动) ===" << std::endl;
        std::cout << "======================================" << std::endl;
        
        // 传入当前飞行器的实时状态 x0 (可选，用于 MPC 滚动优化)
        // std::vector<double> current_state = {...}; 
        // bool success_phase2 = planner.plan(current_state);
        
        bool success_phase2 = planner.plan();
        if (!success_phase2) {
            std::cerr << "[错误] 阶段 2 热启动重规划失败！" << std::endl;
            return -1;
        }

        std::cout << "\n>>> 所有轨迹规划任务圆满完成，热启动耗时应显著降低 <<<" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "发生严重异常: " << e.what() << std::endl;
    }

    return 0;
}