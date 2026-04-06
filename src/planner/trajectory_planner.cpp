#include "planner/trajectory_planner.hpp"
#include <iostream>
#include <vector>

namespace aeroplan {

TrajectoryPlanner::TrajectoryPlanner() {
    initialize(create_default_config());
}

OCPConfig TrajectoryPlanner::create_default_config() {
    OCPConfig cfg;
    cfg.K_intervals = 100;
    cfg.nx = 5;       // 状态: [x, y, vx, vy, cost]
    cfg.nu = 21;      // 控制: d*nx + d*nu_real = 15 + 6 = 21
    cfg.ng_defects = 15;
    cfg.ng_ineq = 6;  // 三个内部配点上的 [fx, fy] 边界

    // 初始状态约束 (x=0, y=0, vx=0, vy=0, cost=0)
    cfg.init_idx = {0, 1, 2, 3, 4};
    cfg.init_val = {0.0, 0.0, 0.0, 0.0, 0.0};

    // 【核心修复 1】终端状态约束：摒弃误导性注释，对齐原算例的真实代码要求！
    cfg.term_idx = {0, 1, 2, 3};
    cfg.term_val = {1.0, 2.0, 3.0, 4.0};

    // 对 d=3 的三个配点的控制分量 fx, fy 分别施加边界
    cfg.ineq_lower = {-50.0, -100.0, -50.0, -100.0, -50.0, -100.0};
    cfg.ineq_upper = { 50.0,  100.0,  50.0,  100.0,  50.0,  100.0};

    // 目标函数：最小化终端的 cost 状态 (索引为4)
    cfg.obj_state_idx = 4; 
    
    // 【核心修复 2】目标函数量纲对齐：补偿积分引入的 dt (0.05)
    cfg.obj_weight = 20.0; // 1.0 / 0.05

    cfg.guess_xk = {0.0, 0.0, 0.0, 0.0, 0.0};
    cfg.guess_uk = std::vector<double>(cfg.nu, 0.0);

    return cfg;
}

void TrajectoryPlanner::initialize(const OCPConfig& config) {
    current_config_ = config;
    
    ocp_problem_ = std::make_shared<FlightOCP>(current_config_);
    auto nlp_wrapper = std::make_shared<fatrop::NlpOcp>(ocp_problem_);
    
    fatrop::OptionRegistry options;
    
    builder_ = std::make_shared<fatrop::IpAlgBuilder<fatrop::OcpType>>(nlp_wrapper);
    solver_ = builder_->with_options_registry(&options).build();
}

bool TrajectoryPlanner::plan(const std::vector<double>& current_state) {
    ocp_problem_->update_initial_state(current_state);
    return plan();
}

bool TrajectoryPlanner::plan() {
    fatrop::Timer timer;
    timer.start();
    
    fatrop::IpSolverReturnFlag ret = solver_->optimize();
    
    std::cout << "\n======================================" << std::endl;
    std::cout << "  Fatrop (KRONOS 伪谱法) 求解时间: " << timer.stop() << " ms" << std::endl;
    std::cout << "  求解状态: " << (ret == fatrop::IpSolverReturnFlag::Success ? "成功" : "失败") << std::endl;
    std::cout << "======================================\n" << std::endl;
    
    return ret == fatrop::IpSolverReturnFlag::Success;
}

} // namespace aeroplan