#include "planner/trajectory_planner.hpp"
#include <iostream>
#include <vector>

namespace aeroplan {

TrajectoryPlanner::TrajectoryPlanner() {
    // 实例化时先注入默认的最速降线测试参数
    initialize(create_default_config());
}

OCPConfig TrajectoryPlanner::create_default_config() {
    OCPConfig cfg;
    cfg.K_intervals = 20;
    cfg.nx = 4;
    cfg.nu = 15;
    cfg.ng_defects = 12;

    // 默认初始状态约束 [x, y, v]
    cfg.init_idx = {0, 1, 2};
    cfg.init_val = {0.0, 0.0, 0.0};

    // 默认终端状态约束 [x, y]
    cfg.term_idx = {0, 1};
    cfg.term_val = {1.0, 1.0};

    // 默认不等式边界 (约束第3号状态 tf >= 0.1)
    cfg.ineq_state_idx = {3};
    cfg.ineq_lower = {0.1};
    cfg.ineq_upper = {10.0};

    // 目标函数: min tf
    cfg.obj_state_idx = 3;
    cfg.obj_weight = 1.0;

    // 默认猜想
    cfg.guess_xk = {0.0, 0.0, 2.0, 1.0};
    cfg.guess_uk = std::vector<double>(cfg.nu, 0.5);

    return cfg;
}

void TrajectoryPlanner::initialize(const OCPConfig& config) {
    current_config_ = config;
    
    // 1. 创建你自己的 OCP 物理问题实例 (保存下来以便后续调用 update_initial_state)
    ocp_problem_ = std::make_shared<FlightOCP>(current_config_);
    
    // 2. 关键修复：用 fatrop::NlpOcp 将底层的 OcpAbstract 包装成 Nlp 接口
    auto nlp_wrapper = std::make_shared<fatrop::NlpOcp>(ocp_problem_);
    
    fatrop::OptionRegistry options;
    // options.add_option("tol", 1e-4);
    // options.add_option("mu_init", 0.1);
    // options.add_option("print_level", 5);
    
    // 3. 将包装后的 nlp_wrapper 传给 Builder
    builder_ = std::make_shared<fatrop::IpAlgBuilder<fatrop::OcpType>>(nlp_wrapper);
    solver_ = builder_->with_options_registry(&options).build();
}

bool TrajectoryPlanner::plan(const std::vector<double>& current_state) {
    // 更新实时反馈状态，作为本次轨迹优化的起始条件
    ocp_problem_->update_initial_state(current_state);
    return plan();
}

bool TrajectoryPlanner::plan() {
    fatrop::Timer timer;
    timer.start();
    
    fatrop::IpSolverReturnFlag ret = solver_->optimize();
    
    std::cout << "\n======================================" << std::endl;
    std::cout << "  Fatrop 求解时间: " << timer.stop() << " ms" << std::endl;
    std::cout << "  求解状态: " << (ret == fatrop::IpSolverReturnFlag::Success ? "成功" : "失败") << std::endl;
    
    if (ret == fatrop::IpSolverReturnFlag::Success) {
        auto sol = builder_->get_ipdata();
        std::cout << "  (V2.0 编译通过！OCP框架已彻底实现通用化与解耦)" << std::endl;
    }
    std::cout << "======================================\n" << std::endl;
    
    return ret == fatrop::IpSolverReturnFlag::Success;
}

} // namespace aeroplan