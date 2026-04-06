#include "planner/trajectory_planner.hpp"
#include <iostream>
#include <vector>
#include <cmath>

namespace aeroplan {

TrajectoryPlanner::TrajectoryPlanner() {
    ocp_problem_ = std::make_shared<fatrop::NlpOcp>(std::make_shared<FlightOCP>());
    
    fatrop::OptionRegistry options;
    
    builder_ = std::make_shared<fatrop::IpAlgBuilder<fatrop::OcpType>>(ocp_problem_);
    solver_ = builder_->with_options_registry(&options).build();
}

bool TrajectoryPlanner::plan() {
    fatrop::Timer timer;
    timer.start();
    
    // 执行求解
    fatrop::IpSolverReturnFlag ret = solver_->optimize();
    
    std::cout << "\n======================================" << std::endl;
    std::cout << "  Fatrop 求解时间: " << timer.stop() << " ms" << std::endl;
    std::cout << "  求解状态: " << (ret == fatrop::IpSolverReturnFlag::Success ? "成功" : "失败") << std::endl;
    
    if (ret == fatrop::IpSolverReturnFlag::Success) {
        auto sol = builder_->get_ipdata();
        std::cout << "  (V1.0 编译通过！底层内存已预分配并成功收敛)" << std::endl;
        // 打印底层的迭代与耗时统计
        std::cout << sol->timing_statistics() << std::endl;
    }
    std::cout << "======================================\n" << std::endl;
    
    return ret == fatrop::IpSolverReturnFlag::Success;
}

} // namespace aeroplan