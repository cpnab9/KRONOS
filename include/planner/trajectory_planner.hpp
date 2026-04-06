#pragma once
#include <memory>
#include <fatrop/fatrop.hpp>
#include "ocp/flight_ocp.hpp"

namespace aeroplan {

class TrajectoryPlanner {
public:
    TrajectoryPlanner();
    
    // 显式传入配置初始化（解耦物理问题与求解器）
    void initialize(const OCPConfig& config);
    
    // 带状态反馈的在线闭环规划
    bool plan(const std::vector<double>& current_state);

    // 空载测试规划
    bool plan();

private:
    OCPConfig current_config_;
    std::shared_ptr<FlightOCP> ocp_problem_;
    std::shared_ptr<fatrop::IpAlgorithm<fatrop::OcpType>> solver_;
    std::shared_ptr<fatrop::IpAlgBuilder<fatrop::OcpType>> builder_;
    
    OCPConfig create_default_config();
};

} // namespace aeroplan