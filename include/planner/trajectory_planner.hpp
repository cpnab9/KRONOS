#pragma once
#include "ocp/flight_ocp.hpp"
#include <fatrop/fatrop.hpp>
#include <memory>
#include <vector>
#include <string>

namespace aeroplan {

class TrajectoryPlanner {
public:
    // 接受 JSON 配置文件路径作为启动参数
    TrajectoryPlanner(const std::string& config_file);
    
    bool plan();
    bool plan(const std::vector<double>& current_state);

private:
    // 初始化并构建求解器
    void initialize(const std::string& config_file);

    // 从 JSON 文件反序列化配置的内部函数
    OCPConfig load_config(const std::string& filepath);

    OCPConfig current_config_;
    std::shared_ptr<FlightOCP> ocp_problem_;
    std::shared_ptr<fatrop::IpAlgBuilder<fatrop::OcpType>> builder_;
    std::shared_ptr<fatrop::IpAlgorithm<fatrop::OcpType>> solver_;
};

} // namespace aeroplan