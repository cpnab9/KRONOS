#pragma once
#include <memory>
#include <fatrop/fatrop.hpp>
#include "ocp/flight_ocp.hpp"

namespace aeroplan {

class TrajectoryPlanner {
public:
    TrajectoryPlanner();
    
    // 执行一次在线规划
    bool plan();

private:
    std::shared_ptr<fatrop::NlpOcp> ocp_problem_;
    std::shared_ptr<fatrop::IpAlgorithm<fatrop::OcpType>> solver_;
    std::shared_ptr<fatrop::IpAlgBuilder<fatrop::OcpType>> builder_;
};

} // namespace aeroplan