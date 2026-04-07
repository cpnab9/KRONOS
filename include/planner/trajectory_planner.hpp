#pragma once
#include <memory>
#include <vector>
#include <string>
#include "ocp/flight_ocp.hpp"
#include <fatrop/fatrop.hpp>

namespace aeroplan {

class TrajectoryPlanner {
public:
    TrajectoryPlanner(const std::string& config_file);
    void initialize(const std::string& config_file);
    
    bool plan();
    bool plan(const std::vector<double>& current_state);

    bool update_mesh_and_warmstart();

private:
    OCPConfig load_config(const std::string& filepath);

    std::vector<double> interp1d(const std::vector<double>& t_old, 
                                 const std::vector<std::vector<double>>& data_old, 
                                 double t_query) const;

    void extract_solution(std::vector<std::vector<double>>& x_out, 
                          std::vector<std::vector<double>>& u_out, 
                          double& tf_out) const;

    OCPConfig current_config_;
    std::shared_ptr<FlightOCP> ocp_problem_;
    std::shared_ptr<fatrop::IpAlgBuilder<fatrop::OcpType>> builder_;
    std::shared_ptr<fatrop::IpAlgorithm<fatrop::OcpType>> solver_;
};

} // namespace aeroplan