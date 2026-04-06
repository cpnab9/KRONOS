#include "planner/trajectory_planner.hpp"
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <nlohmann/json.hpp> // 引入 JSON 解析库

namespace aeroplan {

using json = nlohmann::json;

TrajectoryPlanner::TrajectoryPlanner(const std::string& config_file) {
    initialize(config_file);
}

OCPConfig TrajectoryPlanner::load_config(const std::string& filepath) {
    std::ifstream f(filepath);
    if (!f.is_open()) {
        throw std::runtime_error("无法打开配置文件: " + filepath);
    }
    json j;
    f >> j;

    OCPConfig cfg;
    cfg.problem_name = j.value("problem_name", "Unknown Problem");
    cfg.K_intervals = j.at("K_intervals").get<int>();
    cfg.nx = j.at("nx").get<int>();
    cfg.nu = j.at("nu").get<int>();
    cfg.ng_defects = j.at("ng_defects").get<int>();
    cfg.ng_ineq = j.at("ng_ineq").get<int>();
    
    cfg.init_idx = j.at("init_idx").get<std::vector<int>>();
    cfg.init_val = j.at("init_val").get<std::vector<double>>();
    cfg.term_idx = j.at("term_idx").get<std::vector<int>>();
    cfg.term_val = j.at("term_val").get<std::vector<double>>();
    
    cfg.ineq_lower = j.at("ineq_lower").get<std::vector<double>>();
    cfg.ineq_upper = j.at("ineq_upper").get<std::vector<double>>();
    
    cfg.obj_state_idx = j.at("obj_state_idx").get<int>();
    cfg.obj_weight = j.at("obj_weight").get<double>();
    
    cfg.guess_xk = j.at("guess_xk").get<std::vector<double>>();
    cfg.guess_uk = j.at("guess_uk").get<std::vector<double>>();

    return cfg;
}

void TrajectoryPlanner::initialize(const std::string& config_file) {
    std::cout << "正在加载配置文件: " << config_file << std::endl;
    current_config_ = load_config(config_file);
    
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
    std::cout << "\n>>> KRONOS 通用引擎就绪 | 运行问题: [" << current_config_.problem_name << "] <<<" << std::endl;
    fatrop::Timer timer;
    timer.start();
    
    fatrop::IpSolverReturnFlag ret = solver_->optimize();
    
    std::cout << "\n======================================" << std::endl;
    std::cout << "  Fatrop 求解时间: " << timer.stop() << " ms" << std::endl;
    std::cout << "  求解状态: " << (ret == fatrop::IpSolverReturnFlag::Success ? "成功" : "失败") << std::endl;
    std::cout << "======================================\n" << std::endl;
    
    return ret == fatrop::IpSolverReturnFlag::Success;
}

} // namespace aeroplan