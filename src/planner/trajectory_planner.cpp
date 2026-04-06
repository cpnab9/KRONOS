#include "planner/trajectory_planner.hpp"
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <nlohmann/json.hpp> 

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
    
    // === 核心修改：读取端点初始猜想 ===
    if (j.contains("guess_x0")) cfg.guess_x0 = j.at("guess_x0").get<std::vector<double>>();
    if (j.contains("guess_xf")) cfg.guess_xf = j.at("guess_xf").get<std::vector<double>>();
    if (j.contains("guess_u0")) cfg.guess_u0 = j.at("guess_u0").get<std::vector<double>>();
    if (j.contains("guess_uf")) cfg.guess_uf = j.at("guess_uf").get<std::vector<double>>();

    // === 松弛变量配置读取 ===
    cfg.ns = j.contains("ns") ? j.at("ns").get<int>() : 0;
    if (j.contains("idx_s")) cfg.idx_s = j.at("idx_s").get<std::vector<int>>();
    if (j.contains("guess_sk")) cfg.guess_sk = j.at("guess_sk").get<std::vector<double>>();
    if (j.contains("Zl")) cfg.Zl = j.at("Zl").get<std::vector<double>>();
    if (j.contains("zl")) cfg.zl = j.at("zl").get<std::vector<double>>();

    return cfg;
}

void TrajectoryPlanner::initialize(const std::string& config_file) {
    std::cout << "正在加载配置文件: " << config_file << std::endl;
    current_config_ = load_config(config_file);
    
    ocp_problem_ = std::make_shared<FlightOCP>(current_config_);
    auto nlp_wrapper = std::make_shared<fatrop::NlpOcp>(ocp_problem_);
    
    // 1. 创建空的选项注册表
    fatrop::OptionRegistry options;
    
    // 2. 先实例化 builder，并把 options 的指针塞进去
    builder_ = std::make_shared<fatrop::IpAlgBuilder<fatrop::OcpType>>(nlp_wrapper);
    
    // 3. 构建求解器。在 build() 的过程中，Fatrop 会自动把所有默认参数注册进 options 字典里！
    solver_ = builder_->with_options_registry(&options).build();
    
    // 4. 此时 options 已经满了！我们打印全家桶出来看看：
    std::cout << "\n=== Fatrop 可用参数列表 ===" << std::endl;
    std::cout << options << std::endl;
    std::cout << "===========================\n" << std::endl;
    
    // 5. 现在可以安全地修改参数了，因为它们已经在字典里了
    // 注意：这里使用的是 set_option (而不是 addOption)
    options.set_option("mu_init", 0.1); 
    // options.set_option("print_level", 5); 
    // options.set_option("tol", 1e-6); 
}

bool TrajectoryPlanner::plan(const std::vector<double>& current_state) {
    ocp_problem_->update_initial_state(current_state);
    return plan();
}

bool TrajectoryPlanner::plan() {
    std::cout << "\n>>> KRONOS 通用引擎就绪 | 运行问题: [" << current_config_.problem_name << "] <<<" << std::endl;
    if (current_config_.ns > 0) {
        std::cout << ">>> 提示: 已启用松弛变量 (ns = " << current_config_.ns << "), 自动执行软约束及线性插值初始化 <<<" << std::endl;
    }

    fatrop::Timer timer;
    timer.start();
    
    fatrop::IpSolverReturnFlag ret = solver_->optimize();
    
    std::cout << "\n======================================" << std::endl;
    std::cout << "  Fatrop 求解时间: " << timer.stop() << " s" << std::endl; // 修正单位为秒
    std::cout << "  求解状态: " << (ret == fatrop::IpSolverReturnFlag::Success ? "成功" : "失败") << std::endl;
    std::cout << "======================================\n" << std::endl;
    
    return ret == fatrop::IpSolverReturnFlag::Success;
}

} // namespace aeroplan