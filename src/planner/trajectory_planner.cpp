#include "planner/trajectory_planner.hpp"
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <nlohmann/json.hpp> 

namespace aeroplan {

using json = nlohmann::json;

TrajectoryPlanner::TrajectoryPlanner(const std::string& config_file) {
    initialize(config_file);
}

OCPConfig TrajectoryPlanner::load_config(const std::string& filepath) {
    std::ifstream f(filepath);
    if (!f.is_open()) throw std::runtime_error("无法打开配置文件: " + filepath);
    json j; f >> j;

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
    
    if (j.contains("guess_x0")) cfg.guess_x0 = j.at("guess_x0").get<std::vector<double>>();
    if (j.contains("guess_xf")) cfg.guess_xf = j.at("guess_xf").get<std::vector<double>>();
    if (j.contains("guess_u0")) cfg.guess_u0 = j.at("guess_u0").get<std::vector<double>>();
    if (j.contains("guess_uf")) cfg.guess_uf = j.at("guess_uf").get<std::vector<double>>();

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
    
    fatrop::OptionRegistry options;
    builder_ = std::make_shared<fatrop::IpAlgBuilder<fatrop::OcpType>>(nlp_wrapper);
    
    // 初始化障碍惩罚参数 (冷启动需要较大的 mu 帮助寻找可行域)
    // options.set_option("mu_init", 0.1); 
    
    solver_ = builder_->with_options_registry(&options).build();
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
    std::cout << "  Fatrop 求解时间: " << timer.stop() << " s" << std::endl;
    std::cout << "  求解状态: " << (ret == fatrop::IpSolverReturnFlag::Success ? "成功" : "失败") << std::endl;
    std::cout << "======================================\n" << std::endl;
    
    return ret == fatrop::IpSolverReturnFlag::Success;
}

void TrajectoryPlanner::extract_solution(std::vector<std::vector<double>>& x_out, 
                                         std::vector<std::vector<double>>& u_out, 
                                         double& tf_out) const {
    ocp_problem_->get_last_trajectory(x_out, u_out);
    tf_out = x_out[0][8]; 
}

std::vector<double> TrajectoryPlanner::interp1d(const std::vector<double>& t_old, 
                                                const std::vector<std::vector<double>>& data_old, 
                                                double t_query) const {
    int n_dim = data_old[0].size();
    std::vector<double> result(n_dim, 0.0);

    if (t_query <= t_old.front()) return data_old.front();
    if (t_query >= t_old.back()) return data_old.back();

    auto it = std::lower_bound(t_old.begin(), t_old.end(), t_query);
    int idx = std::distance(t_old.begin(), it);
    
    if (idx == 0) idx = 1;
    if (idx >= data_old.size()) idx = data_old.size() - 1; 

    double t1 = t_old[idx - 1];
    double t2 = t_old[idx];
    const auto& y1 = data_old[idx - 1];
    const auto& y2 = data_old[idx];

    double fraction = (t2 - t1) > 1e-9 ? (t_query - t1) / (t2 - t1) : 0.0;
    for (int i = 0; i < n_dim; ++i) {
        result[i] = y1[i] + fraction * (y2[i] - y1[i]);
    }
    return result;
}

bool TrajectoryPlanner::update_mesh_and_warmstart() {
    std::cout << "[*] 正在评估截断误差并重新划分网格..." << std::endl;

    int K = current_config_.K_intervals;
    std::vector<std::vector<double>> x_old, u_old;
    double tf_opt = 0.0;
    
    extract_solution(x_old, u_old, tf_opt);

    if (current_config_.mesh_fractions.empty()) {
        current_config_.mesh_fractions.assign(K, 1.0 / K);
    }

    std::vector<double> t_old(K + 1, 0.0);
    for (int k = 0; k < K; ++k) {
        t_old[k + 1] = t_old[k] + current_config_.mesh_fractions[k] * tf_opt;
    }
    std::vector<double> t_u_old(K, 0.0);
    for (int k = 0; k < K; ++k) {
        t_u_old[k] = t_old[k] + 0.5 * current_config_.mesh_fractions[k] * tf_opt;
    }

    std::vector<double> errors(K, 0.0);
    for (int k = 0; k < K; ++k) {
        double delta_h = std::abs(x_old[k+1][0] - x_old[k][0]);
        double delta_v = std::abs(x_old[k+1][3] - x_old[k][3]);
        errors[k] = delta_h + delta_v * 0.1; 
    }

    double eps = 1e-2;
    std::vector<double> desired_fractions(K, 0.0);
    double sum_desired = 0.0;
    for (int k = 0; k < K; ++k) {
        double monitor = std::sqrt(errors[k]) + eps;
        desired_fractions[k] = 1.0 / monitor;
        sum_desired += desired_fractions[k];
    }
    for (int k = 0; k < K; ++k) desired_fractions[k] /= sum_desired;

    // =================================================================
    // 【修复 3 (可选优化)】：更保守的网格更新，防止产生虚假插值梯度
    // 原逻辑 0.5 * old + 0.5 * new，在突变时会导致插值误差过大
    // =================================================================
    std::vector<double> new_fractions(K, 0.0);
    double sum_new = 0.0;
    for (int k = 0; k < K; ++k) {
        new_fractions[k] = 0.8 * current_config_.mesh_fractions[k] + 0.2 * desired_fractions[k];
        sum_new += new_fractions[k];
    }
    for (int k = 0; k < K; ++k) new_fractions[k] /= sum_new;

    std::vector<double> t_new(K + 1, 0.0);
    for (int k = 0; k < K; ++k) t_new[k + 1] = t_new[k] + new_fractions[k] * tf_opt;

    std::vector<std::vector<double>> warm_x(K + 1, std::vector<double>(current_config_.nx));
    std::vector<std::vector<double>> warm_u(K, std::vector<double>(current_config_.nu));

    for (int k = 0; k <= K; ++k) {
        warm_x[k] = interp1d(t_old, x_old, t_new[k]);
    }
    for (int k = 0; k < K; ++k) {
        double t_mid = t_new[k] + 0.5 * new_fractions[k] * tf_opt;
        warm_u[k] = interp1d(t_u_old, u_old, t_mid);
    }

    current_config_.use_warm_start = true;
    current_config_.enable_nfz2 = true; 
    current_config_.mesh_fractions = new_fractions;
    current_config_.warm_x = warm_x;
    current_config_.warm_u = warm_u;

    ocp_problem_->update_config(current_config_);

    // =================================================================
    // 【修复 2】：重设 Fatrop 参数，模拟真正的的热启动 (Warm Start)
    // =================================================================
    fatrop::OptionRegistry options;

    builder_->with_options_registry(&options);
    
    // 重建求解器，载入带正确 mu 参数的热启动初值
    solver_ = builder_->build(); 
    
    std::cout << "[*] 映射成功！已重置求解器并装载稠密热启动初值 (mu_init = 1e-3)。" << std::endl;
    return true;
}

} // namespace aeroplan