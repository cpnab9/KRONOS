#include "utils/casadi_wrapper.hpp"
#include <iostream>

int main() {
    std::cout << "========== CasADi + Eigen Wrapper 测试 ==========\n";

    // 1. 实例化包装器
    CasadiWrapper nlp;

    // 2. 准备伪造的输入数据
    int N = 10;
    int n_x = 3;
    int n_u = 1;
    int dim_Z = N * n_x + N * n_u; // 40
    int dim_g = N * n_x;           // 30

    Eigen::VectorXd Z_guess = Eigen::VectorXd::Ones(dim_Z) * 0.1; 
    double T_guess = 2.0;                                         
    Eigen::VectorXd lam_guess = Eigen::VectorXd::Zero(dim_g);     

    // ================= 【新增的 5 个参数】 =================
    double mu = 0.1; // 障碍参数
    Eigen::VectorXd w_start = Eigen::VectorXd::Ones(n_x);
    Eigen::VectorXd x_start_ref = Eigen::VectorXd::Zero(n_x);
    Eigen::VectorXd w_end = Eigen::VectorXd::Ones(n_x);
    Eigen::VectorXd x_end_ref = Eigen::VectorXd::Zero(n_x);
    // =======================================================

    // 3. 调用段内 (Local) 评估函数 (传入全部 8 个参数)
    std::cout << "[1] 正在评估 Segment Local 动力学与导数...\n";
    LocalEvalResult local_res = nlp.evaluate_local(
        Z_guess, T_guess, lam_guess, 
        mu, w_start, x_start_ref, w_end, x_end_ref
    );

    // 4. 打印结果以验证映射是否成功
    std::cout << " -> 约束违反度 g_defects 维度: " << local_res.g_defects.rows() << " x " << local_res.g_defects.cols() << "\n";
    std::cout << " -> 雅可比矩阵 jac_g_Z 维度: " << local_res.jac_g_Z.rows() << " x " << local_res.jac_g_Z.cols() 
              << " (非零元素: " << local_res.jac_g_Z.nonZeros() << ")\n";
    
    Eigen::MatrixXd dense_jac_Z = local_res.jac_g_Z;
    std::cout << " -> jac_g_Z 左上角 3x3 块:\n" << dense_jac_Z.block(0, 0, 3, 3) << "\n\n";

    // 5. 调用段间 (Link) 评估函数
    std::cout << "[2] 正在评估 Segment Link 连续性约束...\n";
    Eigen::VectorXd x_end_i = Eigen::VectorXd::Ones(3) * 5.0;
    Eigen::VectorXd x_start_ip1 = Eigen::VectorXd::Ones(3) * 5.1; 
    
    LinkEvalResult link_res = nlp.evaluate_link(x_end_i, x_start_ip1);
    
    std::cout << " -> 连接点误差 g_link:\n" << link_res.g_link.transpose() << "\n";
    std::cout << " -> 前一段末端雅可比 jac_end:\n" << Eigen::MatrixXd(link_res.jac_end) << "\n";
    std::cout << " -> 后一段开端雅可比 jac_start:\n" << Eigen::MatrixXd(link_res.jac_start) << "\n";

    std::cout << "========== 测试成功完成！ ==========\n";
    return 0;
}