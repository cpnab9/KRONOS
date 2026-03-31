#pragma once
#include <Eigen/Dense>

namespace kronos {
    // 统一使用 Eigen 的动态大小矩阵和向量
    using VectorXd = Eigen::VectorXd;
    using MatrixXd = Eigen::MatrixXd;
}