#ifndef KRONOS_TYPES_HPP
#define KRONOS_TYPES_HPP

#include <Eigen/Dense>
#include <Eigen/Sparse>

namespace kronos {
    // 基础密集矩阵/向量类型
    using VectorXd = Eigen::VectorXd;
    using MatrixXd = Eigen::MatrixXd;
    
    // 增加稀疏矩阵类型定义
    using SparseMatrixXd = Eigen::SparseMatrix<double>;
}

#endif // KRONOS_TYPES_HPP