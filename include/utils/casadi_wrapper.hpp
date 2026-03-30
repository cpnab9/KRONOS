#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>
#include <iostream>

// CasADi 默认使用 long long int 作为其整型类型 (64位系统)
#ifndef CASADI_INT_TYPE
#define CASADI_INT_TYPE long long int
#endif
typedef CASADI_INT_TYPE casadi_int;

// ==============================================================================
// 1. 声明 CasADi 自动生成的 C 函数 (extern "C")
// ==============================================================================
extern "C" {
    // --- Segment Local 函数 ---
    int eval_segment_local_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w);
    int eval_segment_local(const double** arg, double** res, casadi_int* iw, double* w, int mem);
    const casadi_int* eval_segment_local_sparsity_out(casadi_int i);

    // --- Segment Link 函数 ---
    int eval_segment_link_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w);
    int eval_segment_link(const double** arg, double** res, casadi_int* iw, double* w, int mem);
    const casadi_int* eval_segment_link_sparsity_out(casadi_int i);
}

// ==============================================================================
// 2. 定义返回结果的结构体 (方便 C++ 统一接收)
// ==============================================================================
struct LocalEvalResult {
    Eigen::VectorXd g_defects;
    Eigen::SparseMatrix<double> jac_g_Z;
    Eigen::SparseMatrix<double> jac_g_T;
    Eigen::SparseMatrix<double> hess_L_Z;
    Eigen::VectorXd grad_L_Z;
};

struct LinkEvalResult {
    Eigen::VectorXd g_link;
    Eigen::SparseMatrix<double> jac_end;
    Eigen::SparseMatrix<double> jac_start;
};

// ==============================================================================
// 3. CasADi 包装器类 (线程安全：每个实例拥有独立的 Workspace)
// ==============================================================================
class CasadiWrapper {
private:
    // 内存工作区
    std::vector<const double*> arg_;
    std::vector<double*> res_;
    std::vector<casadi_int> iw_;
    std::vector<double> w_;

    // 辅助函数：将 CasADi 的 CCS 稀疏指针映射为 Eigen::SparseMatrix
    Eigen::SparseMatrix<double> map_to_eigen_sparse(const casadi_int* sparsity, double* nonzeros) {
        casadi_int nrow = sparsity[0];
        casadi_int ncol = sparsity[1];
        const casadi_int* colind = sparsity + 2;
        const casadi_int* rowind = sparsity + 2 + ncol + 1;
        casadi_int nnz = colind[ncol];

        // 强转索引类型以适配 Eigen (Eigen 默认通常用 int 存储稀疏矩阵的 inner/outer index)
        // 注意：如果矩阵极大(>2GB)，需要给 Eigen 定义 EIGEN_DEFAULT_DENSE_INDEX_TYPE
        std::vector<int> colind_int(colind, colind + ncol + 1);
        std::vector<int> rowind_int(rowind, rowind + nnz);

        Eigen::Map<const Eigen::SparseMatrix<double>> mapped_mat(
            nrow, ncol, nnz, colind_int.data(), rowind_int.data(), nonzeros
        );
        
        // 隐式拷贝返回（因为外部还需要使用，这里脱离了裸指针作用域，所以触发一次安全拷贝。
        // 对于分段伪谱法的段内小矩阵，这种拷贝开销微乎其微）
        return mapped_mat; 
    }

    // 分配 Local 工作区
    void allocate_local_workspace() {
        casadi_int sz_arg, sz_res, sz_iw, sz_w;
        eval_segment_local_work(&sz_arg, &sz_res, &sz_iw, &sz_w);
        arg_.resize(sz_arg);
        res_.resize(sz_res);
        iw_.resize(sz_iw);
        w_.resize(sz_w);
    }

    // 分配 Link 工作区
    void allocate_link_workspace() {
        casadi_int sz_arg, sz_res, sz_iw, sz_w;
        eval_segment_link_work(&sz_arg, &sz_res, &sz_iw, &sz_w);
        arg_.resize(sz_arg);
        res_.resize(sz_res);
        iw_.resize(sz_iw);
        w_.resize(sz_w);
    }

public:
    CasadiWrapper() {}

    // ==========================================================================
    // 评估段内 (Local) 动力学、雅可比与海森矩阵
    // ==========================================================================
    LocalEvalResult evaluate_local(const Eigen::VectorXd& Z, double T_global, const Eigen::VectorXd& lam,
                                   double mu, const Eigen::VectorXd& w_start, const Eigen::VectorXd& x_start,
                                   const Eigen::VectorXd& w_end, const Eigen::VectorXd& x_end) {
        allocate_local_workspace();

        // 绑定 8 个输入参数
        arg_[0] = Z.data();
        arg_[1] = &T_global;
        arg_[2] = lam.data();
        arg_[3] = &mu;
        arg_[4] = w_start.data();
        arg_[5] = x_start.data();
        arg_[6] = w_end.data();
        arg_[7] = x_end.data();

        // 获取 5 个输出内存大小
        const casadi_int* sp_g = eval_segment_local_sparsity_out(0);
        const casadi_int* sp_jac_Z = eval_segment_local_sparsity_out(1);
        const casadi_int* sp_jac_T = eval_segment_local_sparsity_out(2);
        const casadi_int* sp_hess = eval_segment_local_sparsity_out(3);
        const casadi_int* sp_grad = eval_segment_local_sparsity_out(4); // 【新增】

        std::vector<double> res_g(sp_g[0]); 
        std::vector<double> res_jac_Z(sp_jac_Z[2 + sp_jac_Z[1]]); 
        std::vector<double> res_jac_T(sp_jac_T[2 + sp_jac_T[1]]);
        std::vector<double> res_hess(sp_hess[2 + sp_hess[1]]);
        std::vector<double> res_grad(sp_grad[0]); // 【新增】

        res_[0] = res_g.data();
        res_[1] = res_jac_Z.data();
        res_[2] = res_jac_T.data();
        res_[3] = res_hess.data();
        res_[4] = res_grad.data(); // 【新增】

        eval_segment_local(arg_.data(), res_.data(), iw_.data(), w_.data(), 0);

        LocalEvalResult result;
        result.g_defects = Eigen::Map<Eigen::VectorXd>(res_g.data(), res_g.size());
        result.jac_g_Z = map_to_eigen_sparse(sp_jac_Z, res_jac_Z.data());
        result.jac_g_T = map_to_eigen_sparse(sp_jac_T, res_jac_T.data());
        result.hess_L_Z = map_to_eigen_sparse(sp_hess, res_hess.data());
        result.grad_L_Z = Eigen::Map<Eigen::VectorXd>(res_grad.data(), res_grad.size()); // 【新增】

        return result;
    }

    // ==========================================================================
    // 评估段间 (Link) 连续性约束及雅可比
    // ==========================================================================
    LinkEvalResult evaluate_link(const Eigen::VectorXd& x_end_i, const Eigen::VectorXd& x_start_ip1) {
        allocate_link_workspace();

        arg_[0] = x_end_i.data();
        arg_[1] = x_start_ip1.data();

        const casadi_int* sp_g = eval_segment_link_sparsity_out(0);
        const casadi_int* sp_jac_end = eval_segment_link_sparsity_out(1);
        const casadi_int* sp_jac_start = eval_segment_link_sparsity_out(2);

        std::vector<double> res_g(sp_g[0]);
        std::vector<double> res_jac_end(sp_jac_end[2 + sp_jac_end[1]]);
        std::vector<double> res_jac_start(sp_jac_start[2 + sp_jac_start[1]]);

        res_[0] = res_g.data();
        res_[1] = res_jac_end.data();
        res_[2] = res_jac_start.data();

        eval_segment_link(arg_.data(), res_.data(), iw_.data(), w_.data(), 0);

        LinkEvalResult result;
        result.g_link = Eigen::Map<Eigen::VectorXd>(res_g.data(), res_g.size());
        result.jac_end = map_to_eigen_sparse(sp_jac_end, res_jac_end.data());
        result.jac_start = map_to_eigen_sparse(sp_jac_start, res_jac_start.data());

        return result;
    }
};