#include "kronos_nlp_wrapper.hpp"
#include "../generated/kkt_funcs.h" // CasADi 生成的底层 C 接口

namespace kronos {

NlpWrapper::NlpWrapper() {
    // 预先分配非零元素所需的空间大小
    H_nonzeros_.resize(KRONOS_H_NNZ);
    A_nonzeros_.resize(KRONOS_A_NNZ);
    
    // 【兼容性处理】：如果无不等式，分配大小为 1 的假空间防止 Eigen 抛出越界断言
    if (KRONOS_AH_NNZ > 0) {
        Ah_nonzeros_.resize(KRONOS_AH_NNZ);
    } else {
        Ah_nonzeros_.resize(1); 
    }
}

void NlpWrapper::evaluate(const VectorXd& w, const VectorXd& lam, const VectorXd& z,
                          SparseMatrixXd& H, SparseMatrixXd& A_g, SparseMatrixXd& A_h, 
                          VectorXd& grad_L, VectorXd& g, VectorXd& h, double& f_val) {
    
    // 1. 设置 CasADi 的输入指针 (arg)
    // 根据 Python 生成脚本，输入顺序为 [w, lam, z]
    const double* arg[3];
    arg[0] = w.data();
    arg[1] = lam.data();
    arg[2] = z.data();

    // 2. 设置 CasADi 的输出指针 (res)
    // 根据 Python 生成脚本，输出顺序为 [H, A_g, grad_L, g, A_h, h, f]
    double* res[7];
    res[0] = H_nonzeros_.data();
    res[1] = A_nonzeros_.data();
    res[2] = grad_L.data();
    res[3] = g.data();
    res[4] = Ah_nonzeros_.data();
    res[5] = h.data();
    res[6] = &f_val; // <-- 接收目标函数值 f_val

    // 3. 调用 CasADi 生成的 C 代码
    kkt_func(arg, res, nullptr, nullptr, 0);

    // 4. 使用 Eigen::Map 零拷贝构造稀疏矩阵 H 和 A_g
    Eigen::Map<Eigen::SparseMatrix<double>> H_map(
        KRONOS_N_W, KRONOS_N_W, KRONOS_H_NNZ, 
        const_cast<int*>(H_colind), const_cast<int*>(H_row), H_nonzeros_.data()
    );
        
    Eigen::Map<Eigen::SparseMatrix<double>> Ag_map(
        KRONOS_N_G, KRONOS_N_W, KRONOS_A_NNZ, 
        const_cast<int*>(A_colind), const_cast<int*>(A_row), A_nonzeros_.data()
    );

    H = H_map;
    A_g = Ag_map;

    // 5. 【兼容性处理】：仅在存在不等式约束时才映射 A_h
    if (KRONOS_N_H > 0) {
        Eigen::Map<Eigen::SparseMatrix<double>> Ah_map(
            KRONOS_N_H, KRONOS_N_W, KRONOS_AH_NNZ, 
            const_cast<int*>(Ah_colind), const_cast<int*>(Ah_row), Ah_nonzeros_.data()
        );
        A_h = Ah_map;
    } else {
        // 无不等式时，置为一个 0 行 N_W 列的空稀疏矩阵
        A_h.resize(0, KRONOS_N_W);
    }
}

} // namespace kronos