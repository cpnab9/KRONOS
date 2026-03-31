#include "kronos_nlp_wrapper.hpp"
#include "../generated/kkt_funcs.h" // CasADi 生成的底层 C 接口

namespace kronos {

NlpWrapper::NlpWrapper() {
    // 预先分配非零元素所需的空间大小 (来自 kronos_config.h)
    H_nonzeros_.resize(KRONOS_H_NNZ);
    A_nonzeros_.resize(KRONOS_A_NNZ);
}

void NlpWrapper::evaluate(const VectorXd& w, const VectorXd& lam, 
                          SparseMatrixXd& H, SparseMatrixXd& A, 
                          VectorXd& grad_L, VectorXd& g) {
    
    // 1. 设置 CasADi 的输入指针 (arg)
    const double* arg[2];
    arg[0] = w.data();
    arg[1] = lam.data();

    // 2. 设置 CasADi 的输出指针 (res)
    // res[0]: Hessian 的非零元素数组
    // res[1]: Jacobian 的非零元素数组
    // res[2]: 目标函数梯度
    // res[3]: 约束评估值
    double* res[4];
    res[0] = H_nonzeros_.data();
    res[1] = A_nonzeros_.data();
    res[2] = grad_L.data();
    res[3] = g.data();

    // 3. 调用 CasADi 生成的 C 代码
    // 根据 kkt_funcs.h 的标准签名传递 (arg, res, iw, w, mem)
    kkt_func(arg, res, nullptr, nullptr, 0);

    // 4. 使用 Eigen::Map 零拷贝构造稀疏矩阵
    // Map 的构造函数参数顺序: (rows, cols, nnz, outerIndexPtr, innerIndexPtr, valuePtr)
    // 注意：因 Eigen 的 API 需要非 const 指针，这里用 const_cast 转换静态数组，
    // 但 Map 只是做映射，底层绝对不会修改配置头文件中的结构数组。
    Eigen::Map<Eigen::SparseMatrix<double>> H_map(
        KRONOS_N_W, KRONOS_N_W, KRONOS_H_NNZ, 
        const_cast<int*>(H_colind), const_cast<int*>(H_row), H_nonzeros_.data()
    );
        
    Eigen::Map<Eigen::SparseMatrix<double>> A_map(
        KRONOS_N_G, KRONOS_N_W, KRONOS_A_NNZ, 
        const_cast<int*>(A_colind), const_cast<int*>(A_row), A_nonzeros_.data()
    );

    // 5. 赋值给传入的参数（这里会发生一次结构化深拷贝，对稀疏矩阵来说开销极低）
    H = H_map;
    A = A_map;
}

} // namespace kronos