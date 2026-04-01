import casadi as ca
import numpy as np
import os

# ================= 1. 构造简单优化问题 =================
n_w = 2
w = ca.SX.sym('w', n_w)
x1, x2 = w[0], w[1]

# (A) 目标函数
f = x1**2 + x2**2

# (B) 等式约束
g_list = [
    x1 + x2 - 1.0
]
g = ca.vertcat(*g_list) if len(g_list) > 0 else ca.SX(0, 1)
n_g = g.shape[0]

# (C) 不等式约束: h(w) >= 0
# 【兼容性测试】：将 h_list 设为 [] 即可模拟最速降线等无不等式约束的情况！
h_list = [
    x1 - 0.8
]
h = ca.vertcat(*h_list) if len(h_list) > 0 else ca.SX(0, 1)
n_h = h.shape[0]

# ================= 2. 构造拉格朗日函数与自动微分 =================
lam = ca.SX.sym('lam', n_g) if n_g > 0 else ca.SX.sym('lam', 0)
z = ca.SX.sym('z', n_h) if n_h > 0 else ca.SX.sym('z', 0)

# 完整的拉格朗日函数 L = f + lam^T g - z^T h
# 注意：内点法中不等式乘子 z > 0，且通常写为 - z^T h
L = f
if n_g > 0:
    L += ca.dot(lam, g)
if n_h > 0:
    L -= ca.dot(z, h)

# 计算导数信息
grad_L = ca.jacobian(L, w).T      
H_L = ca.hessian(L, w)[0]         
A_g = ca.jacobian(g, w) if n_g > 0 else ca.SX(0, n_w)
A_h = ca.jacobian(h, w) if n_h > 0 else ca.SX(0, n_w)

def get_sparsity(mat):
    sp = mat.sparsity()
    return sp.nnz(), sp.colind(), sp.row()

H_nnz, H_colind, H_row = get_sparsity(H_L)
Ag_nnz, Ag_colind, Ag_row = get_sparsity(A_g)
Ah_nnz, Ah_colind, Ah_row = get_sparsity(A_h)

# 命名输入输出，新增了 z (输入) 以及 A_h, h (输出)
# 命名输入输出，新增了 z (输入) 以及 A_h, h (输出)
kkt_func = ca.Function('kkt_func', 
                       [w, lam, z], 
                       [H_L, A_g, grad_L, g, A_h, h, f], 
                       ['w', 'lam', 'z'], 
                       ['H', 'A_g', 'grad_L', 'g', 'A_h', 'h', 'f_obj'])

# ================= 3. 初值构造 =================
w_init = np.array([0.0, 0.0])
lam_init = np.zeros(n_g)
z_init = np.ones(n_h) # 不等式乘子初始值必须大于0

# ================= 4. C 代码生成 =================
out_dir = os.path.join(os.path.dirname(__file__), '..', 'generated')
os.makedirs(out_dir, exist_ok=True)

# 4.1 生成 CasADi C 代码
opts = {"with_header": True}
cg = ca.CodeGenerator('kkt_funcs.c', opts)
cg.add(kkt_func)
cg.generate(out_dir + '/')
print(f"✅ 成功生成 CasADi C 代码至: {out_dir}/kkt_funcs.c")

# 4.2 生成初始猜测与宏定义的 C++ 头文件
def format_c_array(name, arr):
    # C++ 标准不允许长度为 0 的数组，若为 0，填充一个 dummy 元素以避开编译报错
    if len(arr) == 0:
        return f"static const int {name}[1] = {{0}};\n"
    return f"static const int {name}[{len(arr)}] = {{" + ", ".join(map(str, arr)) + "};\n"

config_file = os.path.join(out_dir, 'kronos_config.h')
with open(config_file, 'w') as f:
    f.write("#ifndef KRONOS_CONFIG_H\n")
    f.write("#define KRONOS_CONFIG_H\n\n")
    
    f.write(f"#define KRONOS_N_W {n_w}\n")
    f.write(f"#define KRONOS_N_G {n_g}\n")
    f.write(f"#define KRONOS_N_H {n_h}\n\n") # 新增不等式数量
    
    f.write(f"#define KRONOS_H_NNZ {H_nnz}\n")
    f.write(f"#define KRONOS_A_NNZ {Ag_nnz}\n") # 改名或兼容旧的 A_NNZ
    f.write(f"#define KRONOS_AH_NNZ {Ah_nnz}\n\n")
    
    f.write("// Sparsity pattern for Hessian\n")
    f.write(format_c_array("H_colind", H_colind))
    f.write(format_c_array("H_row", H_row) + "\n")
    
    f.write("// Sparsity pattern for Equality Jacobian\n")
    f.write(format_c_array("A_colind", Ag_colind))
    f.write(format_c_array("A_row", Ag_row) + "\n")

    f.write("// Sparsity pattern for Inequality Jacobian\n")
    f.write(format_c_array("Ah_colind", Ah_colind))
    f.write(format_c_array("Ah_row", Ah_row) + "\n")

    f.write(f"static const double w_init[{max(n_w, 1)}] = {{" + ", ".join([f"{val:.5f}" for val in w_init]) + "};\n")
    f.write(f"static const double lam_init[{max(n_g, 1)}] = {{" + ", ".join([f"{val:.5f}" for val in lam_init]) + "};\n")
    f.write(f"static const double z_init[{max(n_h, 1)}] = {{" + ", ".join([f"{val:.5f}" for val in z_init]) + "};\n\n")
    
    f.write("#endif // KRONOS_CONFIG_H\n")
    
print(f"✅ 成功生成 KRONOS 配置文件至: {config_file}")