import casadi as ca
import numpy as np
import os
from numpy.polynomial.legendre import Legendre

# ================= 1. LGL 节点与微分矩阵 =================
def get_lgl_nodes_and_D(N):
    n = N - 1
    tau = np.zeros(N)
    tau[0], tau[-1] = -1.0, 1.0
    c = np.zeros(n + 1)
    c[n] = 1
    P_n_deriv = Legendre(c).deriv()
    if n > 1:
        tau[1:-1] = np.sort(P_n_deriv.roots())
        
    c_i = np.ones(N)
    c_i[0], c_i[-1] = 2.0, 2.0
    D = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i != j:
                D[i, j] = (c_i[i] / c_i[j]) * ((-1)**(i+j)) / (tau[i] - tau[j])
    for i in range(N):
        D[i, i] = -np.sum(D[i, :])
    return tau, D

N = 20
tau, D = get_lgl_nodes_and_D(N)

# ================= 2. 构造优化问题与自动微分 =================
n_w = 8 * N + 1

X1 = ca.SX.sym('X1', 3, N)
U1 = ca.SX.sym('U1', 1, N)
X2 = ca.SX.sym('X2', 3, N)
U2 = ca.SX.sym('U2', 1, N)
T  = ca.SX.sym('T')

w = ca.vertcat(ca.vec(X1), ca.vec(U1), ca.vec(X2), ca.vec(U2), T)

g_list = []
# (A) 初始点约束
g_list.append(X1[:, 0] - ca.vertcat(0, 0, 0))
# (B) 终止点约束
g_list.append(X2[0:2, -1] - ca.vertcat(10, -5))
# (C) 段间连续性约束
g_list.append(X1[:, -1] - X2[:, 0])
# (D) 两段的动力学约束
dX1 = ca.mtimes(X1, D.T)
dX2 = ca.mtimes(X2, D.T)
for i in range(N):
    v1, theta1 = X1[2, i], U1[0, i]
    f1 = ca.vertcat(v1 * ca.cos(theta1), v1 * ca.sin(theta1), -9.81 * ca.sin(theta1))
    g_list.append(dX1[:, i] - (T / 4) * f1)
    
    v2, theta2 = X2[2, i], U2[0, i]
    f2 = ca.vertcat(v2 * ca.cos(theta2), v2 * ca.sin(theta2), -9.81 * ca.sin(theta2))
    g_list.append(dX2[:, i] - (T / 4) * f2)

g = ca.vertcat(*g_list)  
n_g = g.shape[0]

# 【新增】：不等式约束（当前为空，自动处理长度为 0）
h_list = []
h = ca.vertcat(*h_list) if len(h_list) > 0 else ca.SX(0, 1)
n_h = h.shape[0]

f = T
lam = ca.SX.sym('lam', n_g)
# 【新增】：不等式乘子 z
z = ca.SX.sym('z', n_h) if n_h > 0 else ca.SX.sym('z', 0)

# 拉格朗日函数 L = f + lam^T g - z^T h
L = f + ca.dot(lam, g)
if n_h > 0:
    L -= ca.dot(z, h)

# 计算导数信息
grad_L = ca.jacobian(L, w).T      
H_L = ca.hessian(L, w)[0]         
A_g = ca.jacobian(g, w)           
A_h = ca.jacobian(h, w) if n_h > 0 else ca.SX(0, n_w) # 【新增】

# 提取稀疏结构
def get_sparsity(mat):
    sp = mat.sparsity()
    return sp.nnz(), sp.colind(), sp.row()

H_nnz, H_colind, H_row = get_sparsity(H_L)
A_nnz, A_colind, A_row = get_sparsity(A_g)
Ah_nnz, Ah_colind, Ah_row = get_sparsity(A_h) # 【新增】

# 命名输入输出，加入 z, A_h, h
kkt_func = ca.Function('kkt_func', 
                       [w, lam, z], 
                       [H_L, A_g, grad_L, g, A_h, h, f], 
                       ['w', 'lam', 'z'], 
                       ['H', 'A', 'grad_L', 'g', 'A_h', 'h', 'f_obj'])

# ================= 3. 初值构造 =================
w_k = np.zeros(n_w)
w_k[-1] = 2.0 

for k_seg, X_idx, U_idx, time_offset in [(0, 0, 3*N, 0.0), (1, 4*N, 7*N, 0.5)]:
    for i in range(N):
        t_global = time_offset + (tau[i] + 1) / 4.0
        w_k[X_idx + i*3 + 0] = 10 * t_global
        w_k[X_idx + i*3 + 1] = -5 * t_global
        w_k[X_idx + i*3 + 2] = np.sqrt(2 * 9.81 * 5 * t_global + 0.1)
        w_k[U_idx + i] = -np.pi / 4

# ================= 4. C 代码生成 =================
out_dir = os.path.join(os.path.dirname(__file__), '..', 'generated')
os.makedirs(out_dir, exist_ok=True)

opts = {"with_header": True}
cg = ca.CodeGenerator('kkt_funcs.c', opts)
cg.add(kkt_func)
cg.generate(out_dir + '/')

# 辅助函数：处理 0 长度数组
def format_c_array(name, arr):
    if len(arr) == 0:
        return f"static const int {name}[1] = {{0}};\n"
    return f"static const int {name}[{len(arr)}] = {{" + ", ".join(map(str, arr)) + "};\n"

config_file = os.path.join(out_dir, 'kronos_config.h')
with open(config_file, 'w') as f:
    f.write("#ifndef KRONOS_CONFIG_H\n")
    f.write("#define KRONOS_CONFIG_H\n\n")
    
    f.write(f"#define KRONOS_N_NODES {N}\n")
    f.write(f"#define KRONOS_N_W {n_w}\n")
    f.write(f"#define KRONOS_N_G {n_g}\n")
    f.write(f"#define KRONOS_N_H {n_h}\n\n") # 【新增】
    
    f.write(f"#define KRONOS_H_NNZ {H_nnz}\n")
    f.write(f"#define KRONOS_A_NNZ {A_nnz}\n")
    f.write(f"#define KRONOS_AH_NNZ {Ah_nnz}\n\n") # 【新增】
    
    f.write("// Sparsity pattern for Hessian\n")
    f.write(format_c_array("H_colind", H_colind))
    f.write(format_c_array("H_row", H_row) + "\n")
    
    f.write("// Sparsity pattern for Equality Jacobian\n")
    f.write(format_c_array("A_colind", A_colind))
    f.write(format_c_array("A_row", A_row) + "\n")

    f.write("// Sparsity pattern for Inequality Jacobian\n")
    f.write(format_c_array("Ah_colind", Ah_colind))
    f.write(format_c_array("Ah_row", Ah_row) + "\n")

    f.write("// Initial guess for state variables (w)\n")
    f.write(f"static const double w_init[{max(n_w, 1)}] = {{\n")
    f.write("    " + ", ".join([f"{val:.15f}" for val in w_k]))
    f.write("\n};\n\n")
    
    f.write("// Initial guess for multipliers (lam)\n")
    f.write(f"static const double lam_init[{max(n_g, 1)}] = {{\n")
    f.write("    " + ", ".join(["0.0"] * n_g))
    f.write("\n};\n\n")

    f.write("// Initial guess for inequality multipliers (z)\n")
    f.write(f"static const double z_init[{max(n_h, 1)}] = {{\n")
    f.write("    " + ", ".join(["1.0"] * max(n_h, 1)))
    f.write("\n};\n\n")
    
    f.write("#endif // KRONOS_CONFIG_H\n")