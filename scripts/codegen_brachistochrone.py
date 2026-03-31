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
    # Seg 1 动力学
    v1, theta1 = X1[2, i], U1[0, i]
    f1 = ca.vertcat(v1 * ca.cos(theta1), v1 * ca.sin(theta1), -9.81 * ca.sin(theta1))
    g_list.append(dX1[:, i] - (T / 4) * f1)
    
    # Seg 2 动力学
    v2, theta2 = X2[2, i], U2[0, i]
    f2 = ca.vertcat(v2 * ca.cos(theta2), v2 * ca.sin(theta2), -9.81 * ca.sin(theta2))
    g_list.append(dX2[:, i] - (T / 4) * f2)

g = ca.vertcat(*g_list)  
n_g = g.shape[0]

f = T
lam = ca.SX.sym('lam', n_g)
L = f + ca.dot(lam, g)

# 计算导数信息（注意：这里使用了 ca.densify 强制转为稠密矩阵，方便第一步的 Eigen 接入）
grad_L = ca.jacobian(L, w).T      
H_L = ca.densify(ca.hessian(L, w)[0])         
A_g = ca.densify(ca.jacobian(g, w))           

# 命名输入输出，便于 C++ 侧识别
kkt_func = ca.Function('kkt_func', 
                       [w, lam], 
                       [H_L, A_g, grad_L, g], 
                       ['w', 'lam'], 
                       ['H', 'A', 'grad_L', 'g'])

# ================= 3. 初值构造 (用于导出至 C++) =================
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

# 4.1 生成 CasADi C 代码
opts = {"with_header": True}
cg = ca.CodeGenerator('kkt_funcs.c', opts)
cg.add(kkt_func)
cg.generate(out_dir + '/')
print(f"✅ 成功生成 CasADi C 代码至: {out_dir}/kkt_funcs.c")

# 4.2 生成初始猜测与宏定义的 C++ 头文件
config_file = os.path.join(out_dir, 'kronos_config.h')
with open(config_file, 'w') as f:
    f.write("#ifndef KRONOS_CONFIG_H\n")
    f.write("#define KRONOS_CONFIG_H\n\n")
    
    f.write(f"#define KRONOS_N_NODES {N}\n")
    f.write(f"#define KRONOS_N_W {n_w}\n")
    f.write(f"#define KRONOS_N_G {n_g}\n\n")
    
    f.write("// Initial guess for state variables (w)\n")
    f.write(f"static const double w_init[{n_w}] = {{\n")
    f.write("    " + ", ".join([f"{val:.15f}" for val in w_k]))
    f.write("\n};\n\n")
    
    f.write("// Initial guess for multipliers (lam)\n")
    f.write(f"static const double lam_init[{n_g}] = {{\n")
    f.write("    " + ", ".join(["0.0"] * n_g))
    f.write("\n};\n\n")
    
    f.write("#endif // KRONOS_CONFIG_H\n")
    
print(f"✅ 成功生成 KRONOS 配置文件至: {config_file}")
print("=======================================")
print(f"N_W (变量数): {n_w}")
print(f"N_G (约束数): {n_g}")
print("=======================================")