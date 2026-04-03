import casadi as cs
import numpy as np
import os

# ==========================================
# 1. 伪谱法参数设置
# ==========================================
d = 3       # Radau 多项式阶数 (对于相对平滑的振荡器，3阶足够)
N = 40      # 区间数 (Van der Pol 振荡较多，可以适当增加网格)
K = N + 1

n_x = 4     # 状态量: [x0, x1, J_cost, tf]
n_u = 1     # 控制量: [u]

tau_root = cs.collocation_points(d, 'radau')
tau = np.append(0, tau_root) 

C = np.zeros((d+1, d+1))
for j in range(d+1):
    p = np.poly1d([1.0])
    for r in range(d+1):
        if r != j:
            p = np.poly1d([1.0, -tau[r]]) * p / (tau[j]-tau[r])
    dp = np.polyder(p)
    for r in range(d+1):
        C[j,r] = dp(tau[r])

# ==========================================
# 2. 连续时间动力学 (Van der Pol)
# ==========================================
def create_continuous_dynamics():
    states = cs.SX.sym('x', n_x)
    controls = cs.SX.sym('u', n_u)
    
    x0, x1, J_cost, tf = states[0], states[1], states[2], states[3]
    u = controls[0]
    
    # Van der Pol 核心动力学方程
    x0_dot = x1
    x1_dot = (1.0 - x0**2) * x1 - x0 + u
    
    # Lagrange 目标函数积分项转化为状态量的微分 (Mayer 形式)
    # J = \int (x0^2 + x1^2 + u^2) dt
    J_cost_dot = x0**2 + x1**2 + u**2
    
    tf_dot = 0.0  # tf 作为全局参数，导数为 0
    
    rhs = cs.vertcat(x0_dot, x1_dot, J_cost_dot, tf_dot)
    return cs.Function("f_cont", [states, controls], [rhs]).expand()

f_cont = create_continuous_dynamics()

# ==========================================
# 3. 伪谱配点函数
# ==========================================
def create_collocation_function():
    X_k = cs.MX.sym('X_k', n_x)
    W_k = cs.MX.sym('W_k', d*n_x + d*n_u)
    
    Xc = []; Uc = []
    offset = 0
    for j in range(d):
        Xc.append(W_k[offset : offset+n_x])
        offset += n_x
    for j in range(d):
        Uc.append(W_k[offset : offset+n_u])
        offset += n_u

    dt = X_k[3] / N  # tf 在索引 3
    eqs = []
    for j in range(1, d+1):
        dX_dt = C[0, j] * X_k
        for r in range(d):
            dX_dt += C[r+1, j] * Xc[r]
        f_eval = f_cont(Xc[j-1], Uc[j-1])
        eqs.append(dX_dt - dt * f_eval)
        
    eqs_cat = cs.vertcat(*eqs)
    X_end = Xc[-1]
    
    return cs.Function("colloc_fun", [X_k, W_k], [X_end, eqs_cat])

colloc_fun = create_collocation_function()

# ==========================================
# 4. 阶段约束与边界条件
# ==========================================
def path_constraints(W_k, X_k, k):
    cc = []
    if k < K - 1:
        # 配点动力学约束
        _, eqs = colloc_fun(X_k, W_k)
        cc.append((np.zeros(d*n_x), eqs, np.zeros(d*n_x))) 
        
        # 控制量约束限制 (可选，为了防止探索发散设定一个合理的物理界限)
        offset_x = 0; offset_u = d * n_x
        for j in range(d):
            U_cj = W_k[offset_u + j*n_u : offset_u + (j+1)*n_u]
            u_val = U_cj[0]
            cc.append((-10.0, u_val, 10.0))  
            
    # Dymos Van der Pol 算例的标准边界条件
    if k == 0:
        cc.append((1.0, X_k[0], 1.0))   # x0(0) = 1.0
        cc.append((1.0, X_k[1], 1.0))   # x1(0) = 1.0
        cc.append((0.0, X_k[2], 0.0))   # 代价积分初始为 0
    
    # 强制固定时间 tf = 10.0
    cc.append((10.0, X_k[3], 10.0))
        
    return cc

def discrete_dynamics(W_k, X_k, k):
    X_next, _ = colloc_fun(X_k, W_k)
    return X_next

# ==========================================
# 5. 装配 Opti 问题与目标函数
# ==========================================
opti = cs.Opti()

nx_list = [n_x for _ in range(K)]
nu_list = [d*n_x + d*n_u for _ in range(K-1)] + [0] 
ng_list = []

x = []; u = []
for k in range(K):
    x.append(opti.variable(nx_list[k]))
    u.append(opti.variable(nu_list[k]))

for k in range(K):
    if k < K - 1:
        opti.subject_to(x[k+1] == discrete_dynamics(u[k], x[k], k)) 
        
    path_constr = path_constraints(u[k], x[k], k)
    ng_list.append(0)
    for constr in path_constr:
        ng_list[-1] += constr[1].nnz()
        opti.subject_to((constr[0] <= constr[1]) <= constr[2])

# 【核心】：最小化终端节点的 J_cost (即 x[N][2])
J_obj = x[N][2]
opti.minimize(J_obj)

# ==========================================
# 6. 初值猜测 (Warm-start)
# ==========================================
for k in range(K):
    opti.set_initial(x[k][0], 1.0)
    opti.set_initial(x[k][1], 1.0)
    opti.set_initial(x[k][2], 0.0)
    opti.set_initial(x[k][3], 10.0)
        
    if k < K - 1:
        u_guess = []
        for j in range(d):
            u_guess.extend([1.0, 1.0, 0.0, 10.0]) # 内部状态猜测
        for j in range(d):
            u_guess.extend([0.0])                 # 内部控制猜测
            
        opti.set_initial(u[k], u_guess)

# ==========================================
# 7. 导出给 fatrop 的 C 代码及展平维度元数据
# ==========================================
opti.solver('fatrop', {
    'structure_detection': 'manual', 
    'nx': nx_list, 
    'nu': nu_list, 
    'ng': ng_list, 
    'N': N, 
    'expand': True, 
    'fatrop.mu_init': 1e-1
})

current_dir = os.path.dirname(os.path.abspath(__file__))
generated_dir = os.path.join(current_dir, '..', 'generated')
os.makedirs(generated_dir, exist_ok=True)
os.chdir(generated_dir)

print("[*] Generating Van der Pol KRONOS NLP C-code ...")
opti.to_function("kronos_nlp", [], [opti.f, opti.x]).generate('nlp_code.c', {"with_header": True})

# ---------------------------------------------------------
# 动态生成用于 C++ 展平输出的元数据 (兼容你的 main.cpp)
# ---------------------------------------------------------
x_base_names = ["x0", "x1", "J_cost", "tf"]
u_base_names = ["u_control"] 

tau_str = ", ".join([str(t) for t in tau_root]) 

with open('problem_metadata.h', 'w') as f:
    f.write("#pragma once\n\n")
    f.write("// KRONOS Van der Pol Metadata\n")
    f.write(f"#define KRONOS_N  {N}\n")
    f.write(f"#define KRONOS_NX {n_x}\n")
    f.write(f"#define KRONOS_NU_BASE {n_u}\n")
    f.write(f"#define KRONOS_D  {d}\n")
    f.write(f"#define KRONOS_TF_INDEX 3\n")
    f.write(f"#define KRONOS_TAU_ROOT {{{tau_str}}}\n\n") 
    f.write("// Variable Names\n")
    f.write(f'#define KRONOS_X_NAMES "{",".join(x_base_names)}"\n')
    f.write(f'#define KRONOS_U_NAMES "{",".join(u_base_names)}"\n')

os.chdir(current_dir)
print(f"✅ Code generation successful! Files automatically saved to: {generated_dir}")