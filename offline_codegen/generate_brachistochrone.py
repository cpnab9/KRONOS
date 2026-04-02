import casadi as cs
import numpy as np
import os

# ==========================================
# 1. 伪谱法参数设置
# ==========================================
d = 5       
N = 10      
K = N + 1   

n_x = 4     # [x, y, v, tf]
n_u = 1     # [theta]

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
# 2. 连续时间动力学
# ==========================================
def create_continuous_dynamics():
    states = cs.SX.sym('x', n_x)
    controls = cs.SX.sym('u', n_u)
    
    x, y, v, tf = states[0], states[1], states[2], states[3]
    theta = controls[0]
    g = 9.81
    
    rhs = cs.vertcat(
        v * cs.cos(theta),
        v * cs.sin(theta),
        -g * cs.sin(theta),
        0.0  
    )
    return cs.Function("f_cont", [states, controls], [rhs]).expand()

f_cont = create_continuous_dynamics()

# ==========================================
# 3. 伪谱配点函数
# ==========================================
def create_collocation_function():
    X_k = cs.MX.sym('X_k', n_x)
    W_k = cs.MX.sym('W_k', d*n_x + d*n_u)
    
    Xc = [] 
    Uc = [] 
    offset = 0
    for j in range(d):
        Xc.append(W_k[offset : offset+n_x])
        offset += n_x
    for j in range(d):
        Uc.append(W_k[offset : offset+n_u])
        offset += n_u

    dt = X_k[3] / N
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
        _, eqs = colloc_fun(X_k, W_k)
        cc.append((np.zeros(d*n_x), eqs, np.zeros(d*n_x)))
        
        offset_x = 0
        offset_u = d * n_x
        for j in range(d):
            X_cj = W_k[offset_x + j*n_x : offset_x + (j+1)*n_x]
            U_cj = W_k[offset_u + j*n_u : offset_u + (j+1)*n_u]
            cc.append((0.0, X_cj[2], 200.0))
            cc.append((-cs.pi, U_cj[0], cs.pi))
            
    if k == 0:
        cc.append((0., X_k[0], 0.))
        cc.append((0., X_k[1], 0.))
        cc.append((1e-3, X_k[2], 1e-3))
    elif k == K - 1:
        cc.append((0., X_k[0] - 10.0, 0.))
        cc.append((0., X_k[1] + 5.0,  0.))
        
    cc.append((0.1, X_k[3], 20.0))
    return cc

# 【修正】：将 discrete_dynamics 的定义提前到调用之前
def discrete_dynamics(W_k, X_k, k):
    X_next, _ = colloc_fun(X_k, W_k)
    return X_next

# ==========================================
# 5. 装配 Opti 问题
# ==========================================
opti = cs.Opti()

nx_list = [n_x for _ in range(K)]
nu_list = [d*n_x + d*n_u for _ in range(K-1)] + [0] 
ng_list = []

x = []
u = []
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

J = x[N][3] 
opti.minimize(J)

# ==========================================
# 6. 提供高质量初始猜测值 (Warm-start)
# ==========================================
for k in range(K):
    tau_guess = k / N
    opti.set_initial(x[k][0], tau_guess * 10.0)
    opti.set_initial(x[k][1], tau_guess * -5.0 - 4.0 * np.sin(tau_guess * np.pi))
    opti.set_initial(x[k][2], 5.0)   
    opti.set_initial(x[k][3], 2.0)   

for k in range(K-1):
    W_guess = []
    tau_k = k / N
    for j in range(d):
        tau_c = tau_k + tau_root[j] * (1.0 / N)
        x_guess = tau_c * 10.0
        y_guess = tau_c * -5.0 - 4.0 * np.sin(tau_c * np.pi)
        v_guess = 5.0
        tf_guess = 2.0
        W_guess.extend([x_guess, y_guess, v_guess, tf_guess])
    for j in range(d):
        W_guess.extend([-1.0]) 
    opti.set_initial(u[k], W_guess)
# ==========================================
# 7. 导出给 fatrop 的 C 代码 (精准对齐 C++ 工程结构)
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

# 获取当前 Python 脚本所在绝对目录 (offline_codegen)
current_dir = os.path.dirname(os.path.abspath(__file__))
# 推导项目根目录下的 generated 文件夹路径
generated_dir = os.path.join(current_dir, '..', 'generated')

# 如果 generated 文件夹不存在，自动创建它
os.makedirs(generated_dir, exist_ok=True)

# 【核心修复】：切换当前 Python 进程的工作目录到 generated 文件夹
os.chdir(generated_dir)

# 现在只需传入纯文件名，避开 CasADi 的绝对路径解析错误
opti.to_function("kronos_nlp", [], [opti.x]).generate('kronos_nlp_functions.c', {"with_header": True})

# 出于良好的脚本习惯，切回原来的工作目录
os.chdir(current_dir)

print(f"✅ Code generation successful! Files automatically saved to: {generated_dir}")