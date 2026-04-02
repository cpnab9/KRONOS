# offline_codegen/generate_simple_constrained.py
import casadi as cs
import numpy as np
import os

# 1. 问题参数
N = 20           
T = 1.0          
dt = T / N       

opti = cs.Opti()

# 2. 按阶段交替定义变量 [x0, u0, x1, u1, ...]
nx_list = [2] * (N + 1)
nu_list = [1] * N + [0] 

x = []
u = []
for k in range(N + 1):
    x.append(opti.variable(nx_list[k]))
    u.append(opti.variable(nu_list[k]))

cost = 0
ng_list = [] # 记录每个阶段除了动力学之外的额外约束数量

# ==========================================
# 【核心修复】：严格按时间步 k 添加所有方程并统计 ng
# ==========================================
for k in range(N + 1):
    ng_k = 0 # 当前阶段 k 的路径/边界约束数量
    
    # 动力学与路径约束 (前 N 步)
    if k < N:
        pos_k = x[k][0]
        vel_k = x[k][1]
        acc_k = u[k]
        
        # 动力学约束 (CasADi 的 fatrop 接口会自动将其识别为 nx，不计入 ng_k)
        pos_next = pos_k + vel_k * dt
        vel_next = vel_k + acc_k * dt
        opti.subject_to(x[k+1][0] == pos_next)
        opti.subject_to(x[k+1][1] == vel_next)
        
        # 路径约束 (计入 ng_k)
        opti.subject_to(opti.bounded(-2.0, vel_k, 2.0)) 
        ng_k += 1
        opti.subject_to(opti.bounded(-5.0, acc_k, 5.0))  
        ng_k += 1
        
        # 累加目标函数
        cost += acc_k**2 * dt
        
    # 边界条件必须在它所属的阶段 k 声明，并计入 ng_k
    if k == 0:
        opti.subject_to(x[k][0] == 0.0) # 起点位置
        ng_k += 1
        opti.subject_to(x[k][1] == 0.0) # 起点速度
        ng_k += 1
    elif k == N:
        opti.subject_to(x[k][0] == 1.0) # 终点位置
        ng_k += 1
        opti.subject_to(x[k][1] == 0.0) # 终点速度
        ng_k += 1
        
    # 将当前阶段的约束数量记录到列表中
    ng_list.append(ng_k)

opti.minimize(cost)

# 3. 配置 Fatrop 求解器结构
opti.solver('fatrop', {
    'structure_detection': 'manual',
    'nx': nx_list,
    'nu': nu_list,
    'ng': ng_list,
    'N': N,
    'expand': True
})

# 4. 导出 C 代码
current_dir = os.path.dirname(os.path.abspath(__file__))
generated_dir = os.path.join(current_dir, '..', 'generated')
os.makedirs(generated_dir, exist_ok=True)
os.chdir(generated_dir)

# 无论什么问题，导出的 C 函数名统一为 "kronos_nlp"
opti.to_function("kronos_nlp", [], [opti.f, opti.x]).generate('nlp_code.c', {"with_header": True})
os.chdir(current_dir)
print(f"✅ Simple constrained problem code generation successful! Saved to: {generated_dir}")