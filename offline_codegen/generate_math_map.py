import casadi as cs
import numpy as np
import os

N = 10
opti = cs.Opti()

# 交替定义变量 [x0, u0, x1, u1, ...]
x = []
u = []
nx_list = [1] * (N + 1)
nu_list = [1] * N + [0] 

for k in range(N + 1):
    x.append(opti.variable(nx_list[k]))
    u.append(opti.variable(nu_list[k]))

cost = 0
ng_list = []

for k in range(N + 1):
    ng_k = 0
    if k < N:
        # 递推方程
        opti.subject_to(x[k+1] == x[k] + u[k])
        # 目标函数
        cost += u[k]**2
        # 前5步步长约束: u >= 0.15
        if k < 5:
            opti.subject_to(u[k] >= 0.15)
            ng_k += 1
            
    # 边界约束
    if k == 0:
        opti.subject_to(x[k] == 0.0)
        ng_k += 1
    elif k == N:
        opti.subject_to(x[k] == 1.0)
        ng_k += 1
    
    ng_list.append(ng_k)

opti.minimize(cost)

# 配置求解器
opti.solver('fatrop', {
    'structure_detection': 'manual',
    'nx': nx_list,
    'nu': nu_list,
    'ng': ng_list,
    'N': N,
    'expand': True
})

# 导出 C 代码，确保顺序为 [目标函数 f, 变量 x]
current_dir = os.path.dirname(os.path.abspath(__file__))
generated_dir = os.path.join(current_dir, '..', 'generated')
os.makedirs(generated_dir, exist_ok=True)
os.chdir(generated_dir)
opti.to_function("kronos_nlp", [], [opti.f, opti.x]).generate('nlp_code.c', {"with_header": True})
print(f"✅ Theoretical test case generated.")