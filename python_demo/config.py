# --- START OF FILE config.py ---
import numpy as np
import casadi as cs

# ==========================================
# 物理常数与飞行器参数
# ==========================================
R0 = 6378000.0         
g0 = 9.81              
V_ref = np.sqrt(R0 * g0) 
t_ref = np.sqrt(R0 / g0) 
Omega_e = 7.292115e-5 * t_ref 

m = 104305.0           
Aref = 391.22          
rho0 = 1.225           
hs = 7200.0            

Q_dot_max = 1500.0 * 1000.0
n_max = 2.5            
q_max = 18000.0        
k_Q = 1.65e-4

# 【修改1】：将速率限制拆分为倾侧角速率和攻角速率
sigma_rate_max = 10.0 * np.pi / 180.0 * t_ref 
alpha_rate_max = 5.0 * np.pi / 180.0 * t_ref   # 限制攻角最大变化率 (5 deg/s)

# 【修改2】：增加物理攻角的上下边界
alpha_max = 40.0 * np.pi / 180.0
alpha_min = 0.0 * np.pi / 180.0

W_penalty = 100000.0

# ==========================================
# 禁飞区设置 (1: 静态, 2: 动态/突发)
# ==========================================
theta_NFZ1, phi_NFZ1, R_NFZ1 = 5.0 * np.pi / 180.0, 30.0 * np.pi / 180.0, 500000.0 / R0
theta_NFZ2, phi_NFZ2, R_NFZ2 = 2.5 * np.pi / 180.0, 60.0 * np.pi / 180.0, 500000.0 / R0

# ==========================================
# 伪谱法配置与矩阵初始化
# ==========================================
d = 4     
N = 20      
K = N + 1

# 【修改3】：扩展状态维度和控制维度
n_x = 9     # 状态量: [r, theta, phi, V, gamma, psi, sigma, alpha, tf]
n_u = 5     # 控制量: [sigma_rate, alpha_rate, slack_Q, slack_q, slack_n] 

tau_root = cs.collocation_points(d, 'radau')
tau = np.append(0, tau_root) 

# 拉格朗日插值多项式求导矩阵 C
C = np.zeros((d+1, d+1))
for j in range(d+1):
    p = np.poly1d([1.0])
    for r in range(d+1):
        if r != j:
            p = np.poly1d([1.0, -tau[r]]) * p / (tau[j]-tau[r])
    dp = np.polyder(p)
    for r in range(d+1):
        C[j,r] = dp(tau[r])
# --- END OF FILE config.py ---