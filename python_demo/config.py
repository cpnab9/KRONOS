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
u_max = 10.0 * np.pi / 180.0 * t_ref 
W_penalty = 100000.0

# ==========================================
# 禁飞区设置 (1: 静态, 2: 动态/突发)
# ==========================================
theta_NFZ1, phi_NFZ1, R_NFZ1 = 5.0 * np.pi / 180.0, 30.0 * np.pi / 180.0, 500000.0 / R0
theta_NFZ2, phi_NFZ2, R_NFZ2 = 2.5 * np.pi / 180.0, 60.0 * np.pi / 180.0, 500000.0 / R0

# ==========================================
# 伪谱法配置与矩阵初始化
# ==========================================
d = 3       
N = 25      
K = N + 1

n_x = 8     # 状态量: [r, theta, phi, V, gamma, psi, sigma, tf]
n_u = 4     # 控制量: [u_rate, slack_Q, slack_q, slack_n] 

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