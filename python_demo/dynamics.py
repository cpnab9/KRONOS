import casadi as cs
import config as cfg

def create_continuous_dynamics():
    """构建并返回连续时间动力学的 CasADi 函数"""
    states = cs.SX.sym('x', cfg.n_x)
    controls = cs.SX.sym('u', cfg.n_u)
    
    r, theta, phi, V, gamma, psi, sigma, tf = states[0], states[1], states[2], states[3], states[4], states[5], states[6], states[7]
    u = controls[0] 
    
    safe_r = cs.fmax(r, 0.5)          
    safe_V = cs.fmax(V, 1e-4)         
    safe_cos_phi = cs.fmax(cs.cos(phi), 1e-4)
    safe_cos_gamma = cs.fmax(cs.cos(gamma), 1e-4)

    h_dim = (r - 1.0) * cfg.R0
    rho = cfg.rho0 * cs.exp(-h_dim / cfg.hs)
    V_dim = V * cfg.V_ref

    alpha_deg = cs.if_else(V_dim > 4570.0, 40.0, 40.0 - 0.20705 * ((V_dim - 4570.0) / 340.0)**2)
    CL = -0.041065 + 0.016292 * alpha_deg + 0.0002602 * alpha_deg**2
    CD = 0.080505 - 0.03026 * CL + 0.86495 * CL**2

    L_acc = (0.5 * rho * V_dim**2 * cfg.Aref * CL) / (cfg.m * cfg.g0)
    D_acc = (0.5 * rho * V_dim**2 * cfg.Aref * CD) / (cfg.m * cfg.g0)

    r_dot = V * cs.sin(gamma)
    theta_dot = V * cs.cos(gamma) * cs.sin(psi) / (safe_r * safe_cos_phi)
    phi_dot = V * cs.cos(gamma) * cs.cos(psi) / safe_r
    V_dot = -D_acc - cs.sin(gamma)/(safe_r**2) + cfg.Omega_e**2 * safe_r * cs.cos(phi) * (cs.sin(gamma)*cs.cos(phi) - cs.cos(gamma)*cs.sin(phi)*cs.cos(psi))
    
    gamma_dot = L_acc * cs.cos(sigma) / safe_V + (V**2 - 1/safe_r) * cs.cos(gamma) / (safe_V*safe_r) + 2*cfg.Omega_e*cs.cos(phi)*cs.sin(psi) + cfg.Omega_e**2 * safe_r * cs.cos(phi)*(cs.cos(gamma)*cs.cos(phi) + cs.sin(gamma)*cs.sin(phi)*cs.cos(psi))/safe_V
    
    psi_dot = L_acc * cs.sin(sigma) / (safe_V*safe_cos_gamma) + V*cs.cos(gamma)*cs.sin(psi)*cs.tan(phi)/safe_r - 2*cfg.Omega_e*(cs.tan(gamma)*cs.cos(psi)*cs.cos(phi) - cs.sin(phi)) + cfg.Omega_e**2*safe_r*cs.sin(phi)*cs.cos(phi)*cs.sin(psi)/(safe_V*safe_cos_gamma)
    
    sigma_dot = u
    tf_dot = 0.0  
    
    rhs = cs.vertcat(r_dot, theta_dot, phi_dot, V_dot, gamma_dot, psi_dot, sigma_dot, tf_dot)
    return cs.Function("f_cont", [states, controls], [rhs, rho, V_dim, L_acc, D_acc])