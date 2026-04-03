import numpy as np
import matplotlib.pyplot as plt
import config as cfg

def extract_and_plot(sol, x):
    """提取解数据并绘制结果"""
    r_opt = np.zeros(cfg.K)
    theta_opt = np.zeros(cfg.K)
    phi_opt = np.zeros(cfg.K)
    V_opt = np.zeros(cfg.K)
    gamma_opt = np.zeros(cfg.K)

    for k in range(cfg.K):
        r_opt[k] = sol.value(x[k][0])
        theta_opt[k] = sol.value(x[k][1])
        phi_opt[k] = sol.value(x[k][2])
        V_opt[k] = sol.value(x[k][3])
        gamma_opt[k] = sol.value(x[k][4])

    alt_km = (r_opt - 1.0) * cfg.R0 / 1000.0  
    lon_deg = theta_opt * 180.0 / np.pi   
    lat_deg = phi_opt * 180.0 / np.pi     
    V_mps = V_opt * cfg.V_ref                 

    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(lon_deg, lat_deg, 'b-o', linewidth=2, label='Optimal Trajectory', markersize=4)

    ax.plot(lon_deg[0], lat_deg[0], 'g^', markersize=10, label='Start')
    ax.plot(lon_deg[-1], lat_deg[-1], 'r*', markersize=12, label='Target')

    circle1 = plt.Circle((cfg.theta_NFZ1 * 180/np.pi, cfg.phi_NFZ1 * 180/np.pi), 
                         (cfg.R_NFZ1 * cfg.R0) / (np.pi * cfg.R0 / 180), 
                         color='red', fill=False, linewidth=2, linestyle='--', label='NFZ 1')
    ax.add_patch(circle1)

    circle2 = plt.Circle((cfg.theta_NFZ2 * 180/np.pi, cfg.phi_NFZ2 * 180/np.pi), 
                         (cfg.R_NFZ2 * cfg.R0) / (np.pi * cfg.R0 / 180), 
                         color='orange', fill=False, linewidth=2, linestyle='--', label='NFZ 2')
    ax.add_patch(circle2)

    ax.set_xlabel('Longitude [deg]')
    ax.set_ylabel('Latitude [deg]')
    ax.set_title('Ground Track with No-Fly Zones')
    ax.legend()
    ax.axis('equal') 
    ax.grid(True)

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.plot(V_mps, alt_km, 'm-s', linewidth=2, markersize=4)
    ax2.set_xlabel('Velocity [m/s]')
    ax2.set_ylabel('Altitude [km]')
    ax2.set_title('Altitude vs Velocity Profile')
    ax2.invert_xaxis() 
    ax2.grid(True)

    plt.show()