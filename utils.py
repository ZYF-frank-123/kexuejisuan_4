"""
工具函数模块

包含：
1. 误差计算（绝对误差、均方根误差）
2. 收敛性分析
"""

import numpy as np

def compute_absolute_error(u_numerical, u_analytical):
    """
    计算绝对误差
    
    参数:
        u_numerical (ndarray): 数值解矩阵
        u_analytical (ndarray): 解析解矩阵
        
    返回:
        abs_error (ndarray): 绝对误差矩阵
    """
    return np.abs(u_numerical - u_analytical)

def compute_rmse(u_numerical, u_analytical):
    """
    计算均方根误差(RMSE)
    
    参数:
        u_numerical (ndarray): 数值解矩阵
        u_analytical (ndarray): 解析解矩阵
        
    返回:
        rmse (float): 全局均方根误差
        rmse_time (ndarray): 各时间步的RMSE
    """
    nt, nx = u_numerical.shape
    rmse_time = np.zeros(nt)
    
    for n in range(nt):
        rmse_time[n] = np.sqrt(np.mean((u_numerical[n, :] - u_analytical[n, :]) ** 2))
    
    rmse_global = np.sqrt(np.mean((u_numerical - u_analytical) ** 2))
    
    return rmse_global, rmse_time

def compute_convergence_order(dx_list, error_list):
    """
    计算收敛阶数
    
    参数:
        dx_list (list): 不同空间步长列表
        error_list (list): 对应步长的误差列表
        
    返回:
        convergence_order (float): 收敛阶数
    """
    if len(dx_list) < 2 or len(error_list) < 2:
        raise ValueError("需要至少两组数据计算收敛阶")
    
    log_dx = np.log(dx_list)
    log_error = np.log(error_list)
    
    slope, _ = np.polyfit(log_dx, log_error, 1)
    convergence_order = -slope if slope < 0 else slope
    
    return convergence_order

def convergence_analysis(L, T, alpha, nx_list):
    """
    收敛性分析（确保CFL稳定条件）
    
    参数:
        L (float): 空间区间长度
        T (float): 总时间
        alpha (float): 热扩散系数
        nx_list (list): 空间节点数列表
        
    返回:
        dx_list (list): 空间步长列表
        error_list (list): 对应步长的RMSE列表
        convergence_order (float): 收敛阶数
    """
    from solver import solve_ftcs, solve_analytical, check_cfl_stability
    
    dx_list = []
    error_list = []
    
    for nx in nx_list:
        dx = L / (nx - 1)
        dx_list.append(dx)
        
        dt = 0.4 * (dx ** 2) / alpha
        nt = int(T / dt) + 1
        
        _, _, u_num, _ = solve_ftcs(L, T, alpha, nx, nt)
        _, _, u_ana = solve_analytical(L, T, alpha, nx, nt)
        
        rmse, _ = compute_rmse(u_num, u_ana)
        error_list.append(rmse)
    
    convergence_order = compute_convergence_order(dx_list, error_list)
    
    return dx_list, error_list, convergence_order
