"""
一维非稳态热传导方程求解器模块

包含：
1. 空间/时间网格生成
2. FTCS显式有限差分法数值解
3. 解析解计算（基于SciPy特殊函数）
4. CFL稳定条件验证
"""

import numpy as np
from scipy.special import erf

def generate_grid(L, T, nx, nt):
    """
    生成空间和时间离散网格
    
    参数:
        L (float): 空间区间长度
        T (float): 总时间
        nx (int): 空间节点数
        nt (int): 时间步数
        
    返回:
        x (ndarray): 空间网格节点
        t (ndarray): 时间网格节点
        dx (float): 空间步长
        dt (float): 时间步长
    """
    x = np.linspace(0, L, nx)
    t = np.linspace(0, T, nt)
    dx = x[1] - x[0]
    dt = t[1] - t[0]
    return x, t, dx, dt

def check_cfl_stability(alpha, dx, dt):
    """
    验证CFL稳定条件
    
    参数:
        alpha (float): 热扩散系数
        dx (float): 空间步长
        dt (float): 时间步长
        
    返回:
        is_stable (bool): 是否满足稳定条件
        cfl_number (float): CFL数（r = alpha*dt/dx²）
    """
    cfl_number = alpha * dt / (dx ** 2)
    is_stable = cfl_number <= 0.5
    return is_stable, cfl_number

def solve_ftcs(L, T, alpha, nx, nt):
    """
    使用FTCS显式有限差分法求解一维非稳态热传导方程
    
    参数:
        L (float): 空间区间长度
        T (float): 总时间
        alpha (float): 热扩散系数
        nx (int): 空间节点数
        nt (int): 时间步数
        
    返回:
        x (ndarray): 空间网格
        t (ndarray): 时间网格
        u (ndarray): 数值解矩阵 [时间步, 空间节点]
        cfl_info (tuple): CFL稳定信息 (is_stable, cfl_number)
    """
    x, t, dx, dt = generate_grid(L, T, nx, nt)
    
    is_stable, cfl_number = check_cfl_stability(alpha, dx, dt)
    
    u = np.zeros((nt, nx))
    u[0, :] = np.sin(np.pi * x / L)
    
    for n in range(nt - 1):
        for i in range(1, nx - 1):
            u[n + 1, i] = u[n, i] + cfl_number * (u[n, i + 1] - 2 * u[n, i] + u[n, i - 1])
        
        u[n + 1, 0] = 0
        u[n + 1, -1] = 0
    
    return x, t, u, (is_stable, cfl_number)

def solve_analytical(L, T, alpha, nx, nt):
    """
    计算一维非稳态热传导方程的解析解
    
    参数:
        L (float): 空间区间长度
        T (float): 总时间
        alpha (float): 热扩散系数
        nx (int): 空间节点数
        nt (int): 时间步数
        
    返回:
        x (ndarray): 空间网格
        t (ndarray): 时间网格
        u_analytical (ndarray): 解析解矩阵 [时间步, 空间节点]
    """
    x, t, _, _ = generate_grid(L, T, nx, nt)
    
    u_analytical = np.zeros((nt, nx))
    
    for n in range(nt):
        u_analytical[n, :] = np.sin(np.pi * x / L) * np.exp(-alpha * (np.pi ** 2) * t[n] / (L ** 2))
    
    return x, t, u_analytical
