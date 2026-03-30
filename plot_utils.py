"""
可视化模块

包含：
1. 温度时空分布（3D曲面图和等高线图）
2. 误差曲线
3. 收敛曲线
4. 不稳定现象展示
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def plot_temperature_distribution(x, t, u, title_prefix="数值解", save_path=None):
    """
    绘制温度时空分布（3D曲面图和等高线图）
    
    参数:
        x (ndarray): 空间网格
        t (ndarray): 时间网格
        u (ndarray): 温度场矩阵
        title_prefix (str): 标题前缀
        save_path (str): 保存路径（可选）
    """
    X, T = np.meshgrid(x, t)
    
    fig = plt.figure(figsize=(15, 6))
    
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    surf = ax1.plot_surface(X, T, u, cmap=cm.viridis, linewidth=0, antialiased=True)
    ax1.set_xlabel('空间坐标 x')
    ax1.set_ylabel('时间 t')
    ax1.set_zlabel('温度 u(x,t)')
    ax1.set_title(f'{title_prefix} - 3D温度分布')
    fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=5)
    
    ax2 = fig.add_subplot(1, 2, 2)
    contour = ax2.contourf(X, T, u, cmap=cm.viridis, levels=20)
    ax2.set_xlabel('空间坐标 x')
    ax2.set_ylabel('时间 t')
    ax2.set_title(f'{title_prefix} - 等高线图')
    fig.colorbar(contour, ax=ax2)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_error_comparison(x, t, abs_error, rmse_time, save_path=None):
    """
    绘制误差对比图
    
    参数:
        x (ndarray): 空间网格
        t (ndarray): 时间网格
        abs_error (ndarray): 绝对误差矩阵
        rmse_time (ndarray): 各时间步的RMSE
        save_path (str): 保存路径（可选）
    """
    X, T = np.meshgrid(x, t)
    
    fig = plt.figure(figsize=(15, 6))
    
    ax1 = fig.add_subplot(1, 2, 1)
    contour = ax1.contourf(X, T, abs_error, cmap=cm.Reds, levels=20)
    ax1.set_xlabel('空间坐标 x')
    ax1.set_ylabel('时间 t')
    ax1.set_title('绝对误差分布')
    fig.colorbar(contour, ax=ax1)
    
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(t, rmse_time, 'b-', linewidth=2)
    ax2.set_xlabel('时间 t')
    ax2.set_ylabel('均方根误差 (RMSE)')
    ax2.set_title('RMSE随时间变化')
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_convergence(dx_list, error_list, convergence_order, save_path=None):
    """
    绘制收敛曲线
    
    参数:
        dx_list (list): 空间步长列表
        error_list (list): 对应步长的RMSE列表
        convergence_order (float): 收敛阶数
        save_path (str): 保存路径（可选）
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    ax1.loglog(dx_list, error_list, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('空间步长 Δx')
    ax1.set_ylabel('均方根误差 (RMSE)')
    ax1.set_title('收敛曲线（双对数坐标）')
    ax1.grid(True, which="both", ls="--")
    
    log_dx = np.log(dx_list)
    log_error = np.log(error_list)
    slope, intercept = np.polyfit(log_dx, log_error, 1)
    
    ax2.scatter(log_dx, log_error, c='b', s=50, label='计算值')
    ax2.plot(log_dx, slope * log_dx + intercept, 'r--', linewidth=2, 
             label=f'拟合直线 (斜率 = {-slope:.3f})')
    ax2.set_xlabel('ln(Δx)')
    ax2.set_ylabel('ln(RMSE)')
    ax2.set_title(f'收敛阶数分析（收敛阶 = {convergence_order:.3f}）')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_stability_comparison(x, t, u_stable, u_unstable, cfl_stable, cfl_unstable, save_path=None):
    """
    绘制稳定与不稳定情况对比图
    
    参数:
        x (ndarray): 空间网格
        t (ndarray): 时间网格
        u_stable (ndarray): 稳定情况下的温度场
        u_unstable (ndarray): 不稳定情况下的温度场
        cfl_stable (float): 稳定情况的CFL数
        cfl_unstable (float): 不稳定情况的CFL数
        save_path (str): 保存路径（可选）
    """
    X, T = np.meshgrid(x, t)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    contour1 = ax1.contourf(X, T, u_stable, cmap=cm.viridis, levels=20)
    ax1.set_xlabel('空间坐标 x')
    ax1.set_ylabel('时间 t')
    ax1.set_title(f'稳定情况 (CFL = {cfl_stable:.3f})')
    fig.colorbar(contour1, ax=ax1)
    
    contour2 = ax2.contourf(X, T, u_unstable, cmap=cm.viridis, levels=20)
    ax2.set_xlabel('空间坐标 x')
    ax2.set_ylabel('时间 t')
    ax2.set_title(f'不稳定情况 (CFL = {cfl_unstable:.3f})')
    fig.colorbar(contour2, ax=ax2)
    
    time_points = [0, len(t)//4, len(t)//2, -1]
    for idx in time_points:
        ax3.plot(x, u_stable[idx, :], label=f't={t[idx]:.3f}')
    ax3.set_xlabel('空间坐标 x')
    ax3.set_ylabel('温度 u(x,t)')
    ax3.set_title(f'稳定情况 - 不同时刻温度分布 (CFL = {cfl_stable:.3f})')
    ax3.legend()
    ax3.grid(True)
    
    for idx in time_points:
        ax4.plot(x, u_unstable[idx, :], label=f't={t[idx]:.3f}')
    ax4.set_xlabel('空间坐标 x')
    ax4.set_ylabel('温度 u(x,t)')
    ax4.set_title(f'不稳定情况 - 不同时刻温度分布 (CFL = {cfl_unstable:.3f})')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_solution_comparison(x, t, u_numerical, u_analytical, time_points=None, save_path=None):
    """
    绘制数值解与解析解的对比图
    
    参数:
        x (ndarray): 空间网格
        t (ndarray): 时间网格
        u_numerical (ndarray): 数值解矩阵
        u_analytical (ndarray): 解析解矩阵
        time_points (list): 要展示的时间点索引列表
        save_path (str): 保存路径（可选）
    """
    if time_points is None:
        time_points = [0, len(t)//4, len(t)//2, len(t)-1]
    
    n_plots = len(time_points)
    fig, axes = plt.subplots(1, n_plots, figsize=(5*n_plots, 5))
    
    for i, idx in enumerate(time_points):
        ax = axes[i] if n_plots > 1 else axes
        ax.plot(x, u_analytical[idx, :], 'r-', linewidth=2, label='解析解')
        ax.plot(x, u_numerical[idx, :], 'bo', markersize=4, label='数值解')
        ax.set_xlabel('空间坐标 x')
        ax.set_ylabel('温度 u(x,t)')
        ax.set_title(f't = {t[idx]:.3f}')
        ax.legend()
        ax.grid(True)
    
    plt.suptitle('数值解与解析解对比', y=1.02, fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
