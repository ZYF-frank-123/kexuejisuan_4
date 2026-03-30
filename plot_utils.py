"""
可视化模块：生成各类分析图表
包含温度分布、误差曲线、收敛曲线、不稳定现象等可视化功能
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from typing import Optional, List, Dict, Tuple
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 150


def plot_temperature_distribution(x: np.ndarray, t: np.ndarray, U: np.ndarray,
                                   title: str = "温度时空分布",
                                   save_path: Optional[str] = None):
    """
    绘制温度时空分布三维曲面图
    
    参数:
        x: 空间坐标数组
        t: 时间坐标数组
        U: 温度解矩阵
        title: 图表标题
        save_path: 保存路径，None则不保存
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    X, T_mesh = np.meshgrid(x, t)
    
    surf = ax.plot_surface(X, T_mesh, U, cmap=cm.hot, linewidth=0,
                           antialiased=True, alpha=0.9)
    
    ax.set_xlabel('空间位置 x', fontsize=12)
    ax.set_ylabel('时间 t', fontsize=12)
    ax.set_zlabel('温度 u', fontsize=12)
    ax.set_title(title, fontsize=14)
    
    fig.colorbar(surf, shrink=0.5, aspect=10, label='温度值')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"图表已保存: {save_path}")
    
    plt.close()


def plot_contour(x: np.ndarray, t: np.ndarray, U: np.ndarray,
                 title: str = "温度等值线图",
                 save_path: Optional[str] = None):
    """
    绘制温度等值线图
    
    参数:
        x: 空间坐标数组
        t: 时间坐标数组
        U: 温度解矩阵
        title: 图表标题
        save_path: 保存路径
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    X, T_mesh = np.meshgrid(x, t)
    
    levels = np.linspace(np.min(U), np.max(U), 20)
    contour = ax.contourf(X, T_mesh, U, levels=levels, cmap='hot')
    
    ax.set_xlabel('空间位置 x', fontsize=12)
    ax.set_ylabel('时间 t', fontsize=12)
    ax.set_title(title, fontsize=14)
    
    fig.colorbar(contour, label='温度值')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"图表已保存: {save_path}")
    
    plt.close()


def plot_comparison(x: np.ndarray, numerical: np.ndarray, analytical: np.ndarray,
                    time_label: str = "t = T",
                    title: str = "数值解与解析解对比",
                    save_path: Optional[str] = None):
    """
    绘制数值解与解析解对比图
    
    参数:
        x: 空间坐标数组
        numerical: 数值解数组
        analytical: 解析解数组
        time_label: 时间标签
        title: 图表标题
        save_path: 保存路径
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(x, analytical, 'b-', linewidth=2, label='解析解', marker='o', 
            markersize=4, markevery=5)
    ax.plot(x, numerical, 'r--', linewidth=2, label='数值解', marker='s',
            markersize=4, markevery=5)
    
    ax.set_xlabel('空间位置 x', fontsize=12)
    ax.set_ylabel('温度 u', fontsize=12)
    ax.set_title(f'{title} ({time_label})', fontsize=14)
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"图表已保存: {save_path}")
    
    plt.close()


def plot_error_distribution(x: np.ndarray, error: np.ndarray,
                            title: str = "误差分布",
                            save_path: Optional[str] = None):
    """
    绘制误差分布图
    
    参数:
        x: 空间坐标数组
        error: 误差数组
        title: 图表标题
        save_path: 保存路径
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(x, error, 'g-', linewidth=2, marker='o', markersize=4)
    ax.fill_between(x, 0, error, alpha=0.3, color='green')
    
    ax.set_xlabel('空间位置 x', fontsize=12)
    ax.set_ylabel('绝对误差', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    
    max_err = np.max(error)
    ax.axhline(y=max_err, color='r', linestyle='--', label=f'最大误差: {max_err:.2e}')
    ax.legend(loc='best', fontsize=11)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"图表已保存: {save_path}")
    
    plt.close()


def plot_error_evolution(t: np.ndarray, errors: np.ndarray,
                         title: str = "误差随时间演化",
                         save_path: Optional[str] = None):
    """
    绘制误差随时间演化曲线
    
    参数:
        t: 时间坐标数组
        errors: 各时刻误差数组
        title: 图表标题
        save_path: 保存路径
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.semilogy(t, errors, 'b-', linewidth=2, marker='o', markersize=3)
    
    ax.set_xlabel('时间 t', fontsize=12)
    ax.set_ylabel('均方根误差 (对数)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"图表已保存: {save_path}")
    
    plt.close()


def plot_convergence(dx_list: List[float], errors: List[float],
                     title: str = "收敛性分析",
                     save_path: Optional[str] = None):
    """
    绘制收敛性曲线（对数坐标）
    
    参数:
        dx_list: 空间步长列表
        errors: 误差列表
        title: 图表标题
        save_path: 保存路径
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.loglog(dx_list, errors, 'bo-', linewidth=2, markersize=8, label='数值误差')
    
    dx_array = np.array(dx_list)
    ref_line = errors[0] * (dx_array / dx_array[0])
    ax.loglog(dx_list, ref_line, 'r--', linewidth=1.5, label='一阶收敛参考线')
    
    ref_line2 = errors[0] * (dx_array / dx_array[0])**2
    ax.loglog(dx_list, ref_line2, 'g--', linewidth=1.5, label='二阶收敛参考线')
    
    ax.set_xlabel('空间步长 dx (对数)', fontsize=12)
    ax.set_ylabel('最大误差 (对数)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"图表已保存: {save_path}")
    
    plt.close()


def plot_stability_comparison(x: np.ndarray, U_stable: np.ndarray, 
                              U_unstable: np.ndarray, t_stable: float, t_unstable: float,
                              save_path: Optional[str] = None):
    """
    绘制稳定与不稳定情况对比图
    
    参数:
        x: 空间坐标数组
        U_stable: 稳定情况的解
        U_unstable: 不稳定情况的解
        t_stable: 稳定情况的时间
        t_unstable: 不稳定情况的时间
        save_path: 保存路径
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1 = axes[0]
    ax1.plot(x, U_stable, 'b-', linewidth=2, marker='o', markersize=4)
    ax1.set_xlabel('空间位置 x', fontsize=12)
    ax1.set_ylabel('温度 u', fontsize=12)
    ax1.set_title(f'稳定情况 (r ≤ 0.5)\nt = {t_stable:.3f}', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.1, 1.1)
    
    ax2 = axes[1]
    ax2.plot(x, U_unstable, 'r-', linewidth=2, marker='s', markersize=4)
    ax2.set_xlabel('空间位置 x', fontsize=12)
    ax2.set_ylabel('温度 u', fontsize=12)
    ax2.set_title(f'不稳定情况 (r > 0.5)\nt = {t_unstable:.3f}', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    y_range = np.max(np.abs(U_unstable))
    if y_range > 10:
        ax2.set_ylim(-y_range*1.1, y_range*1.1)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"图表已保存: {save_path}")
    
    plt.close()


def plot_time_snapshots(x: np.ndarray, t: np.ndarray, U: np.ndarray,
                        snapshot_indices: Optional[List[int]] = None,
                        title: str = "不同时刻温度分布",
                        save_path: Optional[str] = None):
    """
    绘制多个时刻的温度分布快照
    
    参数:
        x: 空间坐标数组
        t: 时间坐标数组
        U: 温度解矩阵
        snapshot_indices: 时间索引列表
        title: 图表标题
        save_path: 保存路径
    """
    if snapshot_indices is None:
        nt = len(t) - 1
        snapshot_indices = [0, nt//4, nt//2, 3*nt//4, nt]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(snapshot_indices)))
    
    for idx, color in zip(snapshot_indices, colors):
        ax.plot(x, U[idx, :], '-', linewidth=2, color=color,
                label=f't = {t[idx]:.3f}')
    
    ax.set_xlabel('空间位置 x', fontsize=12)
    ax.set_ylabel('温度 u', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"图表已保存: {save_path}")
    
    plt.close()


def plot_instability_evolution(x: np.ndarray, t: np.ndarray, U: np.ndarray,
                               num_snapshots: int = 5,
                               title: str = "数值不稳定现象演化",
                               save_path: Optional[str] = None):
    """
    绘制数值不稳定现象的演化过程
    
    参数:
        x: 空间坐标数组
        t: 时间坐标数组
        U: 温度解矩阵
        num_snapshots: 快照数量
        title: 图表标题
        save_path: 保存路径
    """
    nt = len(t) - 1
    indices = np.linspace(0, nt, num_snapshots, dtype=int)
    
    fig, axes = plt.subplots(1, num_snapshots, figsize=(16, 4))
    
    if num_snapshots == 1:
        axes = [axes]
    
    for i, (idx, ax) in enumerate(zip(indices, axes)):
        ax.plot(x, U[idx, :], 'r-', linewidth=1.5)
        ax.set_xlabel('x', fontsize=10)
        ax.set_ylabel('u', fontsize=10)
        ax.set_title(f't = {t[idx]:.4f}', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        y_max = np.max(np.abs(U[idx, :]))
        if y_max > 1:
            ax.set_ylim(-y_max*1.2, y_max*1.2)
    
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"图表已保存: {save_path}")
    
    plt.close()


def create_output_directory(base_dir: str = "output") -> str:
    """
    创建输出目录
    
    参数:
        base_dir: 基础目录名
    
    返回:
        输出目录的完整路径
    """
    output_dir = os.path.join(os.getcwd(), base_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")
    return output_dir


def generate_all_plots(solver, output_dir: str):
    """
    生成所有分析图表
    
    参数:
        solver: HeatEquationSolver实例
        output_dir: 输出目录路径
    """
    x = solver.x
    t = solver.t
    U_num = solver.U_numerical
    U_ana = solver.U_analytical
    
    plot_temperature_distribution(x, t, U_num, "数值解温度时空分布",
                                   os.path.join(output_dir, "temperature_3d.png"))
    
    plot_contour(x, t, U_num, "数值解温度等值线图",
                 os.path.join(output_dir, "contour.png"))
    
    U_num_final = U_num[-1, :]
    U_ana_final = U_ana[-1, :]
    plot_comparison(x, U_num_final, U_ana_final, "t = T",
                    "数值解与解析解对比",
                    os.path.join(output_dir, "comparison.png"))
    
    error = np.abs(U_num_final - U_ana_final)
    plot_error_distribution(x, error, "最终时刻误差分布",
                            os.path.join(output_dir, "error_distribution.png"))
    
    errors_time = [np.sqrt(np.mean((U_num[n, :] - U_ana[n, :])**2)) 
                   for n in range(len(t))]
    plot_error_evolution(t, errors_time, "误差随时间演化",
                         os.path.join(output_dir, "error_evolution.png"))
    
    plot_time_snapshots(x, t, U_num, None, "数值解不同时刻温度分布",
                        os.path.join(output_dir, "time_snapshots.png"))
    
    print(f"\n所有图表已生成并保存至: {output_dir}")
