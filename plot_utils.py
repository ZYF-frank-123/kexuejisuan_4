"""
可视化模块：包含各种图表绘制功能
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from typing import Optional, List, Dict
import os

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def setup_figure(figsize: tuple = (10, 6), dpi: int = 150) -> tuple:
    """
    创建图形和坐标轴

    Parameters
    ----------
    figsize : tuple
        图形尺寸
    dpi : int
        分辨率

    Returns
    -------
    fig, ax : matplotlib图形对象
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    return fig, ax


def save_figure(fig: plt.Figure, filename: str, output_dir: str = './figures') -> None:
    """
    保存图形到文件

    Parameters
    ----------
    fig : plt.Figure
        图形对象
    filename : str
        文件名
    output_dir : str
        输出目录
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"图表已保存: {filepath}")


def plot_temperature_distribution(
    x: np.ndarray,
    u: np.ndarray,
    t_selected: List[float],
    t_array: np.ndarray,
    L: float,
    title: str = "温度空间分布",
    output_dir: str = './figures',
    filename: str = 'temperature_distribution.png'
) -> None:
    """
    绘制不同时刻的温度空间分布曲线

    Parameters
    ----------
    x : np.ndarray
        空间网格点
    u : np.ndarray
        温度分布，形状为 (nt, nx)
    t_selected : List[float]
        要绘制的时刻列表
    t_array : np.ndarray
        完整时间数组
    L : float
        区间长度
    title : str
        图表标题
    output_dir : str
        输出目录
    filename : str
        输出文件名
    """
    fig, ax = setup_figure(figsize=(10, 6))

    colors = cm.viridis(np.linspace(0, 1, len(t_selected)))

    for i, t_val in enumerate(t_selected):
        # 找到最接近的时间索引
        idx = np.argmin(np.abs(t_array - t_val))
        ax.plot(x, u[idx, :], color=colors[i], linewidth=2,
                label=f't = {t_array[idx]:.3f}')

    ax.set_xlabel('位置 x', fontsize=12)
    ax.set_ylabel('温度 u', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, L)

    plt.tight_layout()
    save_figure(fig, filename, output_dir)
    plt.show()


def plot_temperature_heatmap(
    x: np.ndarray,
    t: np.ndarray,
    u: np.ndarray,
    L: float,
    T: float,
    title: str = "温度时空分布热力图",
    output_dir: str = './figures',
    filename: str = 'temperature_heatmap.png'
) -> None:
    """
    绘制温度时空分布热力图

    Parameters
    ----------
    x : np.ndarray
        空间网格点
    t : np.ndarray
        时间网格点
    u : np.ndarray
        温度分布，形状为 (nt, nx)
    L : float
        区间长度
    T : float
        总时间
    title : str
        图表标题
    output_dir : str
        输出目录
    filename : str
        输出文件名
    """
    fig, ax = setup_figure(figsize=(12, 6))

    # 创建热力图
    im = ax.contourf(x, t, u, levels=50, cmap='hot')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('温度 u', fontsize=12)

    ax.set_xlabel('位置 x', fontsize=12)
    ax.set_ylabel('时间 t', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xlim(0, L)
    ax.set_ylim(0, T)

    plt.tight_layout()
    save_figure(fig, filename, output_dir)
    plt.show()


def plot_3d_surface(
    x: np.ndarray,
    t: np.ndarray,
    u: np.ndarray,
    L: float,
    T: float,
    title: str = "温度时空分布3D图",
    output_dir: str = './figures',
    filename: str = 'temperature_3d.png'
) -> None:
    """
    绘制温度分布3D表面图

    Parameters
    ----------
    x : np.ndarray
        空间网格点
    t : np.ndarray
        时间网格点
    u : np.ndarray
        温度分布，形状为 (nt, nx)
    L : float
        区间长度
    T : float
        总时间
    title : str
        图表标题
    output_dir : str
        输出目录
    filename : str
        输出文件名
    """
    fig = plt.figure(figsize=(12, 8), dpi=150)
    ax = fig.add_subplot(111, projection='3d')

    # 创建网格
    X, T_grid = np.meshgrid(x, t)

    # 绘制3D表面
    surf = ax.plot_surface(X, T_grid, u, cmap='hot', alpha=0.9,
                           linewidth=0, antialiased=True)

    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
    cbar.set_label('温度 u', fontsize=12)

    ax.set_xlabel('位置 x', fontsize=12)
    ax.set_ylabel('时间 t', fontsize=12)
    ax.set_zlabel('温度 u', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xlim(0, L)
    ax.set_ylim(0, T)

    plt.tight_layout()
    save_figure(fig, filename, output_dir)
    plt.show()


def plot_error_comparison(
    x: np.ndarray,
    u_numerical: np.ndarray,
    u_analytical: np.ndarray,
    t_selected: List[float],
    t_array: np.ndarray,
    L: float,
    output_dir: str = './figures',
    filename: str = 'error_comparison.png'
) -> None:
    """
    绘制数值解与解析解对比及误差分析

    Parameters
    ----------
    x : np.ndarray
        空间网格点
    u_numerical : np.ndarray
        数值解，形状为 (nt, nx)
    u_analytical : np.ndarray
        解析解，形状为 (nt, nx)
    t_selected : List[float]
        要绘制的时刻列表
    t_array : np.ndarray
        完整时间数组
    L : float
        区间长度
    output_dir : str
        输出目录
    filename : str
        输出文件名
    """
    fig, axes = plt.subplots(2, len(t_selected), figsize=(5*len(t_selected), 8), dpi=150)

    if len(t_selected) == 1:
        axes = axes.reshape(2, 1)

    for i, t_val in enumerate(t_selected):
        idx = np.argmin(np.abs(t_array - t_val))

        # 上排：解的对比
        axes[0, i].plot(x, u_numerical[idx, :], 'b-', linewidth=2, label='数值解')
        axes[0, i].plot(x, u_analytical[idx, :], 'r--', linewidth=2, label='解析解')
        axes[0, i].set_xlabel('位置 x', fontsize=10)
        axes[0, i].set_ylabel('温度 u', fontsize=10)
        axes[0, i].set_title(f't = {t_array[idx]:.3f}', fontsize=11)
        axes[0, i].legend(fontsize=9)
        axes[0, i].grid(True, alpha=0.3)
        axes[0, i].set_xlim(0, L)

        # 下排：绝对误差
        error = np.abs(u_numerical[idx, :] - u_analytical[idx, :])
        axes[1, i].plot(x, error, 'g-', linewidth=2)
        axes[1, i].set_xlabel('位置 x', fontsize=10)
        axes[1, i].set_ylabel('绝对误差', fontsize=10)
        axes[1, i].set_title(f'绝对误差 (t = {t_array[idx]:.3f})', fontsize=11)
        axes[1, i].grid(True, alpha=0.3)
        axes[1, i].set_xlim(0, L)

    plt.suptitle('数值解与解析解对比及误差分析', fontsize=14, y=1.02)
    plt.tight_layout()
    save_figure(fig, filename, output_dir)
    plt.show()


def plot_error_evolution(
    t: np.ndarray,
    rmse_values: np.ndarray,
    max_error_values: np.ndarray,
    output_dir: str = './figures',
    filename: str = 'error_evolution.png'
) -> None:
    """
    绘制误差随时间演化曲线

    Parameters
    ----------
    t : np.ndarray
        时间数组
    rmse_values : np.ndarray
        各时间点的RMSE值
    max_error_values : np.ndarray
        各时间点的最大误差值
    output_dir : str
        输出目录
    filename : str
        输出文件名
    """
    fig, ax = setup_figure(figsize=(10, 6))

    ax.semilogy(t, rmse_values, 'b-', linewidth=2, label='RMSE')
    ax.semilogy(t, max_error_values, 'r--', linewidth=2, label='最大绝对误差')

    ax.set_xlabel('时间 t', fontsize=12)
    ax.set_ylabel('误差 (对数尺度)', fontsize=12)
    ax.set_title('误差随时间演化', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    save_figure(fig, filename, output_dir)
    plt.show()


def plot_convergence_analysis(
    dx_values: np.ndarray,
    errors: np.ndarray,
    rates: np.ndarray,
    output_dir: str = './figures',
    filename: str = 'convergence_analysis.png'
) -> None:
    """
    绘制收敛性分析图

    Parameters
    ----------
    dx_values : np.ndarray
        网格尺寸数组
    errors : np.ndarray
        对应误差数组
    rates : np.ndarray
        收敛阶数数组
    output_dir : str
        输出目录
    filename : str
        输出文件名
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)

    # 左图：误差-网格尺寸双对数图
    axes[0].loglog(dx_values, errors, 'bo-', linewidth=2, markersize=8, label='数值误差')

    # 添加参考线（一阶和二阶收敛）
    dx_ref = np.linspace(dx_values[0], dx_values[-1], 100)
    axes[0].loglog(dx_ref, errors[0] * (dx_ref / dx_values[0])**1, 'k--',
                   linewidth=1.5, alpha=0.7, label='一阶收敛 (O(Δx))')
    axes[0].loglog(dx_ref, errors[0] * (dx_ref / dx_values[0])**2, 'k:',
                   linewidth=1.5, alpha=0.7, label='二阶收敛 (O(Δx²))')

    axes[0].set_xlabel('网格尺寸 Δx', fontsize=12)
    axes[0].set_ylabel('RMSE误差', fontsize=12)
    axes[0].set_title('收敛性分析 (双对数坐标)', fontsize=13)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3, which='both')

    # 右图：收敛阶数柱状图
    x_pos = np.arange(len(rates))
    bars = axes[1].bar(x_pos, rates, color='steelblue', edgecolor='black', alpha=0.8)
    axes[1].axhline(y=2.0, color='r', linestyle='--', linewidth=2, label='理论二阶收敛')
    axes[1].axhline(y=np.mean(rates), color='g', linestyle=':', linewidth=2,
                    label=f'平均收敛阶数: {np.mean(rates):.3f}')

    # 在柱子上标注数值
    for i, (bar, rate) in enumerate(zip(bars, rates)):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{rate:.3f}', ha='center', va='bottom', fontsize=10)

    axes[1].set_xlabel('网格细化步骤', fontsize=12)
    axes[1].set_ylabel('收敛阶数', fontsize=12)
    axes[1].set_title('各步骤收敛阶数', fontsize=13)
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels([f'{i+1}' for i in range(len(rates))])
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    save_figure(fig, filename, output_dir)
    plt.show()


def plot_stability_analysis(
    x: np.ndarray,
    stability_results: Dict,
    t_display: float,
    output_dir: str = './figures',
    filename: str = 'stability_analysis.png'
) -> None:
    """
    绘制数值稳定性分析图

    Parameters
    ----------
    x : np.ndarray
        空间网格点
    stability_results : Dict
        不同时间步长的求解结果字典
    t_display : float
        要显示的时间点
    output_dir : str
        输出目录
    filename : str
        输出文件名
    """
    n_cases = len(stability_results)
    fig, axes = plt.subplots(2, (n_cases + 1) // 2, figsize=(6*((n_cases + 1) // 2), 10), dpi=150)

    if n_cases == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, (key, result) in enumerate(stability_results.items()):
        u = result['u']
        t = result['t']
        r = result['r']
        is_stable = result['is_stable']
        dt = result['dt']

        # 找到最接近显示时间的索引
        t_idx = np.argmin(np.abs(t - t_display))

        # 绘制该时刻的温度分布
        axes[idx].plot(x, u[t_idx, :], 'b-', linewidth=2)
        axes[idx].set_xlabel('位置 x', fontsize=10)
        axes[idx].set_ylabel('温度 u', fontsize=10)

        status = "稳定 ✓" if is_stable else "不稳定 ✗"
        axes[idx].set_title(f'dt={dt:.2e}, r={r:.3f}\n{status}', fontsize=11)
        axes[idx].grid(True, alpha=0.3)
        axes[idx].set_xlim(0, x[-1])

        # 如果不稳定，调整y轴以显示振荡
        if not is_stable:
            u_min, u_max = np.min(u[t_idx, :]), np.max(u[t_idx, :])
            margin = (u_max - u_min) * 0.1 if u_max != u_min else 0.1
            axes[idx].set_ylim(u_min - margin, u_max + margin)

    # 隐藏多余的子图
    for idx in range(n_cases, len(axes)):
        axes[idx].axis('off')

    plt.suptitle(f'数值稳定性分析 (t = {t_display:.3f})', fontsize=14, y=1.02)
    plt.tight_layout()
    save_figure(fig, filename, output_dir)
    plt.show()


def plot_instability_evolution(
    x: np.ndarray,
    t: np.ndarray,
    u_unstable: np.ndarray,
    t_selected: List[float],
    output_dir: str = './figures',
    filename: str = 'instability_evolution.png'
) -> None:
    """
    绘制不稳定解的时间演化过程

    Parameters
    ----------
    x : np.ndarray
        空间网格点
    t : np.ndarray
        时间数组
    u_unstable : np.ndarray
        不稳定的数值解
    t_selected : List[float]
        要绘制的时刻列表
    output_dir : str
        输出目录
    filename : str
        输出文件名
    """
    fig, ax = setup_figure(figsize=(12, 6))

    colors = cm.coolwarm(np.linspace(0, 1, len(t_selected)))

    for i, t_val in enumerate(t_selected):
        idx = np.argmin(np.abs(t - t_val))
        ax.plot(x, u_unstable[idx, :], color=colors[i], linewidth=2,
                label=f't = {t[idx]:.4f}')

    ax.set_xlabel('位置 x', fontsize=12)
    ax.set_ylabel('温度 u', fontsize=12)
    ax.set_title('数值不稳定现象演示 (CFL条件不满足)', fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, x[-1])

    plt.tight_layout()
    save_figure(fig, filename, output_dir)
    plt.show()


def plot_cfl_diagram(
    alpha: float,
    dx: float,
    dt_stable: float,
    dt_unstable: float,
    output_dir: str = './figures',
    filename: str = 'cfl_diagram.png'
) -> None:
    """
    绘制CFL条件示意图

    Parameters
    ----------
    alpha : float
        热扩散系数
    dx : float
        空间步长
    dt_stable : float
        稳定的时间步长
    dt_unstable : float
        不稳定的时间步长
    output_dir : str
        输出目录
    filename : str
        输出文件名
    """
    fig, ax = setup_figure(figsize=(10, 6))

    # 计算CFL数
    r_stable = alpha * dt_stable / (dx ** 2)
    r_unstable = alpha * dt_unstable / (dx ** 2)

    # 绘制稳定性区域
    dt_range = np.linspace(0, dt_unstable * 1.2, 100)
    r_range = alpha * dt_range / (dx ** 2)

    ax.fill_between(dt_range, 0, r_range, where=(r_range <= 0.5),
                    alpha=0.3, color='green', label='稳定区域 (r ≤ 0.5)')
    ax.fill_between(dt_range, 0, r_range, where=(r_range > 0.5),
                    alpha=0.3, color='red', label='不稳定区域 (r > 0.5)')

    # 绘制稳定和不稳定点
    ax.scatter([dt_stable], [r_stable], color='green', s=200, marker='o',
               edgecolors='black', linewidth=2, zorder=5, label=f'稳定点 (r={r_stable:.3f})')
    ax.scatter([dt_unstable], [r_unstable], color='red', s=200, marker='X',
               edgecolors='black', linewidth=2, zorder=5, label=f'不稳定点 (r={r_unstable:.3f})')

    # 绘制临界线
    ax.axhline(y=0.5, color='black', linestyle='--', linewidth=2, label='临界线 r=0.5')

    ax.set_xlabel('时间步长 Δt', fontsize=12)
    ax.set_ylabel('CFL数 r', fontsize=12)
    ax.set_title('CFL稳定性条件示意图', fontsize=14)
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, dt_unstable * 1.2)
    ax.set_ylim(0, r_unstable * 1.1)

    # 添加文本说明
    ax.text(dt_unstable * 0.5, 0.25, '稳定区域\nr ≤ 0.5',
            fontsize=12, ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    ax.text(dt_unstable * 0.9, r_unstable * 0.8, '不稳定区域\nr > 0.5',
            fontsize=12, ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))

    plt.tight_layout()
    save_figure(fig, filename, output_dir)
    plt.show()


def create_summary_figure(
    x: np.ndarray,
    t: np.ndarray,
    u_numerical: np.ndarray,
    u_analytical: np.ndarray,
    L: float,
    T: float,
    output_dir: str = './figures',
    filename: str = 'summary.png'
) -> None:
    """
    创建综合汇总图

    Parameters
    ----------
    x : np.ndarray
        空间网格点
    t : np.ndarray
        时间网格点
    u_numerical : np.ndarray
        数值解
    u_analytical : np.ndarray
        解析解
    L : float
        区间长度
    T : float
        总时间
    output_dir : str
        输出目录
    filename : str
        输出文件名
    """
    fig = plt.figure(figsize=(16, 12), dpi=150)

    # 创建子图布局
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. 温度空间分布（不同时刻）
    ax1 = fig.add_subplot(gs[0, :2])
    t_selected = [0, T*0.25, T*0.5, T*0.75, T]
    colors = cm.viridis(np.linspace(0, 1, len(t_selected)))
    for i, t_val in enumerate(t_selected):
        idx = np.argmin(np.abs(t - t_val))
        ax1.plot(x, u_numerical[idx, :], color=colors[i], linewidth=2,
                label=f't = {t[idx]:.3f}')
    ax1.set_xlabel('位置 x', fontsize=10)
    ax1.set_ylabel('温度 u', fontsize=10)
    ax1.set_title('温度空间分布演化', fontsize=11)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, L)

    # 2. 时空热力图
    ax2 = fig.add_subplot(gs[0, 2])
    im = ax2.contourf(x, t, u_numerical, levels=30, cmap='hot')
    plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    ax2.set_xlabel('位置 x', fontsize=10)
    ax2.set_ylabel('时间 t', fontsize=10)
    ax2.set_title('时空热力图', fontsize=11)

    # 3. 数值解与解析解对比
    ax3 = fig.add_subplot(gs[1, 0])
    idx_mid = len(t) // 2
    ax3.plot(x, u_numerical[idx_mid, :], 'b-', linewidth=2, label='数值解')
    ax3.plot(x, u_analytical[idx_mid, :], 'r--', linewidth=2, label='解析解')
    ax3.set_xlabel('位置 x', fontsize=10)
    ax3.set_ylabel('温度 u', fontsize=10)
    ax3.set_title(f'解对比 (t={t[idx_mid]:.3f})', fontsize=11)
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # 4. 误差分布
    ax4 = fig.add_subplot(gs[1, 1])
    error = np.abs(u_numerical[idx_mid, :] - u_analytical[idx_mid, :])
    ax4.plot(x, error, 'g-', linewidth=2)
    ax4.set_xlabel('位置 x', fontsize=10)
    ax4.set_ylabel('绝对误差', fontsize=10)
    ax4.set_title('空间误差分布', fontsize=11)
    ax4.grid(True, alpha=0.3)

    # 5. 中心点温度演化
    ax5 = fig.add_subplot(gs[1, 2])
    x_center_idx = len(x) // 2
    ax5.plot(t, u_numerical[:, x_center_idx], 'b-', linewidth=2, label='数值解')
    ax5.plot(t, u_analytical[:, x_center_idx], 'r--', linewidth=2, label='解析解')
    ax5.set_xlabel('时间 t', fontsize=10)
    ax5.set_ylabel('温度 u', fontsize=10)
    ax5.set_title(f'中心点温度演化 (x={x[x_center_idx]:.3f})', fontsize=11)
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)

    # 6. 误差统计
    ax6 = fig.add_subplot(gs[2, :])
    rmse_time = np.sqrt(np.mean((u_numerical - u_analytical)**2, axis=1))
    max_error_time = np.max(np.abs(u_numerical - u_analytical), axis=1)
    ax6.semilogy(t, rmse_time, 'b-', linewidth=2, label='RMSE')
    ax6.semilogy(t, max_error_time, 'r--', linewidth=2, label='最大误差')
    ax6.set_xlabel('时间 t', fontsize=10)
    ax6.set_ylabel('误差 (对数)', fontsize=10)
    ax6.set_title('误差随时间演化', fontsize=11)
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.3, which='both')

    plt.suptitle('一维非稳态热传导方程数值求解结果汇总', fontsize=14, y=0.995)
    save_figure(fig, filename, output_dir)
    plt.show()
