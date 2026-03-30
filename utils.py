"""
工具函数模块：误差计算与辅助功能
包含误差分析、收敛性分析等核心工具函数
"""

import numpy as np
from typing import Tuple, List, Dict


def compute_absolute_error(numerical: np.ndarray, analytical: np.ndarray) -> np.ndarray:
    """
    计算数值解与解析解的绝对误差
    
    参数:
        numerical: 数值解数组
        analytical: 解析解数组
    
    返回:
        绝对误差数组
    """
    return np.abs(numerical - analytical)


def compute_relative_error(numerical: np.ndarray, analytical: np.ndarray, 
                           eps: float = 1e-10) -> np.ndarray:
    """
    计算数值解与解析解的相对误差
    
    参数:
        numerical: 数值解数组
        analytical: 解析解数组
        eps: 防止除零的小量
    
    返回:
        相对误差数组
    """
    return np.abs(numerical - analytical) / (np.abs(analytical) + eps)


def compute_rmse(numerical: np.ndarray, analytical: np.ndarray) -> float:
    """
    计算均方根误差(RMSE)
    
    参数:
        numerical: 数值解数组
        analytical: 解析解数组
    
    返回:
        均方根误差值
    """
    return np.sqrt(np.mean((numerical - analytical) ** 2))


def compute_max_error(numerical: np.ndarray, analytical: np.ndarray) -> float:
    """
    计算最大绝对误差(L∞范数误差)
    
    参数:
        numerical: 数值解数组
        analytical: 解析解数组
    
    返回:
        最大绝对误差值
    """
    return np.max(np.abs(numerical - analytical))


def compute_l2_error(numerical: np.ndarray, analytical: np.ndarray, dx: float) -> float:
    """
    计算L2范数误差
    
    参数:
        numerical: 数值解数组
        analytical: 解析解数组
        dx: 空间步长
    
    返回:
        L2范数误差值
    """
    return np.sqrt(dx * np.sum((numerical - analytical) ** 2))


def check_cfl_stability(alpha: float, dt: float, dx: float) -> Tuple[bool, float]:
    """
    检验CFL稳定性条件
    FTCS显式格式的稳定性条件: r = alpha * dt / dx^2 <= 0.5
    
    参数:
        alpha: 热扩散系数
        dt: 时间步长
        dx: 空间步长
    
    返回:
        (是否稳定, CFL数r)
    """
    r = alpha * dt / (dx ** 2)
    is_stable = r <= 0.5
    return is_stable, r


def compute_convergence_rate(errors: List[float], resolutions: List[float]) -> List[float]:
    """
    计算收敛阶
    
    参数:
        errors: 不同网格分辨率下的误差列表
        resolutions: 网格分辨率列表(如dx值)
    
    返回:
        收敛阶列表
    """
    rates = []
    for i in range(1, len(errors)):
        if errors[i] > 0 and errors[i-1] > 0:
            rate = np.log(errors[i-1] / errors[i]) / np.log(resolutions[i-1] / resolutions[i])
            rates.append(rate)
        else:
            rates.append(np.nan)
    return rates


def generate_grid(L: float, T: float, nx: int, nt: int) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    生成空间和时间离散网格
    
    参数:
        L: 空间域长度
        T: 总时间
        nx: 空间网格点数(不含边界点)
        nt: 时间步数
    
    返回:
        (x数组, t数组, dx, dt)
    """
    x = np.linspace(0, L, nx + 2)
    t = np.linspace(0, T, nt + 1)
    dx = L / (nx + 1)
    dt = T / nt
    return x, t, dx, dt


def print_simulation_info(alpha: float, L: float, T: float, nx: int, nt: int,
                          dx: float, dt: float, r: float, is_stable: bool):
    """
    打印仿真参数信息
    
    参数:
        alpha: 热扩散系数
        L: 空间域长度
        T: 总时间
        nx: 空间网格点数
        nt: 时间步数
        dx: 空间步长
        dt: 时间步长
        r: CFL数
        is_stable: 是否满足稳定性条件
    """
    print("=" * 60)
    print("一维非稳态热传导方程数值求解 - 仿真参数")
    print("=" * 60)
    print(f"热扩散系数 α = {alpha}")
    print(f"空间域长度 L = {L}")
    print(f"总时间 T = {T}")
    print(f"空间网格点数 nx = {nx}")
    print(f"时间步数 nt = {nt}")
    print(f"空间步长 dx = {dx:.6f}")
    print(f"时间步长 dt = {dt:.6f}")
    print(f"CFL数 r = α·dt/dx² = {r:.6f}")
    print(f"稳定性条件 (r ≤ 0.5): {'满足 ✓' if is_stable else '不满足 ✗'}")
    print("=" * 60)


def print_error_analysis(rmse: float, max_err: float, l2_err: float):
    """
    打印误差分析结果
    
    参数:
        rmse: 均方根误差
        max_err: 最大绝对误差
        l2_err: L2范数误差
    """
    print("\n" + "=" * 60)
    print("误差分析结果")
    print("=" * 60)
    print(f"均方根误差 (RMSE): {rmse:.6e}")
    print(f"最大绝对误差 (L∞): {max_err:.6e}")
    print(f"L2范数误差: {l2_err:.6e}")
    print("=" * 60)


def run_convergence_study(solver_func, analytical_func, alpha: float, L: float, T: float,
                          nx_list: List[int], cfl_ratio: float = 0.4) -> Dict:
    """
    执行收敛性研究
    
    参数:
        solver_func: 数值求解函数
        analytical_func: 解析解函数
        alpha: 热扩散系数
        L: 空间域长度
        T: 总时间
        nx_list: 不同空间网格点数列表
        cfl_ratio: CFL数比例(用于确定时间步长)
    
    返回:
        包含收敛性分析结果的字典
    """
    results = {
        'nx_list': nx_list,
        'dx_list': [],
        'dt_list': [],
        'rmse_list': [],
        'max_err_list': [],
        'l2_err_list': [],
        'convergence_rates': []
    }
    
    for nx in nx_list:
        dx = L / (nx + 1)
        dt = cfl_ratio * dx**2 / alpha
        nt = int(T / dt) + 1
        dt = T / nt
        
        x, t, dx_actual, dt_actual = generate_grid(L, T, nx, nt)
        
        U_numerical = solver_func(alpha, L, T, nx, nt)
        U_analytical = analytical_func(x, T, alpha, L)
        
        numerical_final = U_numerical[-1, 1:-1]
        analytical_final = U_analytical[1:-1]
        
        rmse = compute_rmse(numerical_final, analytical_final)
        max_err = compute_max_error(numerical_final, analytical_final)
        l2_err = compute_l2_error(numerical_final, analytical_final, dx_actual)
        
        results['dx_list'].append(dx_actual)
        results['dt_list'].append(dt_actual)
        results['rmse_list'].append(rmse)
        results['max_err_list'].append(max_err)
        results['l2_err_list'].append(l2_err)
    
    results['convergence_rates'] = compute_convergence_rate(
        results['max_err_list'], results['dx_list']
    )
    
    return results
