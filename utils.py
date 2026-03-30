"""
工具函数模块：包含误差计算、网格生成、CFL条件验证等辅助功能
"""

import numpy as np
from typing import Tuple, Dict


def generate_grid(L: float, T: float, nx: int, nt: int) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    生成空间和时间离散网格

    Parameters
    ----------
    L : float
        空间区间长度
    T : float
        总时间
    nx : int
        空间网格点数（包含边界）
    nt : int
        时间步数

    Returns
    -------
    x : np.ndarray
        空间网格点，形状为 (nx,)
    t : np.ndarray
        时间网格点，形状为 (nt,)
    dx : float
        空间步长
    dt : float
        时间步长
    """
    x = np.linspace(0, L, nx)
    t = np.linspace(0, T, nt)
    dx = L / (nx - 1)
    dt = T / (nt - 1)
    return x, t, dx, dt


def check_cfl_stability(alpha: float, dt: float, dx: float) -> Tuple[bool, float]:
    """
    验证CFL稳定条件

    FTCS格式的稳定性条件：r = alpha * dt / dx^2 <= 0.5

    Parameters
    ----------
    alpha : float
        热扩散系数
    dt : float
        时间步长
    dx : float
        空间步长

    Returns
    -------
    is_stable : bool
        是否满足稳定性条件
       r : float
        CFL数（稳定性参数）
    """
    r = alpha * dt / (dx ** 2)
    is_stable = r <= 0.5
    return is_stable, r


def calculate_absolute_error(numerical: np.ndarray, analytical: np.ndarray) -> np.ndarray:
    """
    计算绝对误差

    Parameters
    ----------
    numerical : np.ndarray
        数值解
    analytical : np.ndarray
        解析解

    Returns
    -------
    error : np.ndarray
        绝对误差数组
    """
    return np.abs(numerical - analytical)


def calculate_rmse(numerical: np.ndarray, analytical: np.ndarray) -> float:
    """
    计算均方根误差（Root Mean Square Error）

    Parameters
    ----------
    numerical : np.ndarray
        数值解
    analytical : np.ndarray
        解析解

    Returns
    -------
    rmse : float
        均方根误差
    """
    return np.sqrt(np.mean((numerical - analytical) ** 2))


def calculate_max_error(numerical: np.ndarray, analytical: np.ndarray) -> float:
    """
    计算最大绝对误差

    Parameters
    ----------
    numerical : np.ndarray
        数值解
    analytical : np.ndarray
        解析解

    Returns
    -------
    max_error : float
        最大绝对误差
    """
    return np.max(np.abs(numerical - analytical))


def calculate_convergence_rate(errors: np.ndarray, grid_sizes: np.ndarray) -> np.ndarray:
    """
    计算收敛阶数

    使用公式：order = log(error_i / error_{i+1}) / log(dx_i / dx_{i+1})

    Parameters
    ----------
    errors : np.ndarray
        不同网格尺寸下的误差数组
    grid_sizes : np.ndarray
        对应的网格尺寸（dx）数组

    Returns
    -------
    rates : np.ndarray
        收敛阶数数组
    """
    rates = np.zeros(len(errors) - 1)
    for i in range(len(errors) - 1):
        rates[i] = np.log(errors[i] / errors[i + 1]) / np.log(grid_sizes[i] / grid_sizes[i + 1])
    return rates


def get_stability_threshold_dt(alpha: float, dx: float) -> float:
    """
    获取满足CFL条件的最大时间步长

    Parameters
    ----------
    alpha : float
        热扩散系数
    dx : float
        空间步长

    Returns
    -------
    dt_max : float
        满足稳定性条件的最大时间步长
    """
    return 0.5 * dx ** 2 / alpha


def print_simulation_info(params: Dict) -> None:
    """
    打印仿真参数信息

    Parameters
    ----------
    params : Dict
        包含仿真参数的字典
    """
    print("=" * 60)
    print("一维非稳态热传导方程数值求解")
    print("=" * 60)
    print(f"热扩散系数 (α): {params['alpha']:.6e}")
    print(f"空间区间长度 (L): {params['L']:.4f}")
    print(f"总时间 (T): {params['T']:.4f}")
    print(f"空间网格点数 (nx): {params['nx']}")
    print(f"时间步数 (nt): {params['nt']}")
    print(f"空间步长 (dx): {params['dx']:.6f}")
    print(f"时间步长 (dt): {params['dt']:.6f}")
    print(f"CFL数 (r): {params['r']:.6f}")
    print(f"稳定性条件: {'满足 ✓' if params['is_stable'] else '不满足 ✗'}")
    if not params['is_stable']:
        print(f"建议最大dt: {params['dt_max']:.6e}")
    print("=" * 60)


def print_error_analysis(t: np.ndarray, rmse_values: np.ndarray, max_errors: np.ndarray) -> None:
    """
    打印误差分析结果

    Parameters
    ----------
    t : np.ndarray
        时间点数组
    rmse_values : np.ndarray
        各时间点的RMSE值
    max_errors : np.ndarray
        各时间点的最大误差
    """
    print("\n" + "=" * 60)
    print("误差分析结果")
    print("=" * 60)
    print(f"最终时刻 t={t[-1]:.4f}:")
    print(f"  均方根误差 (RMSE): {rmse_values[-1]:.6e}")
    print(f"  最大绝对误差: {max_errors[-1]:.6e}")
    print(f"  平均RMSE: {np.mean(rmse_values):.6e}")
    print(f"  最大RMSE: {np.max(rmse_values):.6e}")
    print("=" * 60)


def print_convergence_analysis(grid_sizes: np.ndarray, errors: np.ndarray, rates: np.ndarray) -> None:
    """
    打印收敛性分析结果

    Parameters
    ----------
    grid_sizes : np.ndarray
        网格尺寸数组
    errors : np.ndarray
        对应误差数组
    rates : np.ndarray
        收敛阶数数组
    """
    print("\n" + "=" * 60)
    print("收敛性分析结果")
    print("=" * 60)
    print(f"{'网格尺寸 (dx)':<15} {'误差':<20} {'收敛阶数':<15}")
    print("-" * 60)
    for i in range(len(grid_sizes)):
        if i == 0:
            print(f"{grid_sizes[i]:<15.6e} {errors[i]:<20.6e} {'-':<15}")
        else:
            print(f"{grid_sizes[i]:<15.6e} {errors[i]:<20.6e} {rates[i-1]:<15.4f}")
    print("-" * 60)
    print(f"平均收敛阶数: {np.mean(rates):.4f}")
    print("=" * 60)
