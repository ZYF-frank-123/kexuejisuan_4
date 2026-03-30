"""
核心求解模块：包含FTCS数值求解器和解析解计算
"""

import numpy as np
from scipy.special import erf
from typing import Tuple, Optional


def initial_condition(x: np.ndarray, L: float) -> np.ndarray:
    """
    初始条件：u(x,0) = sin(πx/L)

    Parameters
    ----------
    x : np.ndarray
        空间网格点
    L : float
        区间长度

    Returns
    -------
    u0 : np.ndarray
        初始温度分布
    """
    return np.sin(np.pi * x / L)


def ftcs_solver(
    x: np.ndarray,
    t: np.ndarray,
    alpha: float,
    L: float,
    check_stability: bool = True
) -> Tuple[np.ndarray, bool]:
    """
    FTCS（Forward Time Centered Space）显式有限差分法求解一维热传导方程

    离散格式：
    u[i,n+1] = u[i,n] + r * (u[i+1,n] - 2*u[i,n] + u[i-1,n])
    其中 r = alpha * dt / dx^2

    Parameters
    ----------
    x : np.ndarray
        空间网格点，形状为 (nx,)
    t : np.ndarray
        时间网格点，形状为 (nt,)
    alpha : float
        热扩散系数
    L : float
        区间长度
    check_stability : bool, optional
        是否检查CFL稳定性条件，默认为True

    Returns
    -------
    u : np.ndarray
        数值解，形状为 (nt, nx)
    is_stable : bool
        是否满足稳定性条件
    """
    nx = len(x)
    nt = len(t)
    dx = x[1] - x[0]
    dt = t[1] - t[0]

    # 计算CFL数
    r = alpha * dt / (dx ** 2)
    is_stable = r <= 0.5

    if check_stability and not is_stable:
        print(f"警告：CFL条件不满足！r = {r:.4f} > 0.5")
        print(f"这可能导致数值不稳定。建议减小dt或增大dx。")

    # 初始化解矩阵
    u = np.zeros((nt, nx))

    # 设置初始条件
    u[0, :] = initial_condition(x, L)

    # 设置边界条件（Dirichlet边界：u=0）
    u[:, 0] = 0.0
    u[:, -1] = 0.0

    # FTCS时间推进
    for n in range(nt - 1):
        for i in range(1, nx - 1):
            u[n + 1, i] = u[n, i] + r * (u[n, i + 1] - 2 * u[n, i] + u[n, i - 1])

    return u, is_stable


def analytical_solution(
    x: np.ndarray,
    t: np.ndarray,
    alpha: float,
    L: float
) -> np.ndarray:
    """
    计算一维热传导方程的解析解

    对于初始条件 u(x,0) = sin(πx/L) 和Dirichlet边界条件，
    解析解为：u(x,t) = sin(πx/L) * exp(-α*(π/L)^2*t)

    Parameters
    ----------
    x : np.ndarray
        空间网格点，形状为 (nx,)
    t : np.ndarray
        时间网格点，形状为 (nt,)
    alpha : float
        热扩散系数
    L : float
        区间长度

    Returns
    -------
    u_exact : np.ndarray
        解析解，形状为 (nt, nx)
    """
    nx = len(x)
    nt = len(t)
    u_exact = np.zeros((nt, nx))

    # 计算衰减系数
    decay_coeff = alpha * (np.pi / L) ** 2

    for n in range(nt):
        for i in range(nx):
            u_exact[n, i] = np.sin(np.pi * x[i] / L) * np.exp(-decay_coeff * t[n])

    return u_exact


def solve_steady_state(x: np.ndarray, L: float) -> np.ndarray:
    """
    计算稳态解（t→∞时的极限解）

    对于Dirichlet边界条件 u(0,t)=u(L,t)=0，
    稳态解为 u(x) = 0

    Parameters
    ----------
    x : np.ndarray
        空间网格点
    L : float
        区间长度

    Returns
    -------
    u_steady : np.ndarray
        稳态解
    """
    return np.zeros_like(x)


def solve_with_different_timesteps(
    x: np.ndarray,
    t_base: np.ndarray,
    alpha: float,
    L: float,
    dt_multipliers: list
) -> dict:
    """
    使用不同时间步长求解，用于稳定性分析

    Parameters
    ----------
    x : np.ndarray
        空间网格点
    t_base : np.ndarray
        基准时间网格
    alpha : float
        热扩散系数
    L : float
        区间长度
    dt_multipliers : list
        时间步长乘数列表，用于调整dt大小

    Returns
    -------
    results : dict
        包含不同时间步长下求解结果的字典
    """
    results = {}
    dx = x[1] - x[0]
    dt_base = t_base[1] - t_base[0]

    for mult in dt_multipliers:
        dt_new = dt_base * mult
        # 根据新的dt重新生成时间网格
        T = t_base[-1]
        nt_new = int(T / dt_new) + 1
        t_new = np.linspace(0, T, nt_new)

        # 求解
        u, is_stable = ftcs_solver(x, t_new, alpha, L, check_stability=True)

        r = alpha * dt_new / (dx ** 2)
        results[f"dt_mult_{mult}"] = {
            'u': u,
            't': t_new,
            'dt': dt_new,
            'r': r,
            'is_stable': is_stable,
            'nt': nt_new
        }

    return results


def solve_convergence_study(
    alpha: float,
    L: float,
    T: float,
    nx_values: list,
    nt_values: list
) -> dict:
    """
    进行网格收敛性研究

    Parameters
    ----------
    alpha : float
        热扩散系数
    L : float
        区间长度
    T : float
        总时间
    nx_values : list
        不同空间网格点数的列表
    nt_values : list
        对应的时间步数列表

    Returns
    -------
    results : dict
        包含收敛性研究结果的字典
    """
    results = {
        'nx_values': np.array(nx_values),
        'dx_values': np.zeros(len(nx_values)),
        'dt_values': np.zeros(len(nx_values)),
        'r_values': np.zeros(len(nx_values)),
        'rmse_values': np.zeros(len(nx_values)),
        'max_error_values': np.zeros(len(nx_values)),
        'solutions': []
    }

    for i, (nx, nt) in enumerate(zip(nx_values, nt_values)):
        # 生成网格
        x = np.linspace(0, L, nx)
        t = np.linspace(0, T, nt)
        dx = L / (nx - 1)
        dt = T / (nt - 1)

        # 数值解
        u_numerical, _ = ftcs_solver(x, t, alpha, L, check_stability=False)

        # 解析解
        u_analytical = analytical_solution(x, t, alpha, L)

        # 计算误差
        rmse = np.sqrt(np.mean((u_numerical - u_analytical) ** 2))
        max_error = np.max(np.abs(u_numerical - u_analytical))

        # 计算CFL数
        r = alpha * dt / (dx ** 2)

        results['dx_values'][i] = dx
        results['dt_values'][i] = dt
        results['r_values'][i] = r
        results['rmse_values'][i] = rmse
        results['max_error_values'][i] = max_error
        results['solutions'].append({
            'x': x,
            't': t,
            'u_numerical': u_numerical,
            'u_analytical': u_analytical
        })

    return results
