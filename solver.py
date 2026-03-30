"""
核心求解模块：数值解与解析解计算
包含FTCS显式有限差分法和解析解计算
"""

import numpy as np
from scipy.special import erfc
from typing import Tuple, Optional


def analytical_solution(x: np.ndarray, t: float, alpha: float, L: float) -> np.ndarray:
    """
    一维热传导方程解析解
    
    对于初始条件 u(x,0) = sin(πx/L) 和边界条件 u(0,t)=u(L,t)=0，
    解析解为：u(x,t) = sin(πx/L) * exp(-α*(π/L)²*t)
    
    参数:
        x: 空间坐标数组
        t: 时间
        alpha: 热扩散系数
        L: 空间域长度
    
    返回:
        解析解数组
    """
    if t == 0:
        return np.sin(np.pi * x / L)
    
    return np.sin(np.pi * x / L) * np.exp(-alpha * (np.pi / L) ** 2 * t)


def analytical_solution_full(x: np.ndarray, t: np.ndarray, alpha: float, L: float) -> np.ndarray:
    """
    计算完整的时空解析解矩阵
    
    参数:
        x: 空间坐标数组
        t: 时间坐标数组
        alpha: 热扩散系数
        L: 空间域长度
    
    返回:
        解析解矩阵 U[nt+1, nx+2]
    """
    nt = len(t) - 1
    nx = len(x) - 2
    
    U = np.zeros((nt + 1, nx + 2))
    
    for n in range(nt + 1):
        U[n, :] = analytical_solution(x, t[n], alpha, L)
    
    return U


def ftcs_solver(alpha: float, L: float, T: float, nx: int, nt: int,
                initial_func: Optional[callable] = None) -> np.ndarray:
    """
    FTCS (Forward Time Central Space) 显式有限差分求解器
    
    离散格式：
    u[i,n+1] = u[i,n] + r*(u[i+1,n] - 2*u[i,n] + u[i-1,n])
    其中 r = α*dt/dx²
    
    参数:
        alpha: 热扩散系数
        L: 空间域长度
        T: 总时间
        nx: 空间内部网格点数
        nt: 时间步数
        initial_func: 初始条件函数，默认为 sin(πx/L)
    
    返回:
        数值解矩阵 U[nt+1, nx+2]，包含边界点
    """
    dx = L / (nx + 1)
    dt = T / nt
    r = alpha * dt / (dx ** 2)
    
    x = np.linspace(0, L, nx + 2)
    t = np.linspace(0, T, nt + 1)
    
    U = np.zeros((nt + 1, nx + 2))
    
    if initial_func is None:
        U[0, :] = np.sin(np.pi * x / L)
    else:
        U[0, :] = initial_func(x)
    
    U[0, 0] = 0
    U[0, -1] = 0
    
    for n in range(nt):
        for i in range(1, nx + 1):
            U[n+1, i] = U[n, i] + r * (U[n, i+1] - 2*U[n, i] + U[n, i-1])
        
        U[n+1, 0] = 0
        U[n+1, -1] = 0
    
    return U


def ftcs_solver_vectorized(alpha: float, L: float, T: float, nx: int, nt: int,
                           initial_func: Optional[callable] = None) -> np.ndarray:
    """
    向量化FTCS求解器（性能优化版本）
    
    使用NumPy向量化操作替代循环，大幅提升计算效率
    
    参数:
        alpha: 热扩散系数
        L: 空间域长度
        T: 总时间
        nx: 空间内部网格点数
        nt: 时间步数
        initial_func: 初始条件函数
    
    返回:
        数值解矩阵 U[nt+1, nx+2]
    """
    dx = L / (nx + 1)
    dt = T / nt
    r = alpha * dt / (dx ** 2)
    
    x = np.linspace(0, L, nx + 2)
    
    U = np.zeros((nt + 1, nx + 2))
    
    if initial_func is None:
        U[0, :] = np.sin(np.pi * x / L)
    else:
        U[0, :] = initial_func(x)
    
    U[0, 0] = 0
    U[0, -1] = 0
    
    for n in range(nt):
        U[n+1, 1:-1] = U[n, 1:-1] + r * (U[n, 2:] - 2*U[n, 1:-1] + U[n, :-2])
        U[n+1, 0] = 0
        U[n+1, -1] = 0
    
    return U


def simulate_instability(alpha: float, L: float, T: float, nx: int,
                         r_target: float = 0.6) -> Tuple[np.ndarray, float, float, float]:
    """
    模拟数值不稳定现象
    
    当CFL条件不满足时(r > 0.5)，数值解将发散
    
    参数:
        alpha: 热扩散系数
        L: 空间域长度
        T: 总时间
        nx: 空间内部网格点数
        r_target: 目标CFL数（应大于0.5以观察不稳定）
    
    返回:
        (数值解矩阵, dx, dt, 实际r值)
    """
    dx = L / (nx + 1)
    dt = r_target * dx ** 2 / alpha
    nt = max(int(T / dt), 1)
    dt = T / nt
    r_actual = alpha * dt / (dx ** 2)
    
    U = ftcs_solver_vectorized(alpha, L, T, nx, nt)
    
    return U, dx, dt, r_actual


class HeatEquationSolver:
    """
    一维热传导方程求解器类
    
    封装了完整的求解流程，便于参数管理和结果访问
    """
    
    def __init__(self, alpha: float = 0.01, L: float = 1.0, T: float = 1.0,
                 nx: int = 50, nt: int = 500):
        """
        初始化求解器
        
        参数:
            alpha: 热扩散系数
            L: 空间域长度
            T: 总时间
            nx: 空间内部网格点数
            nt: 时间步数
        """
        self.alpha = alpha
        self.L = L
        self.T = T
        self.nx = nx
        self.nt = nt
        
        self._compute_grid()
        self._check_stability()
        
        self.U_numerical = None
        self.U_analytical = None
    
    def _compute_grid(self):
        """计算网格参数"""
        self.dx = self.L / (self.nx + 1)
        self.dt = self.T / self.nt
        self.x = np.linspace(0, self.L, self.nx + 2)
        self.t = np.linspace(0, self.T, self.nt + 1)
    
    def _check_stability(self):
        """检验稳定性条件"""
        self.r = self.alpha * self.dt / (self.dx ** 2)
        self.is_stable = self.r <= 0.5
    
    def solve(self, use_vectorized: bool = True):
        """
        执行数值求解
        
        参数:
            use_vectorized: 是否使用向量化求解器
        """
        if use_vectorized:
            self.U_numerical = ftcs_solver_vectorized(
                self.alpha, self.L, self.T, self.nx, self.nt
            )
        else:
            self.U_numerical = ftcs_solver(
                self.alpha, self.L, self.T, self.nx, self.nt
            )
        
        self.U_analytical = analytical_solution_full(
            self.x, self.t, self.alpha, self.L
        )
        
        return self.U_numerical, self.U_analytical
    
    def get_solution_at_time(self, time_index: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取指定时刻的数值解和解析解
        
        参数:
            time_index: 时间索引
        
        返回:
            (数值解, 解析解)
        """
        if self.U_numerical is None:
            raise RuntimeError("请先调用solve()方法")
        return self.U_numerical[time_index, :], self.U_analytical[time_index, :]
    
    def get_final_solution(self) -> Tuple[np.ndarray, np.ndarray]:
        """获取最终时刻的解"""
        return self.get_solution_at_time(-1)
    
    def get_info(self) -> dict:
        """获取求解器参数信息"""
        return {
            'alpha': self.alpha,
            'L': self.L,
            'T': self.T,
            'nx': self.nx,
            'nt': self.nt,
            'dx': self.dx,
            'dt': self.dt,
            'r': self.r,
            'is_stable': self.is_stable
        }
