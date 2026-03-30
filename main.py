"""
一维非稳态热传导方程数值求解主程序
====================================
求解方程: ∂u/∂t = α·∂²u/∂x²
初始条件: u(x,0) = sin(πx/L)
边界条件: u(0,t) = 0, u(L,t) = 0

功能:
1. FTCS显式有限差分法数值求解
2. 解析解计算与精度验证
3. 误差分析与收敛性研究
4. CFL稳定性条件验证
5. 可视化结果输出

运行方式: python main.py
"""

import numpy as np
import os
import time
from typing import Dict, List

from solver import (
    HeatEquationSolver,
    ftcs_solver_vectorized,
    analytical_solution,
    simulate_instability
)
from utils import (
    compute_rmse,
    compute_max_error,
    compute_l2_error,
    compute_absolute_error,
    check_cfl_stability,
    generate_grid,
    print_simulation_info,
    print_error_analysis,
    run_convergence_study
)
from plot_utils import (
    plot_temperature_distribution,
    plot_contour,
    plot_comparison,
    plot_error_distribution,
    plot_error_evolution,
    plot_convergence,
    plot_stability_comparison,
    plot_time_snapshots,
    plot_instability_evolution,
    create_output_directory,
    generate_all_plots
)


def run_basic_simulation(alpha: float, L: float, T: float, nx: int, nt: int,
                         output_dir: str) -> Dict:
    """
    运行基础数值仿真
    
    参数:
        alpha: 热扩散系数
        L: 空间域长度
        T: 总时间
        nx: 空间内部网格点数
        nt: 时间步数
        output_dir: 输出目录
    
    返回:
        包含仿真结果的字典
    """
    print("\n" + "=" * 70)
    print("【第一部分】基础数值仿真")
    print("=" * 70)
    
    solver = HeatEquationSolver(alpha=alpha, L=L, T=T, nx=nx, nt=nt)
    
    info = solver.get_info()
    print_simulation_info(**info)
    
    print("\n开始数值求解...")
    start_time = time.time()
    U_numerical, U_analytical = solver.solve(use_vectorized=True)
    elapsed_time = time.time() - start_time
    print(f"求解完成，耗时: {elapsed_time:.4f} 秒")
    
    numerical_final = U_numerical[-1, 1:-1]
    analytical_final = U_analytical[-1, 1:-1]
    
    rmse = compute_rmse(numerical_final, analytical_final)
    max_err = compute_max_error(numerical_final, analytical_final)
    l2_err = compute_l2_error(numerical_final, analytical_final, solver.dx)
    
    print_error_analysis(rmse, max_err, l2_err)
    
    return {
        'solver': solver,
        'U_numerical': U_numerical,
        'U_analytical': U_analytical,
        'rmse': rmse,
        'max_err': max_err,
        'l2_err': l2_err
    }


def run_convergence_analysis(alpha: float, L: float, T: float,
                             output_dir: str) -> Dict:
    """
    运行收敛性分析
    
    参数:
        alpha: 热扩散系数
        L: 空间域长度
        T: 总时间
        output_dir: 输出目录
    
    返回:
        收敛性分析结果
    """
    print("\n" + "=" * 70)
    print("【第二部分】收敛性分析")
    print("=" * 70)
    
    nx_list = [10, 20, 40, 80, 160]
    
    print(f"\n网格细化序列: {nx_list}")
    print("CFL数设定: r = 0.4 (满足稳定性条件)")
    
    results = {
        'nx_list': nx_list,
        'dx_list': [],
        'rmse_list': [],
        'max_err_list': [],
        'l2_err_list': [],
        'convergence_rates': []
    }
    
    cfl_ratio = 0.4
    
    for nx in nx_list:
        dx = L / (nx + 1)
        dt = cfl_ratio * dx**2 / alpha
        nt = max(int(T / dt), 1)
        dt = T / nt
        
        x, t, dx_actual, dt_actual = generate_grid(L, T, nx, nt)
        
        U_num = ftcs_solver_vectorized(alpha, L, T, nx, nt)
        U_ana = analytical_solution(x, T, alpha, L)
        
        numerical_final = U_num[-1, 1:-1]
        analytical_final = U_ana[1:-1]
        
        rmse = compute_rmse(numerical_final, analytical_final)
        max_err = compute_max_error(numerical_final, analytical_final)
        l2_err = compute_l2_error(numerical_final, analytical_final, dx_actual)
        
        results['dx_list'].append(dx_actual)
        results['rmse_list'].append(rmse)
        results['max_err_list'].append(max_err)
        results['l2_err_list'].append(l2_err)
        
        print(f"nx={nx:3d}, dx={dx_actual:.6f}, max_err={max_err:.6e}, rmse={rmse:.6e}")
    
    rates = []
    for i in range(1, len(results['max_err_list'])):
        rate = np.log(results['max_err_list'][i-1] / results['max_err_list'][i]) / \
               np.log(results['dx_list'][i-1] / results['dx_list'][i])
        rates.append(rate)
    results['convergence_rates'] = rates
    
    print("\n收敛阶分析:")
    for i, rate in enumerate(rates):
        print(f"  从nx={nx_list[i]}到nx={nx_list[i+1]}: 收敛阶 ≈ {rate:.2f}")
    
    plot_convergence(results['dx_list'], results['max_err_list'],
                     "收敛性分析 (最大误差 vs 空间步长)",
                     os.path.join(output_dir, "convergence.png"))
    
    return results


def run_stability_analysis(alpha: float, L: float, T: float,
                           output_dir: str) -> Dict:
    """
    运行稳定性分析
    
    参数:
        alpha: 热扩散系数
        L: 空间域长度
        T: 总时间
        output_dir: 输出目录
    
    返回:
        稳定性分析结果
    """
    print("\n" + "=" * 70)
    print("【第三部分】CFL稳定性条件验证")
    print("=" * 70)
    
    nx = 50
    dx = L / (nx + 1)
    
    print(f"\n空间网格点数 nx = {nx}")
    print(f"空间步长 dx = {dx:.6f}")
    
    r_stable = 0.4
    dt_stable = r_stable * dx**2 / alpha
    nt_stable = max(int(T / dt_stable), 1)
    dt_stable = T / nt_stable
    r_stable_actual = alpha * dt_stable / dx**2
    
    print(f"\n稳定情况: r = {r_stable_actual:.4f} (≤ 0.5)")
    
    x_stable, t_stable, _, _ = generate_grid(L, T, nx, nt_stable)
    U_stable = ftcs_solver_vectorized(alpha, L, T, nx, nt_stable)
    
    r_unstable = 0.6
    dt_unstable = r_unstable * dx**2 / alpha
    nt_unstable = max(int(T / dt_unstable), 1)
    dt_unstable = T / nt_unstable
    r_unstable_actual = alpha * dt_unstable / dx**2
    
    print(f"不稳定情况: r = {r_unstable_actual:.4f} (> 0.5)")
    
    x_unstable, t_unstable, _, _ = generate_grid(L, T, nx, nt_unstable)
    U_unstable = ftcs_solver_vectorized(alpha, L, T, nx, nt_unstable)
    
    is_stable_stable, _ = check_cfl_stability(alpha, dt_stable, dx)
    is_stable_unstable, _ = check_cfl_stability(alpha, dt_unstable, dx)
    
    print(f"\n稳定性检验:")
    print(f"  情况1 (r={r_stable_actual:.4f}): {'稳定 ✓' if is_stable_stable else '不稳定 ✗'}")
    print(f"  情况2 (r={r_unstable_actual:.4f}): {'稳定 ✓' if is_stable_unstable else '不稳定 ✗'}")
    
    plot_stability_comparison(
        x_stable, U_stable[-1, :], U_unstable[-1, :],
        t_stable[-1], t_unstable[-1],
        os.path.join(output_dir, "stability_comparison.png")
    )
    
    plot_instability_evolution(
        x_unstable, t_unstable, U_unstable, 5,
        "数值不稳定现象演化 (r > 0.5)",
        os.path.join(output_dir, "instability_evolution.png")
    )
    
    return {
        'r_stable': r_stable_actual,
        'r_unstable': r_unstable_actual,
        'U_stable': U_stable,
        'U_unstable': U_unstable,
        'x': x_stable,
        't_stable': t_stable,
        't_unstable': t_unstable
    }


def run_error_evolution_analysis(alpha: float, L: float, T: float,
                                 nx: int, nt: int, output_dir: str) -> Dict:
    """
    运行误差演化分析
    
    参数:
        alpha: 热扩散系数
        L: 空间域长度
        T: 总时间
        nx: 空间网格点数
        nt: 时间步数
        output_dir: 输出目录
    
    返回:
        误差演化分析结果
    """
    print("\n" + "=" * 70)
    print("【第四部分】误差随时间演化分析")
    print("=" * 70)
    
    solver = HeatEquationSolver(alpha=alpha, L=L, T=T, nx=nx, nt=nt)
    U_num, U_ana = solver.solve()
    
    x = solver.x
    t = solver.t
    
    errors_rmse = []
    errors_max = []
    
    for n in range(len(t)):
        rmse = compute_rmse(U_num[n, 1:-1], U_ana[n, 1:-1])
        max_err = compute_max_error(U_num[n, 1:-1], U_ana[n, 1:-1])
        errors_rmse.append(rmse)
        errors_max.append(max_err)
    
    print(f"\n初始时刻误差: RMSE = {errors_rmse[0]:.6e}")
    print(f"最终时刻误差: RMSE = {errors_rmse[-1]:.6e}")
    print(f"误差变化趋势: {'递增' if errors_rmse[-1] > errors_rmse[0] else '递减'}")
    
    plot_error_evolution(t, errors_rmse, "RMSE误差随时间演化",
                         os.path.join(output_dir, "error_evolution.png"))
    
    return {
        't': t,
        'errors_rmse': errors_rmse,
        'errors_max': errors_max
    }


def main():
    """
    主函数：执行完整的一维热传导方程数值求解流程
    """
    print("\n" + "=" * 70)
    print("    一维非稳态热传导方程数值求解与性能分析")
    print("    One-Dimensional Unsteady Heat Conduction Equation Solver")
    print("=" * 70)
    
    alpha = 0.01
    L = 1.0
    T = 1.0
    nx = 50
    nt = 500
    
    print("\n【可配置参数】")
    print(f"  热扩散系数 α = {alpha}")
    print(f"  空间域长度 L = {L}")
    print(f"  总时间 T = {T}")
    print(f"  空间网格点数 nx = {nx}")
    print(f"  时间步数 nt = {nt}")
    
    output_dir = create_output_directory("output")
    
    basic_results = run_basic_simulation(alpha, L, T, nx, nt, output_dir)
    
    convergence_results = run_convergence_analysis(alpha, L, T, output_dir)
    
    stability_results = run_stability_analysis(alpha, L, T, output_dir)
    
    error_evolution_results = run_error_evolution_analysis(alpha, L, T, nx, nt, output_dir)
    
    print("\n" + "=" * 70)
    print("【生成可视化图表】")
    print("=" * 70)
    
    solver = basic_results['solver']
    generate_all_plots(solver, output_dir)
    
    print("\n" + "=" * 70)
    print("【数值分析总结】")
    print("=" * 70)
    
    print(f"\n1. 精度验证:")
    print(f"   - 最终时刻RMSE: {basic_results['rmse']:.6e}")
    print(f"   - 最终时刻最大误差: {basic_results['max_err']:.6e}")
    
    print(f"\n2. 收敛性:")
    avg_rate = np.mean(convergence_results['convergence_rates'])
    print(f"   - 平均收敛阶: {avg_rate:.2f}")
    print(f"   - 理论收敛阶: 2.0 (FTCS格式空间二阶精度)")
    
    print(f"\n3. 稳定性:")
    print(f"   - CFL条件: r = α·dt/dx² ≤ 0.5")
    print(f"   - 稳定情况 r = {stability_results['r_stable']:.4f} ✓")
    print(f"   - 不稳定情况 r = {stability_results['r_unstable']:.4f} (数值发散)")
    
    print("\n" + "=" * 70)
    print(f"所有结果已保存至: {output_dir}")
    print("=" * 70)
    print("\n程序执行完毕！")
    
    return {
        'basic': basic_results,
        'convergence': convergence_results,
        'stability': stability_results,
        'error_evolution': error_evolution_results
    }


if __name__ == "__main__":
    results = main()
