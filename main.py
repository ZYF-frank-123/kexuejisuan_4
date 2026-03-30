"""
主程序：一维非稳态热传导方程数值求解与性能分析

运行流程：
1. 参数配置
2. 数值解与解析解计算
3. 误差分析
4. 收敛性分析
5. 稳定性验证与不稳定现象复现
6. 结果可视化
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

from solver import solve_ftcs, solve_analytical, check_cfl_stability
from utils import compute_absolute_error, compute_rmse, convergence_analysis
from plot_utils import (
    plot_temperature_distribution,
    plot_error_comparison,
    plot_convergence,
    plot_stability_comparison,
    plot_solution_comparison
)

def main():
    print("=" * 60)
    print("一维非稳态热传导方程数值求解与性能分析")
    print("=" * 60)
    
    L = 1.0
    T = 0.5
    alpha = 0.1
    nx = 51
    nt = 500
    
    print("\n参数配置:")
    print(f"  空间区间长度 L = {L}")
    print(f"  总时间 T = {T}")
    print(f"  热扩散系数 alpha = {alpha}")
    print(f"  空间节点数 nx = {nx}")
    print(f"  时间步数 nt = {nt}")
    
    print("\n" + "=" * 60)
    print("步骤1: 计算数值解与解析解")
    print("=" * 60)
    
    x, t, u_numerical, (is_stable, cfl_number) = solve_ftcs(L, T, alpha, nx, nt)
    _, _, u_analytical = solve_analytical(L, T, alpha, nx, nt)
    
    print(f"\nCFL稳定条件验证:")
    print(f"  CFL数 r = {cfl_number:.6f}")
    print(f"  是否满足稳定条件 (r <= 0.5): {is_stable}")
    
    print("\n" + "=" * 60)
    print("步骤2: 误差分析")
    print("=" * 60)
    
    abs_error = compute_absolute_error(u_numerical, u_analytical)
    rmse_global, rmse_time = compute_rmse(u_numerical, u_analytical)
    
    print(f"\n误差计算结果:")
    print(f"  全局均方根误差 (RMSE) = {rmse_global:.8f}")
    print(f"  最大绝对误差 = {np.max(abs_error):.8f}")
    print(f"  平均绝对误差 = {np.mean(abs_error):.8f}")
    
    print("\n" + "=" * 60)
    print("步骤3: 收敛性分析")
    print("=" * 60)
    
    nx_list = [11, 21, 41, 81, 161]
    print(f"\n测试的空间节点数: {nx_list}")
    
    dx_list, error_list, convergence_order = convergence_analysis(L, T, alpha, nx_list)
    
    print(f"\n收敛性分析结果:")
    for dx, err in zip(dx_list, error_list):
        print(f"  dx = {dx:.6f}, RMSE = {err:.8f}")
    print(f"  收敛阶数 = {convergence_order:.4f}")
    
    print("\n" + "=" * 60)
    print("步骤4: 不稳定现象验证")
    print("=" * 60)
    
    nx_unstable = 51
    nt_unstable = 150
    
    x_unstable, t_unstable, u_unstable, (is_stable_unstable, cfl_number_unstable) = \
        solve_ftcs(L, T, alpha, nx_unstable, nt_unstable)
    
    print(f"\n不稳定情况参数:")
    print(f"  空间节点数 nx = {nx_unstable}")
    print(f"  时间步数 nt = {nt_unstable}")
    print(f"  CFL数 r = {cfl_number_unstable:.6f}")
    print(f"  是否满足稳定条件: {is_stable_unstable}")
    
    print("\n" + "=" * 60)
    print("步骤5: 结果可视化")
    print("=" * 60)
    
    print("\n正在生成图表...")
    
    plot_temperature_distribution(x, t, u_numerical, title_prefix="数值解")
    plot_temperature_distribution(x, t, u_analytical, title_prefix="解析解")
    plot_solution_comparison(x, t, u_numerical, u_analytical)
    plot_error_comparison(x, t, abs_error, rmse_time)
    plot_convergence(dx_list, error_list, convergence_order)
    plot_stability_comparison(x_unstable, t_unstable, u_numerical[:len(t_unstable), :], u_unstable, cfl_number, cfl_number_unstable)
    
    print("\n所有图表已生成!")
    
    print("\n" + "=" * 60)
    print("分析总结")
    print("=" * 60)
    print(f"\n1. 数值方法: FTCS显式有限差分法")
    print(f"2. 理论收敛阶: O(Δt) + O(Δx²)")
    print(f"3. 实际计算收敛阶: {convergence_order:.4f}")
    print(f"4. 稳定条件: CFL数 <= 0.5 (当前: {cfl_number:.4f})")
    print(f"5. 不稳定现象: 当CFL数 > 0.5 时，数值解出现振荡发散")
    
    print("\n" + "=" * 60)
    print("程序执行完成!")
    print("=" * 60)

if __name__ == "__main__":
    main()
