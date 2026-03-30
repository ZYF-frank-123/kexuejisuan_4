"""
主程序：一维非稳态热传导方程数值求解与性能分析

本程序实现：
1. FTCS显式有限差分法求解一维热传导方程
2. 与解析解对比进行精度验证
3. 误差分析和收敛性研究
4. CFL稳定性条件验证
5. 生成各种可视化图表
"""

import numpy as np
import matplotlib.pyplot as plt

# 导入自定义模块
from utils import (
    generate_grid,
    check_cfl_stability,
    calculate_rmse,
    calculate_max_error,
    calculate_convergence_rate,
    get_stability_threshold_dt,
    print_simulation_info,
    print_error_analysis,
    print_convergence_analysis
)

from solver import (
    ftcs_solver,
    analytical_solution,
    solve_with_different_timesteps,
    solve_convergence_study
)

from plot_utils import (
    plot_temperature_distribution,
    plot_temperature_heatmap,
    plot_3d_surface,
    plot_error_comparison,
    plot_error_evolution,
    plot_convergence_analysis as plot_convergence,
    plot_stability_analysis,
    plot_instability_evolution,
    plot_cfl_diagram,
    create_summary_figure
)


def main():
    """
    主函数：运行完整的数值求解流程
    """
    print("\n" + "="*70)
    print("一维非稳态热传导方程数值求解与性能分析")
    print("1D Unsteady Heat Conduction Equation Solver")
    print("="*70)

    # ============================================================
    # 参数设置（可根据需要自由修改）
    # ============================================================
    
    # 物理参数
    ALPHA = 0.01          # 热扩散系数 [m²/s]
    L = 1.0               # 空间区间长度 [m]
    T = 2.0               # 总计算时间 [s]
    
    # 数值参数（稳定情况）
    NX = 101              # 空间网格点数（包含边界）
    NT = 2001             # 时间步数
    
    # 收敛性研究参数
    CONVERGENCE_NX = [21, 41, 81, 161, 321]  # 不同网格点数
    CONVERGENCE_NT = [401, 801, 1601, 3201, 6401]  # 对应时间步数
    
    # 稳定性分析参数
    DT_MULTIPLIERS = [0.5, 1.0, 2.0, 4.0]  # 时间步长乘数
    
    # 输出设置
    OUTPUT_DIR = './figures'
    
    # ============================================================
    # 第一部分：基本求解与验证
    # ============================================================
    print("\n" + "-"*70)
    print("第一部分：基本求解与精度验证")
    print("-"*70)
    
    # 生成网格
    x, t, dx, dt = generate_grid(L, T, NX, NT)
    
    # 检查CFL稳定性条件
    is_stable, r = check_cfl_stability(ALPHA, dt, dx)
    dt_max = get_stability_threshold_dt(ALPHA, dx)
    
    # 打印仿真信息
    params = {
        'alpha': ALPHA,
        'L': L,
        'T': T,
        'nx': NX,
        'nt': NT,
        'dx': dx,
        'dt': dt,
        'r': r,
        'is_stable': is_stable,
        'dt_max': dt_max
    }
    print_simulation_info(params)
    
    # 数值求解
    print("\n正在进行数值求解...")
    u_numerical, _ = ftcs_solver(x, t, ALPHA, L, check_stability=True)
    
    # 计算解析解
    print("正在计算解析解...")
    u_analytical = analytical_solution(x, t, ALPHA, L)
    
    # 误差分析
    print("正在进行误差分析...")
    rmse_values = np.zeros(len(t))
    max_error_values = np.zeros(len(t))
    
    for i in range(len(t)):
        rmse_values[i] = calculate_rmse(u_numerical[i, :], u_analytical[i, :])
        max_error_values[i] = calculate_max_error(u_numerical[i, :], u_analytical[i, :])
    
    print_error_analysis(t, rmse_values, max_error_values)
    
    # ============================================================
    # 第二部分：可视化
    # ============================================================
    print("\n" + "-"*70)
    print("第二部分：生成可视化图表")
    print("-"*70)
    
    # 1. 温度空间分布（不同时刻）
    print("\n[1/8] 绘制温度空间分布...")
    t_selected = [0, T*0.25, T*0.5, T*0.75, T]
    plot_temperature_distribution(
        x, u_numerical, t_selected, t, L,
        title="数值解：温度空间分布演化",
        output_dir=OUTPUT_DIR,
        filename='01_temperature_distribution.png'
    )
    
    # 2. 温度时空分布热力图
    print("[2/8] 绘制时空热力图...")
    plot_temperature_heatmap(
        x, t, u_numerical, L, T,
        title="数值解：温度时空分布",
        output_dir=OUTPUT_DIR,
        filename='02_temperature_heatmap.png'
    )
    
    # 3. 3D表面图
    print("[3/8] 绘制3D表面图...")
    plot_3d_surface(
        x, t, u_numerical, L, T,
        title="数值解：温度时空分布3D图",
        output_dir=OUTPUT_DIR,
        filename='03_temperature_3d.png'
    )
    
    # 4. 数值解与解析解对比
    print("[4/8] 绘制解对比与误差分析...")
    plot_error_comparison(
        x, u_numerical, u_analytical, t_selected, t, L,
        output_dir=OUTPUT_DIR,
        filename='04_error_comparison.png'
    )
    
    # 5. 误差随时间演化
    print("[5/8] 绘制误差演化曲线...")
    plot_error_evolution(
        t, rmse_values, max_error_values,
        output_dir=OUTPUT_DIR,
        filename='05_error_evolution.png'
    )
    
    # ============================================================
    # 第三部分：收敛性分析
    # ============================================================
    print("\n" + "-"*70)
    print("第三部分：网格收敛性分析")
    print("-"*70)
    
    print("正在进行网格收敛性研究...")
    convergence_results = solve_convergence_study(
        ALPHA, L, T,
        CONVERGENCE_NX,
        CONVERGENCE_NT
    )
    
    # 计算收敛阶数
    rates = calculate_convergence_rate(
        convergence_results['rmse_values'],
        convergence_results['dx_values']
    )
    
    print_convergence_analysis(
        convergence_results['dx_values'],
        convergence_results['rmse_values'],
        rates
    )
    
    # 绘制收敛性分析图
    print("[6/8] 绘制收敛性分析图...")
    plot_convergence(
        convergence_results['dx_values'],
        convergence_results['rmse_values'],
        rates,
        output_dir=OUTPUT_DIR,
        filename='06_convergence_analysis.png'
    )
    
    # ============================================================
    # 第四部分：稳定性分析
    # ============================================================
    print("\n" + "-"*70)
    print("第四部分：CFL稳定性条件验证")
    print("-"*70)
    
    print("正在进行稳定性分析...")
    stability_results = solve_with_different_timesteps(
        x, t, ALPHA, L, DT_MULTIPLIERS
    )
    
    # 打印稳定性结果
    print("\n稳定性分析结果：")
    print(f"{'时间步长乘数':<15} {'dt':<15} {'CFL数 r':<15} {'稳定性':<10}")
    print("-" * 60)
    for key, result in stability_results.items():
        status = "稳定 ✓" if result['is_stable'] else "不稳定 ✗"
        print(f"{key:<15} {result['dt']:<15.6e} {result['r']:<15.4f} {status:<10}")
    
    # 绘制稳定性分析图
    print("\n[7/8] 绘制稳定性分析图...")
    plot_stability_analysis(
        x, stability_results, T*0.5,
        output_dir=OUTPUT_DIR,
        filename='07_stability_analysis.png'
    )
    
    # 绘制不稳定现象演化
    unstable_key = [k for k, v in stability_results.items() if not v['is_stable']]
    if unstable_key:
        print("   绘制数值不稳定现象演化...")
        unstable_result = stability_results[unstable_key[0]]
        t_unstable = unstable_result['t']
        u_unstable = unstable_result['u']
        
        # 选择几个时刻展示不稳定现象
        t_instability = [t_unstable[i] for i in [0, len(t_unstable)//4, len(t_unstable)//2, 
                                                   len(t_unstable)*3//4, -1]]
        plot_instability_evolution(
            x, t_unstable, u_unstable, t_instability,
            output_dir=OUTPUT_DIR,
            filename='08_instability_evolution.png'
        )
    
    # 绘制CFL条件示意图
    print("   绘制CFL条件示意图...")
    dt_stable = dt
    dt_unstable = dt * 4.0  # 使用4倍时间步长作为不稳定示例
    plot_cfl_diagram(
        ALPHA, dx, dt_stable, dt_unstable,
        output_dir=OUTPUT_DIR,
        filename='09_cfl_diagram.png'
    )
    
    # ============================================================
    # 第五部分：综合汇总图
    # ============================================================
    print("\n" + "-"*70)
    print("第五部分：生成综合汇总图")
    print("-"*70)
    
    print("[8/8] 生成综合汇总图...")
    create_summary_figure(
        x, t, u_numerical, u_analytical, L, T,
        output_dir=OUTPUT_DIR,
        filename='10_summary.png'
    )
    
    # ============================================================
    # 完成
    # ============================================================
    print("\n" + "="*70)
    print("所有计算和可视化任务已完成！")
    print(f"图表已保存至: {OUTPUT_DIR}/")
    print("="*70)
    
    # 最终数值结果汇总
    print("\n" + "="*70)
    print("数值结果汇总")
    print("="*70)
    print(f"物理参数:")
    print(f"  热扩散系数 α = {ALPHA:.6e} m²/s")
    print(f"  区间长度 L = {L:.4f} m")
    print(f"  总时间 T = {T:.4f} s")
    print(f"\n数值参数:")
    print(f"  空间网格点数 NX = {NX}")
    print(f"  时间步数 NT = {NT}")
    print(f"  空间步长 dx = {dx:.6f} m")
    print(f"  时间步长 dt = {dt:.6f} s")
    print(f"  CFL数 r = {r:.6f}")
    print(f"\n误差结果:")
    print(f"  最终时刻 RMSE = {rmse_values[-1]:.6e}")
    print(f"  最终时刻最大误差 = {max_error_values[-1]:.6e}")
    print(f"  全局平均 RMSE = {np.mean(rmse_values):.6e}")
    print(f"\n收敛性分析:")
    print(f"  平均收敛阶数 = {np.mean(rates):.4f}")
    print(f"  理论收敛阶数 = 2.0000 (空间二阶精度)")
    print("="*70)


if __name__ == "__main__":
    main()
