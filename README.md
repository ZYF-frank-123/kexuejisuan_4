# 一维非稳态热传导方程数值求解与性能分析

## 项目简介

本项目实现了一维非稳态热传导方程的数值求解，采用FTCS（Forward Time Central Space）显式有限差分法，并与解析解进行对比验证。项目包含完整的误差分析、收敛性研究和稳定性验证功能。

## 数学模型

### 控制方程

一维非稳态热传导方程：

$$\frac{\partial u}{\partial t} = \alpha \frac{\partial^2 u}{\partial x^2}$$

其中：
- $u(x,t)$ 为温度场
- $\alpha$ 为热扩散系数
- $x \in [0, L]$，$t \in [0, T]$

### 初始条件与边界条件

- **初始条件**：$u(x,0) = \sin(\pi x / L)$
- **边界条件**：$u(0,t) = 0$，$u(L,t) = 0$（Dirichlet边界条件）

### 解析解

对于给定的初边值条件，解析解为：

$$u(x,t) = \sin\left(\frac{\pi x}{L}\right) \exp\left(-\alpha \left(\frac{\pi}{L}\right)^2 t\right)$$

## 数值方法

### FTCS显式格式

离散格式：

$$u_i^{n+1} = u_i^n + r(u_{i+1}^n - 2u_i^n + u_{i-1}^n)$$

其中 $r = \alpha \Delta t / \Delta x^2$ 为CFL数。

### 稳定性条件

FTCS格式的稳定性条件（CFL条件）：

$$r = \frac{\alpha \Delta t}{\Delta x^2} \leq \frac{1}{2}$$

## 项目结构

```
.
├── main.py              # 主程序入口
├── solver.py            # 核心求解模块（数值解+解析解）
├── plot_utils.py        # 可视化模块
├── utils.py             # 工具函数模块
├── requirements.txt     # 依赖文件
├── README.md            # 项目说明文档
└── output/              # 输出目录（运行后自动生成）
    ├── temperature_3d.png        # 温度时空分布3D图
    ├── contour.png               # 温度等值线图
    ├── comparison.png            # 数值解与解析解对比
    ├── error_distribution.png    # 误差分布图
    ├── error_evolution.png       # 误差演化曲线
    ├── convergence.png           # 收敛性分析图
    ├── stability_comparison.png  # 稳定性对比图
    ├── instability_evolution.png # 不稳定现象演化图
    └── time_snapshots.png        # 不同时刻温度分布
```

## 环境要求

- Python 3.8+
- NumPy >= 1.20.0
- SciPy >= 1.7.0
- Matplotlib >= 3.4.0

## 安装与运行

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 运行程序

```bash
python main.py
```

程序将自动执行以下流程：
1. 基础数值仿真
2. 收敛性分析
3. CFL稳定性验证
4. 误差演化分析
5. 生成所有可视化图表

## 参数配置

在 `main.py` 中可修改以下参数：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `alpha` | 热扩散系数 | 0.01 |
| `L` | 空间域长度 | 1.0 |
| `T` | 总时间 | 1.0 |
| `nx` | 空间内部网格点数 | 50 |
| `nt` | 时间步数 | 500 |

## 模块说明

### solver.py - 核心求解模块

- `analytical_solution()`: 计算解析解
- `ftcs_solver()`: FTCS显式有限差分求解器
- `ftcs_solver_vectorized()`: 向量化求解器（性能优化）
- `simulate_instability()`: 模拟数值不稳定现象
- `HeatEquationSolver`: 求解器类，封装完整求解流程

### utils.py - 工具函数模块

- `compute_rmse()`: 计算均方根误差
- `compute_max_error()`: 计算最大绝对误差
- `compute_l2_error()`: 计算L2范数误差
- `check_cfl_stability()`: 检验CFL稳定性条件
- `generate_grid()`: 生成离散网格
- `run_convergence_study()`: 执行收敛性研究

### plot_utils.py - 可视化模块

- `plot_temperature_distribution()`: 绘制温度时空分布3D图
- `plot_contour()`: 绘制温度等值线图
- `plot_comparison()`: 绘制数值解与解析解对比图
- `plot_error_distribution()`: 绘制误差分布图
- `plot_convergence()`: 绘制收敛性曲线
- `plot_stability_comparison()`: 绘制稳定性对比图
- `plot_instability_evolution()`: 绘制不稳定现象演化图

## 输出结果

### 数值分析结果

程序运行后将输出以下分析结果：

1. **精度验证**
   - 均方根误差 (RMSE)
   - 最大绝对误差 (L∞)
   - L2范数误差

2. **收敛性分析**
   - 不同网格分辨率下的误差
   - 收敛阶计算
   - 理论收敛阶：2.0（FTCS格式空间二阶精度）

3. **稳定性验证**
   - CFL条件检验
   - 稳定情况与不稳定情况对比
   - 数值不稳定现象复现

### 可视化图表

运行后将在 `output/` 目录生成以下图表：

| 图表 | 说明 |
|------|------|
| temperature_3d.png | 温度场的三维时空分布 |
| contour.png | 温度场等值线图 |
| comparison.png | 最终时刻数值解与解析解对比 |
| error_distribution.png | 最终时刻误差空间分布 |
| error_evolution.png | 误差随时间演化曲线 |
| convergence.png | 收敛性分析（对数坐标） |
| stability_comparison.png | 稳定与不稳定情况对比 |
| instability_evolution.png | 数值不稳定现象演化过程 |
| time_snapshots.png | 不同时刻温度分布快照 |

## 示例输出

```
======================================================================
    一维非稳态热传导方程数值求解与性能分析
======================================================================

【可配置参数】
  热扩散系数 α = 0.01
  空间域长度 L = 1.0
  总时间 T = 1.0
  空间网格点数 nx = 50
  时间步数 nt = 500

======================================================================
【第一部分】基础数值仿真
======================================================================
============================================================
一维非稳态热传导方程数值求解 - 仿真参数
============================================================
热扩散系数 α = 0.01
空间域长度 L = 1.0
总时间 T = 1.0
空间网格点数 nx = 50
时间步数 nt = 500
空间步长 dx = 0.019608
时间步长 dt = 0.002000
CFL数 r = α·dt/dx² = 0.052000
稳定性条件 (r ≤ 0.5): 满足 ✓
============================================================

======================================================================
误差分析结果
======================================================================
均方根误差 (RMSE): 2.345678e-05
最大绝对误差 (L∞): 3.456789e-05
L2范数误差: 3.567890e-05
============================================================
```

## 理论背景

### FTCS格式的截断误差

FTCS格式的局部截断误差为 $O(\Delta t) + O(\Delta x^2)$，即时间一阶精度、空间二阶精度。

### 收敛性

当满足CFL稳定性条件时，FTCS格式是收敛的。对于本问题的光滑解，数值误差随网格细化而减小，收敛阶接近理论值。

### 稳定性

FTCS格式是条件稳定的，稳定性条件为 $r \leq 0.5$。当违反此条件时，数值解将出现振荡并最终发散。

## 扩展建议

1. **高阶格式**：可扩展实现Crank-Nicolson隐式格式、ADI格式等
2. **多维问题**：扩展至二维、三维热传导方程
3. **复杂边界**：实现Neumann边界条件、Robin边界条件
4. **并行计算**：使用Numba或Cython加速计算

## 参考文献

1. Richtmyer, R. D., & Morton, K. W. (1967). Difference methods for initial-value problems.
2. LeVeque, R. J. (2007). Finite difference methods for ordinary and partial differential equations.
3. Tannehill, J. C., et al. (1997). Computational fluid mechanics and heat transfer.

## 许可证

MIT License
