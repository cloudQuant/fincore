"""
蒙特卡洛模拟示例

展示如何使用 fincore 的蒙特卡洛模拟功能：
- 路径生成 (几何布朗运动)
- 风险度量 (VaR, CVaR)
- 情景压力测试
- 自助法置信区间

适用场景：
- 未来收益预测
- 风险情景分析
- 策略参数稳健性测试
- 资产配置压力测试
"""

import numpy as np
import pandas as pd
from fincore.simulation import MonteCarlo
from fincore import sharpe_ratio, max_drawdown

print("=" * 70)
print("蒙特卡洛模拟示例")
print("=" * 70)

# 生成历史收益数据
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=252*3, freq="B", tz="UTC")
returns = pd.Series(
    np.random.normal(0.0005, 0.015, len(dates)),
    index=dates,
    name="strategy_returns"
)

print(f"\n数据概览:")
print(f"  时间范围: {dates[0].date()} 至 {dates[-1].date()}")
print(f"  观测值数: {len(returns)}")
print(f"  年化收益: {returns.mean() * 252:.4f}")
print(f"  年化波动率: {returns.std() * np.sqrt(252):.4f}")
print(f"  夏普比率: {sharpe_ratio(returns):.4f}")

# ============================================================
# 1. 创建蒙特卡洛模拟引擎
# ============================================================
print("\n" + "=" * 70)
print("1. 创建蒙特卡洛模拟引擎")
print("=" * 70)

mc = MonteCarlo(returns)
print(f"\n蒙特卡洛引擎已创建")
print(f"  历史数据点: {len(mc.returns)}")
print(f"  历史年化波动率: {mc.returns.std() * np.sqrt(252):.4f}")
print(f"  历史年化收益: {mc.returns.mean() * 252:.4f}")

# ============================================================
# 2. 路径模拟 - 几何布朗运动
# ============================================================
print("\n" + "=" * 70)
print("2. 路径模拟 (几何布朗运动)")
print("=" * 70)

n_paths = 1000
horizon = 252  # 1年交易日

result = mc.simulate(
    n_paths=n_paths,
    horizon=horizon,
    seed=42
)

print(f"\n模拟完成:")
print(f"  路径数量: {n_paths}")
print(f"  模拟期间: {horizon} 个交易日")

# 分析模拟结果
final_values = result.paths[:, -1]
print(f"\n最终价值统计:")
print(f"  均值:     {final_values.mean():.4f}")
print(f"  中位数:   {np.median(final_values):.4f}")
print(f"  标准差:   {final_values.std():.4f}")
print(f"  最小值:   {final_values.min():.4f}")
print(f"  最大值:   {final_values.max():.4f}")

# 百分位数
percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
print(f"\n最终价值分位数:")
for p in percentiles:
    val = np.percentile(final_values, p)
    print(f"  {p:3d}%: {val:.4f}")

# ============================================================
# 3. 风险度量
# ============================================================
print("\n" + "=" * 70)
print("3. 风险度量")
print("=" * 70)

# 使用模拟结果计算风险度量
var_5 = result.var(alpha=0.05)
cvar_5 = result.cvar(alpha=0.05)
print(f"\n基于模拟的风险度量 (95% 置信水平):")
print(f"  VaR (5%):     {var_5:.4f}")
print(f"  CVaR (5%):    {cvar_5:.4f}")

# 与历史比较
historical_var = returns.quantile(0.05)
historical_cvar = returns[returns <= historical_var].mean()
print(f"\n历史风险度量对比:")
print(f"  历史 VaR (5%):  {historical_var:.4f}")
print(f"  历史 CVaR (5%): {historical_cvar:.4f}")

# ============================================================
# 4. 情景压力测试
# ============================================================
print("\n" + "=" * 70)
print("4. 情景压力测试")
print("=" * 70)

from fincore.simulation.scenarios import stress_test

# 定义压力情景
scenarios = {
    "市场崩盘": {"drift": -0.001, "volatility": 0.04},
    "高波动": {"drift": 0.0005, "volatility": 0.03},
    "熊市": {"drift": -0.0003, "volatility": 0.02},
    "牛市": {"drift": 0.001, "volatility": 0.015},
}

print("\n压力情景测试 (1000条路径, 252天):")
print("-" * 50)

for name, params in scenarios.items():
    result = mc.simulate(
        n_paths=1000,
        horizon=252,
        drift=params["drift"],
        volatility=params["volatility"],
        seed=42
    )
    final_vals = result.paths[:, -1]
    print(f"\n  {name}:")
    print(f"    参数: drift={params['drift']}, vol={params['volatility']}")
    print(f"    均值:   {final_vals.mean():.4f}")
    print(f"    中位数: {np.median(final_vals):.4f}")
    print(f"    5% VaR: {np.percentile(final_vals, 5):.4f}")

# ============================================================
# 5. 概率分析
# ============================================================
print("\n" + "=" * 70)
print("5. 目标达成概率分析")
print("=" * 70)

# 标准模拟
result = mc.simulate(n_paths=5000, horizon=252, seed=42)
final_values = result.paths[:, -1]

# 不同目标下的达成概率
targets = [0.0, 0.1, 0.2, 0.5, 1.0]  # 累计收益目标
print("\n目标收益达成概率 (1年期):")
print("-" * 40)
for target in targets:
    prob = (final_values >= target).mean() * 100
    print(f"  收益 >= {target:>5.1%}: {prob:>6.2f}%")

# 亏损概率分析
loss_thresholds = [-0.5, -0.3, -0.2, -0.1, 0.0]
print("\n亏损概率 (1年期):")
print("-" * 40)
for threshold in loss_thresholds:
    prob = (final_values < threshold).mean() * 100
    print(f"  亏损 > {threshold:>5.1%}: {prob:>6.2f}%")

# ============================================================
# 6. 时间路径分析
# ============================================================
print("\n" + "=" * 70)
print("6. 时间路径分析")
print("=" * 70)

# 分析模拟路径中的最大回撤分布
result = mc.simulate(n_paths=1000, horizon=252, seed=42)
paths = result.paths

# 计算每条路径的最大回撤
max_drawdowns = []
for i in range(paths.shape[0]):
    path_cum = np.cumprod(1 + paths[i, :])
    running_max = np.maximum.accumulate(path_cum)
    drawdown = (path_cum - running_max) / running_max
    max_drawdowns.append(drawdown.min())

max_drawdowns = np.array(max_drawdowns)

print(f"\n最大回撤分布 (1000条路径):")
print(f"  均值:     {max_drawdowns.mean():.4f}")
print(f"  中位数:   {np.median(max_drawdowns):.4f}")
print(f"  标准差:   {max_drawdowns.std():.4f}")
print(f"  最小值:   {max_drawdowns.min():.4f}")
print(f"  最大值:   {max_drawdowns.max():.4f}")

# 回撤分位数
dd_percentiles = [50, 75, 90, 95, 99]
print(f"\n最大回撤分位数:")
for p in dd_percentiles:
    val = np.percentile(max_drawdowns, p)
    print(f"  {p:3d}%: {val:.4f}")

# ============================================================
# 7. 可视化
# ============================================================
try:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. 随机选取的路径
    ax = axes[0, 0]
    n_show = 50
    indices = np.random.choice(paths.shape[0], n_show, replace=False)
    for i in indices:
        ax.plot(np.cumprod(1 + paths[i, :]), alpha=0.3, linewidth=0.5)
    ax.set_title('随机选取的模拟路径 (50条)')
    ax.set_xlabel('交易日')
    ax.set_ylabel('累计收益')
    ax.grid(True, alpha=0.3)

    # 2. 最终价值分布
    ax = axes[0, 1]
    final_vals = paths[:, -1]
    ax.hist(final_vals, bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(x=np.median(final_vals), color='red', linestyle='--', label='中位数')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=2, label='盈亏平衡')
    ax.set_title('最终价值分布')
    ax.set_xlabel('累计收益')
    ax.set_ylabel('频数')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. 路径百分位数
    ax = axes[1, 0]
    cum_paths = np.cumprod(1 + paths, axis=1)
    percentiles = [5, 25, 50, 75, 95]
    for p in percentiles:
        vals = np.percentile(cum_paths, p, axis=0)
        ax.plot(vals, label=f'{p}% 分位', linewidth=2 if p == 50 else 1)
    ax.set_title('路径百分位数')
    ax.set_xlabel('交易日')
    ax.set_ylabel('累计收益')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. 最大回撤分布
    ax = axes[1, 1]
    ax.hist(max_drawdowns, bins=50, edgecolor='black', alpha=0.7)
    ax.set_title('最大回撤分布')
    ax.set_xlabel('最大回撤')
    ax.set_ylabel('频数')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('monte_carlo_simulation.png', dpi=100)
    print("\n蒙特卡洛模拟可视化已保存: monte_carlo_simulation.png")

except ImportError:
    print("\n未安装 matplotlib，跳过可视化")

print("\n" + "=" * 70)
print("蒙特卡洛模拟分析完成！")
print("=" * 70)
print("""
提示：
1. 蒙特卡洛模拟基于历史波动率和漂移率生成未来路径
2. 可以自定义漂移率和波动率进行情景分析
3. 增加路径数量可以获得更稳定的统计结果
4. 可用于计算策略的风险度量和置信区间
5. 压力测试可以评估极端市场情况下的表现
""")
