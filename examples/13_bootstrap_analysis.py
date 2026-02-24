"""
自助法统计分析示例

展示如何使用 fincore 的自助法 (Bootstrap) 功能：
- 统计量置信区间估计
- Sharpe 比率置信区间
- 收益分布 bootstrap
- 假设检验

适用场景：
- 评估策略统计显著性
- 计算置信区间
- 参数估计的稳健性检验
- 小样本统计推断
"""

import numpy as np
import pandas as pd
from fincore.simulation import bootstrap
from fincore import sharpe_ratio, max_drawdown

print("=" * 70)
print("自助法统计分析示例")
print("=" * 70)

# 生成模拟收益数据
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=252, freq="B", tz="UTC")
returns = pd.Series(
    np.random.normal(0.0008, 0.015, len(dates)),
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
# 1. 基本统计量 Bootstrap
# ============================================================
print("\n" + "=" * 70)
print("1. 基本统计量 Bootstrap")
print("=" * 70)

# 均值的 Bootstrap 分布
print("\n均值 Bootstrap (n_samples=10000):")
boot_mean = bootstrap(returns, n_samples=10000, statistic="mean", seed=42)

print(f"  原始均值:     {returns.mean():.6f}")
print(f"  Bootstrap均值: {boot_mean.mean():.6f}")
print(f"  标准误:       {boot_mean.std():.6f}")

# 95% 置信区间
ci_95 = np.percentile(boot_mean, [2.5, 97.5])
print(f"  95% 置信区间: [{ci_95[0]:.6f}, {ci_95[1]:.6f}]")

# 标准差的 Bootstrap
print("\n标准差 Bootstrap:")
boot_std = bootstrap(returns, n_samples=10000, statistic="std", seed=42)

print(f"  原始标准差:   {returns.std():.6f}")
print(f"  Bootstrap均值: {boot_std.mean():.6f}")
ci_95_std = np.percentile(boot_std, [2.5, 97.5])
print(f"  95% 置信区间: [{ci_95_std[0]:.6f}, {ci_95_std[1]:.6f}]")

# ============================================================
# 2. Sharpe 比率 Bootstrap
# ============================================================
print("\n" + "=" * 70)
print("2. Sharpe 比率 Bootstrap")
print("=" * 70)

print("\nSharpe 比率 Bootstrap (n_samples=10000):")
boot_sharpe = bootstrap(returns, n_samples=10000, statistic="sharpe", seed=42)

original_sharpe = sharpe_ratio(returns)
print(f"  原始 Sharpe:  {original_sharpe:.6f}")
print(f"  Bootstrap均值: {boot_sharpe.mean():.6f}")
print(f"  标准误:       {boot_sharpe.std():.6f}")

# 95% 置信区间
ci_95_sharpe = np.percentile(boot_sharpe, [2.5, 97.5])
print(f"  95% 置信区间: [{ci_95_sharpe[0]:.6f}, {ci_95_sharpe[1]:.6f}]")

# Sharpe 比率的显著性检验
sharpe_p_value = (boot_sharpe <= 0).mean()
print(f"\nSharpe 比率显著性检验:")
print(f"  H0: Sharpe = 0")
print(f"  p值: {sharpe_p_value:.4f}")
if sharpe_p_value < 0.05:
    print(f"  结论: Sharpe 比率在 5% 水平下显著异于 0")
else:
    print(f"  结论: Sharpe 比率在 5% 水平下不显著")

# ============================================================
# 3. 自定义统计量 Bootstrap
# ============================================================
print("\n" + "=" * 70)
print("3. 自定义统计量 Bootstrap")
print("=" * 70)

# 定义自定义统计量函数
def custom_statistic(returns):
    """自定义统计量: 95分位数与5分位数的比率"""
    p95 = np.percentile(returns, 95)
    p5 = np.percentile(returns, 5)
    return p95 / abs(p5) if p5 != 0 else np.nan

print("\n自定义统计量 Bootstrap (尾部比率):")
boot_custom = bootstrap(returns, n_samples=10000,
                        statistic=custom_statistic, seed=42)

original_custom = custom_statistic(returns)
print(f"  原始值:       {original_custom:.6f}")
print(f"  Bootstrap均值: {boot_custom.mean():.6f}")
ci_95_custom = np.percentile(boot_custom, [2.5, 97.5])
print(f"  95% 置信区间: [{ci_95_custom[0]:.6f}, {ci_95_custom[1]:.6f}]")

# 定义最大回撤统计量
def max_dd_stat(returns):
    """最大回撤统计量"""
    cum = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cum)
    drawdown = (cum - running_max) / running_max
    return drawdown.min()

print("\n最大回撤 Bootstrap:")
boot_dd = bootstrap(returns, n_samples=10000,
                   statistic=max_dd_stat, seed=42)

original_dd = max_dd_stat(returns)
print(f"  原始最大回撤:  {original_dd:.6f}")
print(f"  Bootstrap均值: {boot_dd.mean():.6f}")
ci_95_dd = np.percentile(boot_dd, [5, 95])  # 单侧置信区间
print(f"  90% 置信区间: [{ci_95_dd[0]:.6f}, {ci_95_dd[1]:.6f}]")

# ============================================================
# 4. 中位数 Bootstrap
# ============================================================
print("\n" + "=" * 70)
print("4. 中位数 Bootstrap (稳健性分析)")
print("=" * 70)

print("\n中位数 Bootstrap:")
boot_median = bootstrap(returns, n_samples=10000, statistic="median", seed=42)

print(f"  原始中位数:   {returns.median():.6f}")
print(f"  原始均值:     {returns.mean():.6f}")
print(f"  Bootstrap中位数: {boot_median.mean():.6f}")
ci_95_median = np.percentile(boot_median, [2.5, 97.5])
print(f"  95% 置信区间: [{ci_95_median[0]:.6f}, {ci_95_median[1]:.6f}]")

# 均值 vs 中位数的比较
print("\n均值 vs 中位数分析:")
print(f"  偏度:         {returns.skew():.4f}")
print(f"  峰度:         {returns.kurtosis():.4f}")
if abs(returns.skew()) > 0.5:
    print(f"  说明: 数据存在明显偏态，中位数可能更稳健")

# ============================================================
# 5. Bootstrap 样本量敏感性分析
# ============================================================
print("\n" + "=" * 70)
print("5. Bootstrap 样本量敏感性分析")
print("=" * 70)

sample_sizes = [100, 500, 1000, 5000, 10000]
print(f"\n不同 Bootstrap 样本量下的标准误估计:")
print(f"{'样本量':<10} {'均值标准误':>15} {'Sharpe标准误':>15}")
print("-" * 45)

for n in sample_sizes:
    boot_mean_n = bootstrap(returns, n_samples=n, statistic="mean", seed=42)
    boot_sharpe_n = bootstrap(returns, n_samples=n, statistic="sharpe", seed=42)
    print(f"{n:<10} {boot_mean_n.std():>15.6f} {boot_sharpe_n.std():>15.6f}")

# ============================================================
# 6. 分位数 Bootstrap 分析
# ============================================================
print("\n" + "=" * 70)
print("6. 收益分布分位数 Bootstrap")
print("=" * 70)

# 计算多个分位数的 Bootstrap 置信区间
quantiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
print(f"\n收益分布分位数 Bootstrap 95% 置信区间:")
print(f"{'分位数':<8} {'原始值':>12} {'下限':>12} {'上限':>12} {'宽度':>12}")
print("-" * 60)

boot_median_full = bootstrap(returns, n_samples=5000, statistic="median", seed=42)

for q in quantiles:
    original = np.percentile(returns, q)
    boot_q = np.percentile(boot_median_full, [q])
    ci_low = np.percentile(returns, q) - 2 * (np.percentile(returns, q) - np.percentile(boot_median_full, q))
    ci_high = np.percentile(returns, q) + 2 * (np.percentile(boot_median_full, q) - np.percentile(returns, q))
    print(f"{q:>3}%     {original:>10.4f}  {ci_low:>10.4f}  {ci_high:>10.4f}  {ci_high-ci_low:>10.4f}")

# ============================================================
# 7. 假设检验
# ============================================================
print("\n" + "=" * 70)
print("7. 假设检验")
print("=" * 70)

# 检验年化收益是否显著大于 0
print("\n检验: 年化收益是否显著大于 0?")

# Bootstrap 年化收益
boot_annual_return = bootstrap(returns, n_samples=10000,
                                statistic=lambda x: x.mean() * 252,
                                seed=42)

# 原假设: 年化收益 = 0
original_annual = returns.mean() * 252
p_value = (boot_annual_return <= 0).mean()

print(f"  原始年化收益: {original_annual:.4f}")
print(f"  p值: {p_value:.4f}")

if p_value < 0.01:
    print(f"  结论: 在 1% 显著性水平下拒绝原假设，收益显著 > 0")
elif p_value < 0.05:
    print(f"  结论: 在 5% 显著性水平下拒绝原假设，收益显著 > 0")
elif p_value < 0.10:
    print(f"  结论: 在 10% 显著性水平下拒绝原假设，收益显著 > 0")
else:
    print(f"  结论: 无法拒绝原假设，收益不显著")

# ============================================================
# 8. 可视化
# ============================================================
try:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. 均值 Bootstrap 分布
    ax = axes[0, 0]
    ax.hist(boot_mean, bins=50, edgecolor='black', alpha=0.7, density=True)
    ax.axvline(x=returns.mean(), color='red', linestyle='--', linewidth=2, label='原始均值')
    ax.axvline(x=ci_95[0], color='orange', linestyle=':', label='95% CI')
    ax.axvline(x=ci_95[1], color='orange', linestyle=':')
    ax.set_title('均值 Bootstrap 分布')
    ax.set_xlabel('均值')
    ax.set_ylabel('密度')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Sharpe 比率 Bootstrap 分布
    ax = axes[0, 1]
    ax.hist(boot_sharpe, bins=50, edgecolor='black', alpha=0.7, density=True)
    ax.axvline(x=original_sharpe, color='red', linestyle='--', linewidth=2, label='原始 Sharpe')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1, label='零值')
    ax.axvline(x=ci_95_sharpe[0], color='orange', linestyle=':', label='95% CI')
    ax.axvline(x=ci_95_sharpe[1], color='orange', linestyle=':')
    ax.set_title('Sharpe 比率 Bootstrap 分布')
    ax.set_xlabel('Sharpe 比率')
    ax.set_ylabel('密度')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. 最大回撤 Bootstrap 分布
    ax = axes[1, 0]
    ax.hist(boot_dd, bins=50, edgecolor='black', alpha=0.7, density=True)
    ax.axvline(x=original_dd, color='red', linestyle='--', linewidth=2, label='原始最大回撤')
    ax.axvline(x=ci_95_dd[0], color='orange', linestyle=':', label='90% CI')
    ax.axvline(x=ci_95_dd[1], color='orange', linestyle=':')
    ax.set_title('最大回撤 Bootstrap 分布')
    ax.set_xlabel('最大回撤')
    ax.set_ylabel('密度')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. 样本量影响
    ax = axes[1, 1]
    sample_sizes_range = range(100, 5100, 100)
    se_means = []
    se_sharpes = []

    for n in sample_sizes_range:
        bm = bootstrap(returns, n_samples=n, statistic="mean", seed=42)
        bs = bootstrap(returns, n_samples=n, statistic="sharpe", seed=42)
        se_means.append(bm.std())
        se_sharpes.append(bs.std())

    ax.plot(sample_sizes_range, se_means, label='均值标准误', marker='o', markersize=3)
    ax.plot(sample_sizes_range, se_sharpes, label='Sharpe标准误', marker='s', markersize=3)
    ax.set_xlabel('Bootstrap 样本量')
    ax.set_ylabel('标准误')
    ax.set_title('Bootstrap 样本量对标准误的影响')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('bootstrap_analysis.png', dpi=100)
    print("\nBootstrap 分析可视化已保存: bootstrap_analysis.png")

except ImportError:
    print("\n未安装 matplotlib，跳过可视化")

print("\n" + "=" * 70)
print("Bootstrap 统计分析完成！")
print("=" * 70)
print("""
提示：
1. Bootstrap 是一种非参数统计方法，不需要假设数据分布
2. 通过重采样可以得到统计量的分布和置信区间
3. 增加样本量可以获得更稳定的估计
4. 可用于评估策略的统计显著性
5. 自定义统计量可以灵活应用于各种场景
""")
