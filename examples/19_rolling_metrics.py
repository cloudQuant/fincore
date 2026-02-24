"""
滚动指标分析示例

展示如何使用 fincore 计算和分析滚动指标：
- 滚动夏普比率
- 滚动波动率
- 滚动最大回撤
- 滚动 Alpha/Beta
- 滚动 Sortino 比率

适用场景：
- 策略稳定性分析
- 性能衰减检测
- 参数敏感性分析
- 风险动态监控
"""

import numpy as np
import pandas as pd
import numpy as np
import pandas as pd
from fincore import Empyrical
from fincore.metrics.rolling import (
    roll_sharpe_ratio,
    roll_max_drawdown,
    roll_beta,
)

# 滚动波动率辅助函数
def roll_annual_volatility(returns, window=252):
    """计算滚动年化波动率"""
    return returns.rolling(window).std() * np.sqrt(252)

# 滚动 Sortino 比率辅助函数
def roll_sortino_ratio(returns, window=252, risk_free=0.0):
    """计算滚动 Sortino 比率"""
    def sortino_r(rets):
        downside_rets = rets[rets < risk_free]
        if len(downside_rets) == 0:
            return np.nan
        excess_returns = rets - risk_free
        mean_excess = excess_returns.mean()
        downside_std = downside_rets.std()
        if downside_std == 0:
            return np.nan if mean_excess >= 0 else -np.inf
        return mean_excess / downside_std * np.sqrt(252)

    return returns.rolling(window).apply(sortino_r)

print("=" * 70)
print("滚动指标分析示例")
print("=" * 70)

# 生成模拟数据
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=252*3, freq="B", tz="UTC")

# 策略收益 - 带有一些周期性变化
trend = np.linspace(0.001, 0.0003, len(dates))
seasonal = 0.0005 * np.sin(np.linspace(0, 4*np.pi, len(dates)))
noise = np.random.normal(0, 0.015, len(dates))

strategy_returns = pd.Series(
    trend + seasonal + noise,
    index=dates,
    name="strategy"
)

# 基准收益
market_returns = pd.Series(
    np.random.normal(0.0005, 0.012, len(dates)),
    index=dates,
    name="market"
)

print(f"\n数据概览:")
print(f"  时间范围: {dates[0].date()} 至 {dates[-1].date()}")
print(f"  观测值数: {len(strategy_returns)}")

# ============================================================
# 1. 滚动夏普比率
# ============================================================
print("\n" + "=" * 70)
print("1. 滚动夏普比率")
print("=" * 70)

windows = [60, 126, 252]
print(f"\n不同窗口期的滚动夏普比率:")
print("-" * 50)

rolling_sharpe_summary = []
for window in windows:
    rs = roll_sharpe_ratio(strategy_returns, window=window)
    rolling_sharpe_summary.append({
        '窗口': window,
        '均值': rs.mean(),
        '标准差': rs.std(),
        '最小值': rs.min(),
        '最大值': rs.max(),
    })
    print(f"  窗口={window:3d}天: 均值={rs.mean():.4f}, 最小={rs.min():.4f}, 最大={rs.max():.4f}")

# 选择126天窗口进行详细分析
window = 126
rs_126 = roll_sharpe_ratio(strategy_returns, window=window)

print(f"\n滚动夏普比率 (窗口={window}天) 详细统计:")
print(f"  均值:     {rs_126.mean():.4f}")
print(f"  标准差:   {rs_126.std():.4f}")
print(f"  最小值:   {rs_126.min():.4f}")
print(f"  最大值:   {rs_126.max():.4f}")
print(f"  中位数:   {rs_126.median():.4f}")

# 夏普比率稳定性
sharpe_stable = (rs_126 > 0).sum() / len(rs_126) * 100
print(f"  正值占比: {sharpe_stable:.2f}%")

# ============================================================
# 2. 滚动波动率
# ============================================================
print("\n" + "=" * 70)
print("2. 滚动波动率")
print("=" * 70)

roll_vol = roll_annual_volatility(strategy_returns, window=window)

print(f"\n滚动年化波动率 (窗口={window}天):")
print(f"  均值:     {roll_vol.mean():.4f}")
print(f"  标准差:   {roll_vol.std():.4f}")
print(f"  最小值:   {roll_vol.min():.4f}")
print(f"  最大值:   {roll_vol.max():.4f}")

# 波动率区间分析
low_vol = (roll_vol < 0.15).sum()
med_vol = ((roll_vol >= 0.15) & (roll_vol < 0.25)).sum()
high_vol = (roll_vol >= 0.25).sum()

print(f"\n波动率区间分布:")
print(f"  低波动 (<15%):   {low_vol:>4} 天 ({low_vol/len(roll_vol)*100:.1f}%)")
print(f"  中波动 (15-25%): {med_vol:>4} 天 ({med_vol/len(roll_vol)*100:.1f}%)")
print(f"  高波动 (>25%):   {high_vol:>4} 天 ({high_vol/len(roll_vol)*100:.1f}%)")

# ============================================================
# 3. 滚动最大回撤
# ============================================================
print("\n" + "=" * 70)
print("3. 滚动最大回撤")
print("=" * 70)

roll_dd = roll_max_drawdown(strategy_returns, window=window)

print(f"\n滚动最大回撤 (窗口={window}天):")
print(f"  均值:     {roll_dd.mean():.4f}")
print(f"  标准差:   {roll_dd.std():.4f}")
print(f"  最小值:   {roll_dd.min():.4f}")
print(f"  最大值:   {roll_dd.max():.4f}")

# 回撤风险期间
high_dd_periods = (roll_dd < -0.10).sum()
print(f"\n严重回撤期间 (回撤>10%): {high_dd_periods} 天")

# ============================================================
# 4. 滚动 Sortino 比率
# ============================================================
print("\n" + "=" * 70)
print("4. 滚动 Sortino 比率")
print("=" * 70)

roll_sortino = roll_sortino_ratio(strategy_returns, window=window)

print(f"\n滚动 Sortino 比率 (窗口={window}天):")
print(f"  均值:     {roll_sortino.mean():.4f}")
print(f"  标准差:   {roll_sortino.std():.4f}")
print(f"  最小值:   {roll_sortino.min():.4f}")
print(f"  最大值:   {roll_sortino.max():.4f}")

# Sharpe vs Sortino 对比
print(f"\nSharpe vs Sortino (滚动值):")
print(f"  相关系数: {rs_126.corr(roll_sortino):.4f}")

# ============================================================
# 5. 滚动 Alpha/Beta
# ============================================================
print("\n" + "=" * 70)
print("5. 滚动 Alpha/Beta")
print("=" * 70)

roll_beta = roll_beta(strategy_returns, market_returns, window=window)

print(f"\n滚动 Beta (窗口={window}天):")
print(f"  均值:     {roll_beta.mean():.4f}")
print(f"  标准差:   {roll_beta.std():.4f}")
print(f"  最小值:   {roll_beta.min():.4f}")
print(f"  最大值:   {roll_beta.max():.4f}")

# Beta 稳定性分析
beta_range = roll_beta.max() - roll_beta.min()
print(f"\nBeta 波动范围: {beta_range:.4f}")

if beta_range < 0.3:
    print(f"  Beta 稳定性: 高")
elif beta_range < 0.5:
    print(f"  Beta 稳定性: 中")
else:
    print(f"  Beta 稳定性: 低")

# ============================================================
# 6. 滚动指标综合分析
# ============================================================
print("\n" + "=" * 70)
print("6. 滚动指标综合分析")
print("=" * 70)

# 计算多个滚动指标
metrics_summary = pd.DataFrame({
    'Sharpe': rs_126.describe(),
    'Volatility': roll_vol.describe(),
    'MaxDD': roll_dd.describe(),
    'Sortino': roll_sortino.describe(),
})

print(f"\n滚动指标统计汇总:")
print(metrics_summary.to_string())

# 相关性分析
corr_matrix = pd.DataFrame({
    'Sharpe': rs_126,
    'Volatility': roll_vol,
    'MaxDD': roll_dd,
    'Sortino': roll_sortino,
}).corr()

print(f"\n滚动指标相关性矩阵:")
print(corr_matrix.to_string())

# ============================================================
# 7. 性能衰减检测
# ============================================================
print("\n" + "=" * 70)
print("7. 性能衰减检测")
print("=" * 70)

# 将数据分为三个时期
n = len(rs_126)
p1_end = n // 3
p2_end = 2 * n // 3

rs_p1 = rs_126.iloc[:p1_end]
rs_p2 = rs_126.iloc[p1_end:p2_end]
rs_p3 = rs_126.iloc[p2_end:]

print(f"\nSharpe 比率分期对比:")
print(f"  早期 (前1/3): {rs_p1.mean():.4f}")
print(f"  中期 (中1/3): {rs_p2.mean():.4f}")
print(f"  近期 (后1/3): {rs_p3.mean():.4f}")

# 检测衰减
trend = np.polyfit(range(len(rs_126)), rs_126.values, 1)[0]
print(f"\nSharpe 趋势系数: {trend:.6f}")
if trend < -0.001:
    print(f"  结论: 策略性能有衰减趋势")
elif trend > 0.001:
    print(f"  结论: 策略性能有改善趋势")
else:
    print(f"  结论: 策略性能相对稳定")

# ============================================================
# 8. 窗口敏感性分析
# ============================================================
print("\n" + "=" * 70)
print("8. 窗口敏感性分析")
print("=" * 70)

print(f"\n不同窗口对滚动指标的影响:")

# 分析不同窗口下的均值和标准差
window_sensitivity = []
for w in [30, 60, 90, 126, 180, 252]:
    rs_w = roll_sharpe_ratio(strategy_returns, window=w)
    vol_w = roll_annual_volatility(strategy_returns, window=w)
    window_sensitivity.append({
        '窗口': w,
        'Sharpe均值': rs_w.mean(),
        'Sharpe标准差': rs_w.std(),
        'Vol均值': vol_w.mean(),
    })

sensitivity_df = pd.DataFrame(window_sensitivity)
print("\n" + sensitivity_df.to_string(index=False))

# ============================================================
# 9. 滚动指标分布分析
# ============================================================
print("\n" + "=" * 70)
print("9. 滚动指标分布分析")
print("=" * 70)

# Sharpe 比率分位数
sharpe_percentiles = np.percentile(rs_126.dropna(), [5, 25, 50, 75, 95])
print(f"\nSharpe 比率分位数:")
print(f"  5%:  {sharpe_percentiles[0]:.4f}")
print(f"  25%: {sharpe_percentiles[1]:.4f}")
print(f"  50%: {sharpe_percentiles[2]:.4f}")
print(f"  75%: {sharpe_percentiles[3]:.4f}")
print(f"  95%: {sharpe_percentiles[4]:.4f}")

# 波动率分位数
vol_percentiles = np.percentile(roll_vol.dropna(), [5, 25, 50, 75, 95])
print(f"\n波动率分位数:")
print(f"  5%:  {vol_percentiles[0]:.4f}")
print(f"  25%: {vol_percentiles[1]:.4f}")
print(f"  50%: {vol_percentiles[2]:.4f}")
print(f"  75%: {vol_percentiles[3]:.4f}")
print(f"  95%: {vol_percentiles[4]:.4f}")

# ============================================================
# 10. 可视化
# ============================================================
try:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. 滚动夏普比率
    ax = axes[0, 0]
    ax.plot(rs_126.index, rs_126.values, label='Sharpe', linewidth=1)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax.axhline(y=rs_126.mean(), color='red', linestyle='--',
               label=f'均值: {rs_126.mean():.2f}')
    ax.set_title(f'滚动夏普比率 ({window}天)')
    ax.set_ylabel('Sharpe')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. 滚动波动率
    ax = axes[0, 1]
    ax.plot(roll_vol.index, roll_vol.values * 100, label='波动率', linewidth=1, color='orange')
    ax.axhline(y=roll_vol.mean() * 100, color='red', linestyle='--',
               label=f'均值: {roll_vol.mean():.2%}')
    ax.set_title(f'滚动年化波动率 ({window}天)')
    ax.set_ylabel('波动率 (%)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. 滚动最大回撤
    ax = axes[1, 0]
    ax.fill_between(roll_dd.index, roll_dd.values * 100, 0,
                     alpha=0.3, color='red')
    ax.set_title(f'滚动最大回撤 ({window}天)')
    ax.set_ylabel('回撤 (%)')
    ax.grid(True, alpha=0.3)

    # 4. Sharpe vs Volatility 散点图
    ax = axes[1, 1]
    ax.scatter(roll_vol.values * 100, rs_126.values, alpha=0.3, s=10)
    ax.set_xlabel('波动率 (%)')
    ax.set_ylabel('Sharpe 比率')
    ax.set_title('风险-收益关系 (滚动)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('rolling_metrics.png', dpi=100)
    print("\n滚动指标可视化已保存: rolling_metrics.png")

except ImportError:
    print("\n未安装 matplotlib，跳过可视化")

print("\n" + "=" * 70)
print("滚动指标分析完成！")
print("=" * 70)
print("""
分析要点:
1. 滚动指标能反映策略的动态表现
2. 窗口选择影响指标的敏感度
3. 指标间的相关性揭示风险特征
4. 趋势分析检测性能衰减
5. 分位数分析了解极端情况

应用建议:
- 使用多个窗口期进行稳健性检验
- 关注滚动指标的趋势变化
- 结合静态和动态指标综合评估
""")
