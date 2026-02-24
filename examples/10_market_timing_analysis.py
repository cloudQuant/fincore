"""
择时能力分析示例

展示如何使用 fincore 分析策略的择时能力：
- Treynor-Mazuy 择时模型
- Henriksson-Merton 择时模型
- 上下行捕获比率
- 牛熊市分析

适用场景：
- 评估基金经理的择时能力
- 分析策略在牛熊市中的表现
- 识别市场时机把握能力
- 绩效归因分解
"""

import numpy as np
import pandas as pd
from fincore import Empyrical
from fincore.metrics import alpha_beta, timing

print("=" * 70)
print("择时能力分析示例")
print("=" * 70)

# 生成模拟数据：策略收益 + 市场基准
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=252*3, freq="B", tz="UTC")
n = len(dates)

# 市场收益 (基准)
market_returns = pd.Series(
    np.random.normal(0.0004, 0.012, n),
    index=dates,
    name="market"
)

# 策略收益：有一定择时能力 (在市场上涨时超额表现)
strategy_returns = pd.Series(
    0.0003 + 0.8 * market_returns.values +
    0.5 * (market_returns.values ** 2) +  # 择时能力
    np.random.normal(0, 0.008, n),
    index=dates,
    name="strategy"
)

print(f"\n数据概览:")
print(f"  时间范围: {dates[0].date()} 至 {dates[-1].date()}")
print(f"  观测值数: {len(strategy_returns)}")

# 基本统计
print(f"\n基本统计:")
print(f"  {'策略':<10} 年化收益: {strategy_returns.mean() * 252:>8.2%}")
print(f"  {'市场':<10} 年化收益: {market_returns.mean() * 252:>8.2%}")
print(f"  {'策略':<10} 年化波动: {strategy_returns.std() * np.sqrt(252):>8.2%}")
print(f"  {'市场':<10} 年化波动: {market_returns.std() * np.sqrt(252):>8.2%}")

# ============================================================
# 1. Alpha 和 Beta 分析
# ============================================================
print("\n" + "=" * 70)
print("1. Alpha 和 Beta 分析")
print("=" * 70)

alpha, beta = Empyrical.alpha_beta(strategy_returns, market_returns)
print(f"\n标准回归:")
print(f"  Alpha (年化):  {alpha * 252:>8.4f}")
print(f"  Beta:          {beta:>8.4f}")

# 年化 alpha
ann_alpha = Empyrical.alpha(strategy_returns, market_returns)
print(f"  Alpha (年化):  {ann_alpha:>8.4f}")

# ============================================================
# 2. Treynor-Mazuy 择时模型
# ============================================================
print("\n" + "=" * 70)
print("2. Treynor-Mazuy 择时模型")
print("=" * 70)

tm_timing = timing.treynor_mazuy_timing(strategy_returns, market_returns)
print(f"\nTreynor-Mazuy 择时系数 (gamma): {tm_timing:.6f}")
print(f"  说明: 该系数衡量策略管理人的市场时机把握能力")
print(f"        > 0 表示正择时能力 (市场上涨时超额收益)")
print(f"        < 0 表示负择时能力")
print(f"        ≈ 0 表示无择时能力")

if tm_timing > 0.01:
    print(f"  结论: 策略显示出良好的择时能力")
elif tm_timing < -0.01:
    print(f"  结论: 策略择时能力较弱")
else:
    print(f"  结论: 策略择时能力不明显")

# ============================================================
# 3. Henriksson-Merton 择时模型
# ============================================================
print("\n" + "=" * 70)
print("3. Henriksson-Merton 择时模型")
print("=" * 70)

hm_timing = timing.henriksson_merton_timing(strategy_returns, market_returns)
print(f"\nHenriksson-Merton 择时系数: {hm_timing:.6f}")
print(f"  说明: 该模型通过二元回归评估择时能力")
print(f"        > 0 表示成功预测市场方向")
print(f"        < 0 表示市场时机把握不佳")

if hm_timing > 0:
    print(f"  结论: 管理人具有成功预测市场方向的能力")
else:
    print(f"  结论: 管理人择时能力有待提高")

# ============================================================
# 4. 上下行捕获比率
# ============================================================
print("\n" + "=" * 70)
print("4. 上下行捕获比率")
print("=" * 70)

# 计算上行和下行市场收益
up_alpha, up_beta = Empyrical.up_alpha_beta(strategy_returns, market_returns)
down_alpha, down_beta = Empyrical.down_alpha_beta(strategy_returns, market_returns)

print(f"\n上行市场 (市场上涨时):")
print(f"  Up Beta:      {up_beta:.4f}")
print(f"  Up Alpha:     {up_alpha * 252:.4f}")

print(f"\n下行市场 (市场下跌时):")
print(f"  Down Beta:    {down_beta:.4f}")
print(f"  Down Alpha:   {down_alpha * 252:.4f}")

# 捕获比率
up_capture = Empyrical.up_capture(strategy_returns, market_returns)
down_capture = Empyrical.down_capture(strategy_returns, market_returns)
up_down_capture = Empyrical.up_down_capture(strategy_returns, market_returns)

print(f"\n捕获比率:")
print(f"  上行捕获率:   {up_capture:.2%}")
print(f"  下行捕获率:   {down_capture:.2%}")
print(f"  上下行比率:   {up_down_capture:.2f}")

print(f"\n捕获比率解读:")
print(f"  上行捕获率 > 100%: 策略在上涨市场中表现优于市场")
print(f"  下行捕获率 < 100%: 策略在下跌市场中损失小于市场")
print(f"  上下行比率 > 1: 整体风险调整后表现优异")

# ============================================================
# 5. 牛熊市分段分析
# ============================================================
print("\n" + "=" * 70)
print("5. 牛熊市分段分析")
print("=" * 70)

# 识别牛熊市周期
market_cum = (1 + market_returns).cumprod()
# 简单牛熊市划分: 以累计收益高低点划分
rolling_max = market_cum.expanding().max()
drawdown = (market_cum - rolling_max) / rolling_max

# 找出主要牛市和熊市期间
is_bull_market = market_returns > 0
is_bear_market = market_returns < 0

bull_strategy = strategy_returns[is_bull_market]
bear_strategy = strategy_returns[is_bear_market]
bull_market = market_returns[is_bull_market]
bear_market = market_returns[is_bear_market]

print(f"\n牛市期间统计 (市场上涨日):")
print(f"  天数: {is_bull_market.sum()}")
print(f"  策略平均日收益: {bull_strategy.mean():.4f}")
print(f"  市场平均日收益: {bull_market.mean():.4f}")
print(f"  策略累计收益: {(1 + bull_strategy).prod() - 1:.4f}")

print(f"\n熊市期间统计 (市场下跌日):")
print(f"  天数: {is_bear_market.sum()}")
print(f"  策略平均日收益: {bear_strategy.mean():.4f}")
print(f"  市场平均日收益: {bear_market.mean():.4f}")
print(f"  策略累计收益: {(1 + bear_strategy).prod() - 1:.4f}")

# ============================================================
# 6. 择时收益分解
# ============================================================
print("\n" + "=" * 70)
print("6. 择时收益分解")
print("=" * 70)

# 计算择时对收益的贡献
market_timing_return = timing.market_timing_return(
    strategy_returns, market_returns
)
print(f"\n市场择时收益: {market_timing_return:.4f}")
print(f"  说明: 该指标量化择时决策对策略收益的贡献")

# Cornell 择时指标
try:
    cornell = timing.cornell_timing(strategy_returns, market_returns)
    print(f"\nCornell 择时指标: {cornell:.4f}")
except Exception as e:
    print(f"\nCornell 择时指标: 计算失败 ({e})")

# R-squared 指标
try:
    r_squared = timing.r_cubed(strategy_returns, market_returns)
    print(f"R-squared: {r_squared:.4f}")
except Exception as e:
    print(f"R-squared: 计算失败 ({e})")

# ============================================================
# 7. 综合择时评估
# ============================================================
print("\n" + "=" * 70)
print("7. 综合择时能力评估")
print("=" * 70)

# 计算综合得分
timing_score = 0
max_score = 0

# Treynor-Mazuy (权重: 30%)
max_score += 30
if tm_timing > 0.05:
    timing_score += 30
elif tm_timing > 0:
    timing_score += 15

# Henriksson-Merton (权重: 30%)
max_score += 30
if hm_timing > 0.01:
    timing_score += 30
elif hm_timing > 0:
    timing_score += 15

# 捕获比率 (权重: 40%)
max_score += 40
if up_down_capture > 1.2:
    timing_score += 40
elif up_down_capture > 1.0:
    timing_score += 20
elif up_down_capture > 0.8:
    timing_score += 10

timing_grade = timing_score / max_score * 100

print(f"\n择时能力综合评分: {timing_score}/{max_score} ({timing_grade:.1f}分)")
print(f"\n评分明细:")
print(f"  Treynor-Mazuy 择时:   {'+' if tm_timing > 0 else ''}{tm_timing:.4f}")
print(f"  Henriksson-Merton 择时: {'+' if hm_timing > 0 else ''}{hm_timing:.4f}")
print(f"  上下行捕获比率:       {up_down_capture:.2f}")

print(f"\n择时能力等级:")
if timing_grade >= 80:
    print(f"  优秀 - 策略表现出色的择时能力")
elif timing_grade >= 60:
    print(f"  良好 - 策略具有一定的择时能力")
elif timing_grade >= 40:
    print(f"  一般 - 策略择时能力有限")
else:
    print(f"  较弱 - 策略择时能力不足")

# ============================================================
# 8. 可视化
# ============================================================
try:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. 累计收益对比
    ax = axes[0, 0]
    ((1 + strategy_returns).cumprod() * 100).plot(ax=ax, label='策略')
    ((1 + market_returns).cumprod() * 100).plot(ax=ax, label='市场')
    ax.set_title('累计收益对比')
    ax.set_ylabel('收益 (%)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. 散点图 (策略 vs 市场)
    ax = axes[0, 1]
    ax.scatter(market_returns, strategy_returns, alpha=0.3, s=10)
    # 添加拟合线
    z = np.polyfit(market_returns, strategy_returns, 1)
    p = np.poly1d(z)
    x_line = np.linspace(market_returns.min(), market_returns.max(), 100)
    ax.plot(x_line, p(x_line), "r--", alpha=0.8, label=f'拟合线 (β={z[0]:.2f})')
    ax.set_title('策略收益 vs 市场收益')
    ax.set_xlabel('市场收益')
    ax.set_ylabel('策略收益')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. 滚动 Beta
    ax = axes[1, 0]
    window = 126  # 半年
    rolling_alpha = []
    rolling_beta = []
    for i in range(window, len(strategy_returns)):
        a, b = Empyrical.alpha_beta(
            strategy_returns.iloc[i-window:i],
            market_returns.iloc[i-window:i]
        )
        rolling_alpha.append(a)
        rolling_beta.append(b)

    pd.Series(rolling_beta, index=strategy_returns.index[window:]).plot(ax=ax)
    ax.axhline(y=beta, color='r', linestyle='--', label=f'总体 Beta={beta:.2f}')
    ax.set_title(f'滚动 Beta (窗口={window}天)')
    ax.set_ylabel('Beta')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. 牛熊市收益对比
    ax = axes[1, 1]
    categories = ['牛市期间', '熊市期间']
    strategy_means = [bull_strategy.mean() * 100, bear_strategy.mean() * 100]
    market_means = [bull_market.mean() * 100, bear_market.mean() * 100]

    x = np.arange(len(categories))
    width = 0.35

    ax.bar(x - width/2, strategy_means, width, label='策略', alpha=0.8)
    ax.bar(x + width/2, market_means, width, label='市场', alpha=0.8)
    ax.set_ylabel('平均日收益 (%)')
    ax.set_title('牛熊市平均收益对比')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('market_timing_analysis.png', dpi=100)
    print("\n择时能力分析可视化已保存: market_timing_analysis.png")

except ImportError:
    print("\n未安装 matplotlib，跳过可视化")

print("\n" + "=" * 70)
print("择时能力分析完成！")
print("=" * 70)
print("""
提示：
1. 择时能力评估策略管理人对市场方向的预测能力
2. Treynor-Mazuy 模型通过二次回归检测择时能力
3. Henriksson-Merton 模型使用虚拟变量检测择时
4. 捕获比率显示策略在牛熊市中的相对表现
5. 综合评分结合多个指标给出整体评估
""")
