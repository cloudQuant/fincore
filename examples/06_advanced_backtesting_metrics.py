"""
高级回测指标示例

展示如何使用 fincore 计算量化交易策略的高级指标，
包括回撤周期分析、连胜连败统计、月度热力图等。

适用场景：
- 交易策略回测分析
- 策略绩效报告生成
- 风险管理和仓位控制
"""

import numpy as np
import pandas as pd
from fincore import Empyrical

print("=" * 70)
print("高级回测指标分析示例")
print("=" * 70)

# 生成模拟交易策略收益数据
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=1260, freq="B", tz="UTC")

# 模拟一个有趋势但波动的策略
trend = np.linspace(0.0003, -0.0001, 1260)
noise = np.random.normal(0, 0.012, 1260)
returns = pd.Series(trend + noise, index=dates, name="strategy_returns")

# 计算累计收益
from fincore import cum_returns
cum_ret = cum_returns(returns, starting_value=1.0)

print(f"\n数据概览:")
print(f"  时间范围: {dates[0].date()} 至 {dates[-1].date()}")
print(f"  交易日数: {len(returns)}")
print(f"  累计收益: {cum_ret.iloc[-1]:.2%}")
print(f"" )

# ============================================================
# 1. 回撤分析
# ============================================================
print("-" * 70)
print("1. 回撤分析 (Drawdown Analysis)")
print("-" * 70)

# 最大回撤
max_dd = Empyrical.max_drawdown(returns)
print(f"最大回撤: {max_dd:.4f} ({max_dd*100:.2f}%)")

# 回撤周期
dd_period = Empyrical.get_max_drawdown_period(returns)
print(f"\n最大回撤周期:")
print(f"  回撤开始: {dd_period[0]}")
print(f"  回撤谷底: {dd_period[1]}")
# 检查是否有恢复日期
if len(dd_period) > 2:
    print(f"  恢复日期: {dd_period[2]}")
    print(f"  回撤天数: {(dd_period[2] - dd_period[0]).days} 天")
else:
    print(f"  状态: 尚未恢复")

# 回撤恢复天数
recovery_days = Empyrical.max_drawdown_recovery_days(returns)
print(f"\n回撤恢复天数: {recovery_days} 天")

# 第二大回撤
second_dd = Empyrical.second_max_drawdown(returns)
print(f"\n第二大回撤: {second_dd:.4f}")
# 第二大回撤的持续天数和恢复天数需要用单独的函数
second_dd_days = Empyrical.second_max_drawdown_days(returns)
second_dd_recovery = Empyrical.second_max_drawdown_recovery_days(returns)
print(f"  回撤天数: {second_dd_days} 天")
print(f"  恢复天数: {second_dd_recovery} 天")

# 回撤表
drawdown_table = Empyrical.gen_drawdown_table(returns, top=5)
print(f"\n前 5 大回撤:")
print(drawdown_table.to_string())

# ============================================================
# 2. 连胜连败统计
# ============================================================
print("\n" + "-" * 70)
print("2. 连胜连败统计 (Streak Analysis)")
print("-" * 70)

# 最大连胜/连败天数
max_up_days = Empyrical.max_consecutive_up_days(returns)
max_down_days = Empyrical.max_consecutive_down_days(returns)
print(f"最大连胜天数: {max_up_days} 天")
print(f"最大连败天数: {max_down_days} 天")

# 胜率/败率
win_rate = Empyrical.win_rate(returns)
loss_rate = Empyrical.loss_rate(returns)
print(f"胜率: {win_rate:.2%}")
print(f"败率: {loss_rate:.2%}")

# 最大单日收益/损失
max_gain = Empyrical.max_single_day_gain(returns)
max_gain_date = Empyrical.max_single_day_gain_date(returns)
max_loss = Empyrical.max_single_day_loss(returns)
max_loss_date = Empyrical.max_single_day_loss_date(returns)
print(f"\n最大单日收益: {max_gain:.4f} (日期: {max_gain_date.date() if max_gain_date else 'N/A'})")
print(f"最大单日损失: {max_loss:.4f} (日期: {max_loss_date.date() if max_loss_date else 'N/A'})")

# ============================================================
# 3. 年度/月度绩效分解
# ============================================================
print("\n" + "-" * 70)
print("3. 年度/月度绩效分解 (Performance Decomposition)")
print("-" * 70)

# 按年度分解
annual_returns = Empyrical.annual_return_by_year(returns)
print("\n年度收益率:")
for year, ret in annual_returns.items():
    print(f"  {year}: {ret:>8.2%}")

# 按月度分解
monthly_returns = Empyrical.aggregate_returns(returns, "monthly")
print(f"\n月度收益率 (前10 个):")
for i in range(min(10, len(monthly_returns))):
    year, month = monthly_returns.index[i]
    print(f"  {year}-{month:02d}: {monthly_returns.iloc[i]:>7.2%}")

# ============================================================
# 4. 分布统计
# ============================================================
print("\n" + "-" * 70)
print("4. 收益分布统计 (Distribution Statistics)")
print("-" * 70)

skew = Empyrical.skewness(returns)
kurtosis = Empyrical.kurtosis(returns)
print(f"偏度 (Skewness): {skew:.4f}")
print(f"峰度 (Kurtosis): {kurtosis:.4f}")

tail_ratio = Empyrical.tail_ratio(returns)
print(f"尾部比率 (95th/5th): {tail_ratio:.4f}")

stability = Empyrical.stability_of_timeseries(returns)
print(f"时间序列稳定性 (R²): {stability:.4f}")

# ============================================================
# 5. 风险调整收益比率汇总
# ============================================================
print("\n" + "-" * 70)
print("5. 风险调整收益比率 (Risk-Adjusted Ratios)")
print("-" * 70)

ratios = [
    ("Sharpe Ratio", Empyrical.sharpe_ratio(returns)),
    ("Sortino Ratio", Empyrical.sortino_ratio(returns)),
    ("Calmar Ratio", Empyrical.calmar_ratio(returns)),
    ("Omega Ratio", Empyrical.omega_ratio(returns)),
    ("Sterling Ratio", Empyrical.sterling_ratio(returns)),
    ("Burke Ratio", Empyrical.burke_ratio(returns)),
]

for name, value in ratios:
    print(f"  {name:16s}: {value:>10.4f}")

# ============================================================
# 6. 绩合绩效汇总
# ============================================================
print("\n" + "=" * 70)
print("综合绩效汇总 (Performance Summary)")
print("=" * 70)

stats = Empyrical.perf_stats(returns)
print(stats.to_string())

# ============================================================
# 7. 可视化（如果安装了 matplotlib）
# ============================================================
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 累计收益曲线
    ax = axes[0, 0]
    cum_ret.plot(ax=ax, title='累计收益曲线')
    ax.axhline(y=1, color='black', linestyle='--', alpha=0.3)
    ax.set_ylabel('累计收益')
    ax.grid(True, alpha=0.3)

    # 回撤曲线
    ax = axes[0, 1]
    underwater = cum_ret / cum_ret.cummax() - 1
    underwater.plot(ax=ax, title='回撤曲线', color='red')
    ax.fill_between(underwater.index, underwater, 0, alpha=0.3, color='red')
    ax.set_ylabel('回撤')
    ax.grid(True, alpha=0.3)

    # 月度收益热力图
    ax = axes[1, 0]
    monthly_ret_table = Empyrical.aggregate_returns(returns, "monthly")
    monthly_ret_table = monthly_ret_table.to_frame()
    monthly_ret_table.columns = ['Returns']

    # 重塑为年x月的矩阵
    monthly_matrix = monthly_ret_table['Returns'].unstack()
    if not monthly_matrix.empty:
        im = ax.imshow(monthly_matrix.T, cmap='RdYlGn', aspect='auto')
        ax.set_title('月度收益热力图')
        ax.set_ylabel('月份')
        ax.set_xlabel('年份')

        # 设置 y 轴标签
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        ax.set_yticks(range(12))
        ax.set_yticklabels(months)

        plt.colorbar(im, ax=ax, label='收益率')
        ax.grid(False)

    # 滚动 Sharpe 比率
    ax = axes[1, 1]
    from fincore.metrics.rolling import roll_sharpe_ratio
    rolling_sharpe = roll_sharpe_ratio(returns, window=60)
    rolling_sharpe.plot(ax=ax, title='滚动 Sharpe 比率 (60日)')
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax.set_ylabel('Sharpe 比率')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('backtesting_metrics.png', dpi=100)
    print("\n可视化图表已保存: backtesting_metrics.png")

except ImportError:
    print("\n未安装 matplotlib，跳过可视化")

print("\n" + "=" * 70)
print("高级回测指标分析完成！")
print("=" * 70)
