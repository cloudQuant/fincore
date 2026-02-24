"""
压力测试与极端情景分析示例

展示如何使用 fincore 进行压力测试：
- 历史极端事件分析
- 情景压力测试
- 尾部风险分析
- 投资组合压力测试

适用场景：
- 风险管理
- 极端情况评估
- 投资组合稳健性检验
- 监管报告
"""

import numpy as np
import pandas as pd
from fincore import Empyrical
from fincore.simulation import MonteCarlo
from fincore.constants.interesting_periods import PERIODS

print("=" * 70)
print("压力测试与极端情景分析示例")
print("=" * 70)

# 生成模拟策略数据
np.random.seed(42)
dates = pd.date_range("2018-01-01", periods=252*5, freq="B", tz="UTC")

# 模拟策略收益
strategy_returns = pd.Series(
    np.random.normal(0.0006, 0.012, len(dates)),
    index=dates,
    name="strategy"
)

# 模拟基准收益 (市场)
market_returns = pd.Series(
    np.random.normal(0.0004, 0.010, len(dates)),
    index=dates,
    name="market"
)

print(f"\n数据概览:")
print(f"  时间范围: {dates[0].date()} 至 {dates[-1].date()}")
print(f"  观测值数: {len(strategy_returns)}")

# 基本统计
print(f"\n策略基本统计:")
print(f"  年化收益: {strategy_returns.mean() * 252:.4f}")
print(f"  年化波动: {strategy_returns.std() * np.sqrt(252):.4f}")
print(f"  夏普比率: {Empyrical.sharpe_ratio(strategy_returns):.4f}")
print(f"  最大回撤: {Empyrical.max_drawdown(strategy_returns):.4f}")

# ============================================================
# 1. 历史极端事件分析
# ============================================================
print("\n" + "=" * 70)
print("1. 历史极端事件分析")
print("=" * 70)

# 显示可用的历史事件期间
print(f"\n预定义的历史极端事件:")
print("-" * 50)
for event_name, (start, end) in list(PERIODS.items())[:10]:
    print(f"  {event_name:<30} {start.date()} ~ {end.date()}")

# 分析策略在这些历史期间的表现
print(f"\n策略在历史极端事件期间的表现 (假设覆盖期间):")
print("-" * 70)
print(f"{'事件':<25} {'收益':>12} {'波动率':>12} {'最大回撤':>12}")
print("-" * 70)

# 检查哪些事件期间在数据范围内
events_in_range = []
for event_name, (start, end) in PERIODS.items():
    # 转换为带时区的日期
    start_tz = pd.Timestamp(start).tz_localize('UTC')
    end_tz = pd.Timestamp(end).tz_localize('UTC')
    # 找到该期间的数据
    mask = (strategy_returns.index >= start_tz) & (strategy_returns.index <= end_tz)
    if mask.sum() > 0:
        event_returns = strategy_returns[mask]
        ret = event_returns.sum()
        vol = event_returns.std() * np.sqrt(252)
        dd = Empyrical.max_drawdown(event_returns)
        print(f"{event_name:<25} {ret:>10.2%} {vol:>10.2%} {dd:>10.2%}")
        events_in_range.append(event_name)

if not events_in_range:
    print("  (数据未覆盖任何预定义的历史事件期间)")

# ============================================================
# 2. 自定义情景压力测试
# ============================================================
print("\n" + "=" * 70)
print("2. 自定义情景压力测试")
print("=" * 70)

# 定义压力情景
scenarios = {
    "温和下跌": {"days": 5, "daily_mean": -0.005, "daily_std": 0.015},
    "急剧下跌": {"days": 5, "daily_mean": -0.02, "daily_std": 0.03},
    "持续下跌": {"days": 20, "daily_mean": -0.008, "daily_std": 0.02},
    "极度波动": {"days": 10, "daily_mean": 0, "daily_std": 0.04},
    "崩盘情景": {"days": 3, "daily_mean": -0.05, "daily_std": 0.05},
}

print(f"\n压力情景测试:")
print("-" * 60)
print(f"{'情景':<12} {'天数':>6} {'日均值':>10} {'日波动':>10} {'预期收益':>12} {'VaR(5%)':>10}")
print("-" * 60)

for name, params in scenarios.items():
    # 模拟该情景下的收益
    np.random.seed(42)
    scenario_returns = np.random.normal(
        params["daily_mean"],
        params["daily_std"],
        params["days"]
    )

    total_return = scenario_returns.sum()
    var_5 = np.percentile(scenario_returns, 5)

    print(f"{name:<12} {params['days']:>6} "
          f"{params['daily_mean']:>9.2%} {params['daily_std']:>9.2%} "
          f"{total_return:>11.2%} {var_5:>9.2%}")

# ============================================================
# 3. 尾部风险分析
# ============================================================
print("\n" + "=" * 70)
print("3. 尾部风险分析")
print("=" * 70)

# 分析策略的最差表现日
worst_days = strategy_returns.nsmallest(20)
print(f"\n最差20个交易日:")
print("-" * 40)
for i, (date, ret) in enumerate(worst_days.items(), 1):
    print(f"  {i:2d}. {date.date()}: {ret:>7.2%}")

# 尾部风险统计
print(f"\n尾部风险统计:")
print(f"  1% 分位数 (日度):  {np.percentile(strategy_returns, 1):.4f}")
print(f"  5% 分位数 (日度):  {np.percentile(strategy_returns, 5):.4f}")
print(f"  10% 分位数 (日度): {np.percentile(strategy_returns, 10):.4f}")

# 条件VaR (CVaR) - 平均损失
var_5 = np.percentile(strategy_returns, 5)
cvar_5 = strategy_returns[strategy_returns <= var_5].mean()
print(f"  CVaR (5%):         {cvar_5:.4f}")

# 尾部比率
tail_ratio = Empyrical.tail_ratio(strategy_returns)
print(f"  尾部比率 (95/5):   {tail_ratio:.4f}")

# ============================================================
# 4. 蒙特卡洛压力测试
# ============================================================
print("\n" + "=" * 70)
print("4. 蒙特卡洛压力测试")
print("=" * 70)

mc = MonteCarlo(strategy_returns)

# 模拟极端情景
print(f"\n蒙特卡洛极端情景模拟 (1000条路径, 252天):")
print("-" * 50)

scenarios_mc = [
    ("基准", None, None),
    ("高波动", None, 0.025),
    ("负漂移", -0.0005, None),
    ("极端情景", -0.001, 0.03),
]

for name, drift, vol in scenarios_mc:
    result = mc.simulate(
        n_paths=1000,
        horizon=252,
        drift=drift,
        volatility=vol,
        seed=42
    )

    final_vals = result.paths[:, -1]
    print(f"\n  {name}:")
    if drift is not None:
        print(f"    漂移率: {drift}")
    if vol is not None:
        print(f"    波动率: {vol}")

    print(f"    均值:   {final_vals.mean():.4f}")
    print(f"    中位数: {np.median(final_vals):.4f}")
    print(f"    5% VaR: {result.var(alpha=0.05):.4f}")
    print(f"    最小值: {final_vals.min():.4f}")

# ============================================================
# 5. 回撤恢复分析
# ============================================================
print("\n" + "=" * 70)
print("5. 回撤恢复分析")
print("=" * 70)

# 计算回撤周期
drawdown_periods = Empyrical.gen_drawdown_table(strategy_returns, top=5)
print(f"\n前5大回撤周期:")
print(drawdown_periods.to_string())

# 最大回撤恢复天数
recovery_days = Empyrical.max_drawdown_recovery_days(strategy_returns)
print(f"\n最大回撤恢复天数: {recovery_days} 个交易日")

# 估算恢复概率 (假设当前处于最大回撤状态)
current_dd = Empyrical.max_drawdown(strategy_returns.tail(60))
print(f"最近60天最大回撤: {current_dd:.2%}")

# 简单恢复概率计算
historical_recoveries = []
for i in range(60, len(strategy_returns) - 60):
    past_returns = strategy_returns.iloc[i-60:i]
    dd = Empyrical.max_drawdown(past_returns)
    if dd < -0.05:  # 5%以上回撤
        future_returns = strategy_returns.iloc[i:i+60]
        recovered = future_returns.cumsum().max() >= abs(dd)
        historical_recoveries.append(recovered)

if historical_recoveries:
    recovery_prob = np.mean(historical_recoveries) * 100
    print(f"基于历史的恢复概率: {recovery_prob:.1f}%")

# ============================================================
# 6. 相关性崩溃压力测试
# ============================================================
print("\n" + "=" * 70)
print("6. 相关性崩溃压力测试")
print("=" * 70)

# 正常情况下的相关性
normal_corr = strategy_returns.rolling(60).corr(market_returns)
avg_normal_corr = normal_corr.mean()
print(f"\n正常市场环境下与市场的相关性: {avg_normal_corr:.4f}")

# 模拟相关性崩溃 (市场剧烈下跌时)
market_stress = market_returns < market_returns.quantile(0.05)
if market_stress.sum() > 10:
    stress_corr = strategy_returns[market_stress].corr(market_returns[market_stress])
    print(f"市场压力下的相关性: {stress_corr:.4f}")
    print(f"相关性变化: {stress_corr - avg_normal_corr:+.4f}")
else:
    # 模拟压力情景
    print("\n模拟相关性崩溃情景:")
    print("  假设市场下跌10%时，策略相关性增加至0.95")
    print("  策略将无法通过分散化降低风险")

# ============================================================
# 7. 综合压力测试报告
# ============================================================
print("\n" + "=" * 70)
print("7. 综合压力测试报告")
print("=" * 70)

# 计算压力测试指标
print(f"\n压力测试汇总:")

# 1. 最大损失
max_loss = strategy_returns.min()
max_loss_date = strategy_returns.idxmin()
print(f"  单日最大损失:     {max_loss:.2%} ({max_loss_date.date()})")

# 2. 最大回撤
max_dd = Empyrical.max_drawdown(strategy_returns)
print(f"  最大回撤:         {max_dd:.2%}")

# 3. 连续损失
max_consecutive_losses = Empyrical.max_consecutive_down_days(strategy_returns)
print(f"  最长连续损失天数: {max_consecutive_losses}")

# 4. 尾部风险
print(f"  1% VaR (日度):    {np.percentile(strategy_returns, 1):.2%}")
print(f"  5% CVaR (日度):   {cvar_5:.2%}")

# 5. 波动率飙升
rolling_vol = strategy_returns.rolling(20).std() * np.sqrt(252)
max_vol = rolling_vol.max()
print(f"  最大20日波动率:   {max_vol:.2%}")

# 风险等级评估
print(f"\n风险等级评估:")
risk_score = 0
if max_dd < -0.10:
    risk_score += 2
if max_dd < -0.20:
    risk_score += 2
if abs(max_loss) > 0.05:
    risk_score += 1
if abs(max_loss) > 0.08:
    risk_score += 1
if cvar_5 < -0.03:
    risk_score += 2
if max_vol > 0.25:
    risk_score += 2

if risk_score >= 8:
    level = "高风险"
elif risk_score >= 5:
    level = "中等风险"
elif risk_score >= 3:
    level = "低风险"
else:
    level = "很低风险"

print(f"  综合风险评分: {risk_score}/10")
print(f"  风险等级:     {level}")

# ============================================================
# 8. 可视化
# ============================================================
try:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. 累计收益与回撤
    ax = axes[0, 0]
    cum_returns = (1 + strategy_returns).cumprod()
    ax.plot(cum_returns.index, cum_returns.values * 100, label='策略')
    ax.set_ylabel('累计收益 (%)')
    ax.set_title('策略累计收益')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. 收益分布
    ax = axes[0, 1]
    ax.hist(strategy_returns * 100, bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(x=np.percentile(strategy_returns, 5)*100, color='red',
               linestyle='--', label='5% 分位数')
    ax.axvline(x=np.percentile(strategy_returns, 95)*100, color='green',
               linestyle='--', label='95% 分位数')
    ax.set_xlabel('日收益率 (%)')
    ax.set_ylabel('频数')
    ax.set_title('收益分布')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. 滚动波动率
    ax = axes[1, 0]
    rolling_vol_plot = strategy_returns.rolling(60).std() * np.sqrt(252) * 100
    ax.plot(rolling_vol_plot.index, rolling_vol_plot.values)
    ax.axhline(y=rolling_vol_plot.mean(), color='red', linestyle='--',
               label=f'平均: {rolling_vol_plot.mean():.1f}%')
    ax.set_ylabel('年化波动率 (%)')
    ax.set_title('滚动波动率 (60日)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. 尾部收益
    ax = axes[1, 1]
    worst_100 = strategy_returns.nsmallest(100)
    ax.bar(range(100), worst_100 * 100, edgecolor='black', alpha=0.7)
    ax.axhline(y=np.percentile(strategy_returns, 5)*100, color='red',
               linestyle='--', label='5% 分位数')
    ax.set_xlabel('排名')
    ax.set_ylabel('日收益率 (%)')
    ax.set_title('最差100个交易日')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('stress_testing.png', dpi=100)
    print("\n压力测试可视化已保存: stress_testing.png")

except ImportError:
    print("\n未安装 matplotlib，跳过可视化")

print("\n" + "=" * 70)
print("压力测试分析完成！")
print("=" * 70)
print("""
提示：
1. 压力测试用于评估投资组合在极端市场条件下的表现
2. 历史极端事件分析可以参考过去的市场危机
3. 自定义情景可以测试特定的风险因素
4. 尾部风险分析关注极端损失的分布
5. 定期进行压力测试有助于风险管理
""")
