"""
完整量化工作流示例

展示一个完整的量化交易策略分析工作流：
- 数据获取与处理
- 策略回测
- 性能分析
- 风险评估
- 归因分析
- 报告生成

适用场景：
- 量化策略开发
- 完整分析流程参考
- 报告自动化
- 最佳实践演示
"""

import numpy as np
import pandas as pd
from fincore import analyze, sharpe_ratio, max_drawdown
from fincore.core.engine import RollingEngine
from fincore.report import create_strategy_report
from fincore.optimization import optimize
from fincore.simulation import MonteCarlo

print("=" * 70)
print("完整量化工作流示例")
print("=" * 70)

# ============================================================
# 第一步: 数据准备
# ============================================================
print("\n" + "=" * 70)
print("第一步: 数据准备")
print("=" * 70)

np.random.seed(42)
dates = pd.date_range("2023-01-01", periods=252*2, freq="B", tz="UTC")

# 模拟策略数据 (双均线策略信号)
price = 100 * np.cumprod(1 + np.random.normal(0.0005, 0.015, len(dates)))
price = pd.Series(price, index=dates)

# 计算移动平均线
short_ma = price.rolling(20).mean()
long_ma = price.rolling(60).mean()

# 生成交易信号
signals = np.where(short_ma > long_ma, 1, 0)
signals = pd.Series(signals, index=dates).shift(1).fillna(0)

# 计算收益率
price_returns = price.pct_change()
strategy_returns = (signals * price_returns).fillna(0)

# 基准收益
benchmark_returns = pd.Series(
    np.random.normal(0.0005, 0.012, len(dates)),
    index=dates
)

print(f"数据准备完成:")
print(f"  时间范围: {dates[0].date()} 至 {dates[-1].date()}")
print(f"  信号数量: {signals.sum()}")
print(f"  策略收益率: {len(strategy_returns)} 个交易日")

# ============================================================
# 第二步: 基本性能分析
# ============================================================
print("\n" + "=" * 70)
print("第二步: 基本性能分析")
print("=" * 70)

# 使用 AnalysisContext 进行分析
ctx = analyze(strategy_returns, factor_returns=benchmark_returns)

# 获取性能统计
perf_stats = ctx.perf_stats()

print(f"\n基本性能指标:")
print("-" * 50)
print(f"  年化收益:   {perf_stats.get('Annual return', 'N/A')}")
print(f"  累计收益:   {perf_stats.get('Cumulative returns', 'N/A')}")
print(f"  年化波动:   {perf_stats.get('Annual volatility', 'N/A')}")
print(f"  夏普比率:   {perf_stats.get('Sharpe ratio', 'N/A')}")
print(f"  最大回撤:   {perf_stats.get('Max drawdown', 'N/A')}")
print(f"  Calmar比率:  {perf_stats.get('Calmar ratio', 'N/A')}")

# ============================================================
# 第三步: 滚动指标分析
# ============================================================
print("\n" + "=" * 70)
print("第三步: 滚动指标分析")
print("=" * 70)

# 使用 RollingEngine 批量计算滚动指标
engine = RollingEngine(
    returns=strategy_returns,
    factor_returns=benchmark_returns,
    window=126  # 半年
)

rolling_results = engine.compute([
    'sharpe', 'volatility', 'max_drawdown', 'beta'
])

print(f"\n滚动指标统计 (窗口=126天):")
print("-" * 50)
for metric, values in rolling_results.items():
    if isinstance(values, pd.Series):
        print(f"  {metric}:")
        print(f"    均值:   {values.mean():.4f}")
        print(f"    标准差: {values.std():.4f}")
        print(f"    最小值: {values.min():.4f}")
        print(f"    最大值: {values.max():.4f}")

# ============================================================
# 第四步: 风险分析
# ============================================================
print("\n" + "=" * 70)
print("第四步: 风险分析")
print("=" * 70)

# VaR 和 CVaR
var_95 = np.percentile(strategy_returns, 5)
cvar_95 = strategy_returns[strategy_returns <= var_95].mean()

print(f"\n风险指标:")
print(f"  95% VaR (日度):  {var_95:.4f}")
print(f"  95% CVaR (日度): {cvar_95:.4f}")

# 回撤分析
drawdown_periods = ctx.gen_drawdown_table(top=3)
print(f"\n前3大回撤:")
print(drawdown_periods.to_string())

# ============================================================
# 第五步: 风险归因
# ============================================================
print("\n" + "=" * 70)
print("第五步: 风险归因")
print("=" * 70)

# Alpha 和 Beta
alpha = ctx.alpha()
beta = ctx.beta()

print(f"\n市场风险归因:")
print(f"  Alpha (年化):  {alpha * 252:.4f}")
print(f"  Beta:         {beta:.4f}")
print(f"  R-squared:    {ctx.r_squared():.4f}")

# 信息比率
ir = ctx.information_ratio()
print(f"  信息比率:     {ir:.4f}")

# 捕获比率
up_capture = ctx.up_capture()
down_capture = ctx.down_capture()
print(f"  上行捕获:     {up_capture:.2%}")
print(f"  下行捕获:     {down_capture:.2%}")

# ============================================================
# 第六步: 压力测试
# ============================================================
print("\n" + "=" * 70)
print("第六步: 压力测试")
print("=" * 70)

# 使用蒙特卡洛模拟进行压力测试
mc = MonteCarlo(strategy_returns)

# 模拟极端情景
scenarios = [
    ("基准", None, None),
    ("高波动", None, 0.025),
    ("负漂移", -0.0003, None),
]

print(f"\n压力测试结果 (1000路径, 252天):")
for name, drift, vol in scenarios:
    result = mc.simulate(n_paths=1000, horizon=252,
                        drift=drift, volatility=vol, seed=42)
    final_vals = result.paths[:, -1]
    print(f"  {name}:")
    print(f"    均值:   {final_vals.mean():.4f}")
    print(f"    5% VaR: {result.var(alpha=0.05):.4f}")

# ============================================================
# 第七步: 组合优化建议
# ============================================================
print("\n" + "=" * 70)
print("第七步: 组合优化建议")
print("=" * 70)

# 假设有多资产收益用于优化
np.random.seed(42)
n_assets = 5
asset_returns = pd.DataFrame(
    np.random.multivariate_normal(
        [0.0008, 0.0006, 0.0005, 0.0007, 0.0004],
        np.diag([0.015, 0.012, 0.018, 0.020, 0.014]),
        len(dates)
    ),
    index=dates,
    columns=['Asset1', 'Asset2', 'Asset3', 'Asset4', 'Asset5']
)

try:
    # 优化最大夏普组合
    opt_result = optimize(
        asset_returns,
        objective="max_sharpe",
        risk_free_rate=0.02,
        short_allowed=False,
        max_weight=0.4
    )

    print(f"\n优化组合建议:")
    print(f"  最优权重:")
    for asset, weight in zip(asset_returns.columns, opt_result['weights']):
        if weight > 0.001:
            print(f"    {asset}: {weight:.2%}")

    print(f"  预期夏普: {opt_result['sharpe']:.4f}")

except Exception as e:
    print(f"\n优化计算失败: {e}")

# ============================================================
# 第八步: 生成报告
# ============================================================
print("\n" + "=" * 70)
print("第八步: 生成分析报告")
print("=" * 70)

# 生成 HTML 报告
report_file = "workflow_report.html"
create_strategy_report(
    strategy_returns,
    benchmark_rets=benchmark_returns,
    title="量化策略分析报告",
    output=report_file
)
print(f"\n报告已生成: {report_file}")

# ============================================================
# 第九步: 综合评估
# ============================================================
print("\n" + "=" * 70)
print("第九步: 综合评估")
print("=" * 70)

# 策略评分
scores = {}

# 收益评分
ann_ret = strategy_returns.mean() * 252
scores['收益'] = min(100, max(0, (ann_ret + 0.1) * 500))

# 风险调整收益评分
sharpe = sharpe_ratio(strategy_returns)
scores['Sharpe'] = min(100, max(0, sharpe * 50))

# 回撤评分
max_dd = max_drawdown(strategy_returns)
scores['回撤'] = min(100, max(0, (1 + max_dd) * 100))

# 稳定性评分
sharpe_std = rolling_results['sharpe'].std()
scores['稳定性'] = min(100, max(0, 100 - sharpe_std * 200))

overall_score = np.mean(list(scores.values()))

print(f"\n策略评分 (0-100):")
for metric, score in scores.items():
    print(f"  {metric:<8}: {score:>6.1f}")
print(f"  {'综合':<8}: {overall_score:>6.1f}")

# 等级评定
if overall_score >= 80:
    grade = "A (优秀)"
elif overall_score >= 60:
    grade = "B (良好)"
elif overall_score >= 40:
    grade = "C (一般)"
else:
    grade = "D (较差)"

print(f"\n策略等级: {grade}")

# ============================================================
# 第十步: 可视化
# ============================================================
print("\n" + "=" * 70)
print("第十步: 可视化")
print("=" * 70)

try:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. 累计收益对比
    ax = axes[0, 0]
    ((1 + strategy_returns).cumprod() * 100).plot(ax=ax, label='策略')
    ((1 + benchmark_returns).cumprod() * 100).plot(ax=ax, label='基准')
    ax.set_ylabel('累计收益 (%)')
    ax.set_title('累计收益对比')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. 回撤
    ax = axes[0, 1]
    cum = (1 + strategy_returns).cumprod()
    running_max = cum.expanding().max()
    dd = (cum - running_max) / running_max
    ax.fill_between(dd.index, dd.values * 100, 0, alpha=0.3, color='red')
    ax.set_ylabel('回撤 (%)')
    ax.set_title('策略回撤')
    ax.grid(True, alpha=0.3)

    # 3. 滚动夏普
    ax = axes[1, 0]
    rolling_results['sharpe'].plot(ax=ax)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax.set_ylabel('Sharpe 比率')
    ax.set_title('滚动夏普比率')
    ax.grid(True, alpha=0.3)

    # 4. 评分雷达图
    ax = axes[1, 1]
    categories = list(scores.keys())
    values = list(scores.values())
    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]

    ax = plt.subplot(2, 2, 4, polar=True)
    ax.plot(angles, values, 'o-', linewidth=2)
    ax.fill(angles, values, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 100)
    ax.set_title('策略评分雷达图')

    plt.tight_layout()
    plt.savefig('workflow_analysis.png', dpi=100)
    print("\n可视化已保存: workflow_analysis.png")

except ImportError:
    print("\n未安装 matplotlib，跳过可视化")

# ============================================================
# 总结
# ============================================================
print("\n" + "=" * 70)
print("工作流总结")
print("=" * 70)

print("""
完成的步骤:
  ✓ 1. 数据准备 - 生成模拟策略数据
  ✓ 2. 基本性能分析 - 使用 AnalysisContext
  ✓ 3. 滚动指标分析 - 使用 RollingEngine
  ✓ 4. 风险分析 - VaR, CVaR, 回撤
  ✓ 5. 风险归因 - Alpha, Beta, 捕获比率
  ✓ 6. 压力测试 - 蒙特卡洛模拟
  ✓ 7. 组合优化 - 优化建议
  ✓ 8. 报告生成 - HTML 输出
  ✓ 9. 综合评估 - 策略评分
  ✓ 10. 可视化 - 图表输出

输出文件:
  - workflow_report.html  : 策略分析报告
  - workflow_analysis.png : 可视化图表

建议:
  1. 定期更新分析 (每月/季度)
  2. 监控滚动指标变化
  3. 进行压力测试评估
  4. 根据市场调整参数
""")
