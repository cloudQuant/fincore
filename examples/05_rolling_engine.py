"""
RollingEngine - 批量滚动指标计算示例

RollingEngine 是 fincore 提供的高性能批量滚动指标计算工具，
可以在一次调用中计算多个滚动指标，避免冗余迭代。

主要优势：
- 批量计算多个滚动指标，只需一次迭代
- 自动处理因子收益
- 支持的指标：sharpe, volatility, max_drawdown, beta, sortino, mean_return
"""

import numpy as np
import pandas as pd
from fincore.core.engine import RollingEngine
from fincore.metrics.rolling import roll_sharpe_ratio, roll_max_drawdown

# 生成示例数据
np.random.seed(42)
dates = pd.bdate_range('2020-01-01', periods=2520)
returns = pd.Series(np.random.normal(0.0005, 0.015, 2520), index=dates)
benchmark = pd.Series(np.random.normal(0.0003, 0.012, 2520), index=dates)

print("=" * 60)
print("RollingEngine 批量滚动指标计算示例")
print("=" * 60)
print(f"\n数据范围: {returns.index[0].date()} 到 {returns.index[-1].date()}")
print(f"数据点数: {len(returns)}")

# 传统方式：逐个计算滚动指标（效率较低）
print("\n" + "-" * 60)
print("传统方式：逐个计算滚动指标")
print("-" * 60)

window = 252
sharpe_rolling = roll_sharpe_ratio(returns, window)
dd_rolling = roll_max_drawdown(returns, window)

print(f"滚动 Sharpe 比率（最后一个值）: {sharpe_rolling.iloc[-1]:.4f}")
print(f"滚动最大回撤（最后一个值）: {dd_rolling.iloc[-1]:.4f}")

# 使用 RollingEngine：批量计算（效率更高）
print("\n" + "-" * 60)
print("RollingEngine：批量计算多个滚动指标")
print("-" * 60)

# 创建 RollingEngine
engine = RollingEngine(
    returns=returns,
    factor_returns=benchmark,
    window=252  # 滚动窗口大小
)

# 一次性计算多个滚动指标
results = engine.compute([
    'sharpe',       # Sharpe 比率
    'volatility',   # 波动率
    'max_drawdown', # 最大回撤
    'beta',         # Beta（相对于基准）
    'sortino',      # Sortino 比率
    'mean_return'   # 平均收益
])

# 将字典转换为 DataFrame 以便分析
results_df = pd.DataFrame(results)

print("\n计算结果（DataFrame）:")
print(results_df.head(10))
print(f"\n结果形状: {results_df.shape}")
print(f"包含指标: {list(results_df.columns)}")

# 分析结果
print("\n" + "-" * 60)
print("滚动指标统计摘要")
print("-" * 60)

for col in results_df.columns:
    print(f"\n{col}:")
    print(f"  均值:   {results_df[col].mean():.4f}")
    print(f"  标准差: {results_df[col].std():.4f}")
    print(f"  最小值: {results_df[col].min():.4f}")
    print(f"  最大值: {results_df[col].max():.4f}")

# 可视化（如果安装了 matplotlib）
try:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # Sharpe 比率
    results_df['sharpe'].plot(ax=axes[0], title='Rolling Sharpe Ratio (252-day window)')
    axes[0].axhline(y=0, color='red', linestyle='--', alpha=0.5)

    # 波动率
    results_df['volatility'].plot(ax=axes[1], title='Rolling Volatility (252-day window)')

    # 最大回撤
    results_df['max_drawdown'].plot(ax=axes[2], title='Rolling Max Drawdown (252-day window)')
    axes[2].fill_between(results_df.index, results_df['max_drawdown'], 0, alpha=0.3)

    plt.tight_layout()
    plt.savefig('rolling_metrics.png', dpi=100)
    print("\n图表已保存至: rolling_metrics.png")

except ImportError:
    print("\n未安装 matplotlib，跳过可视化")

print("\n" + "=" * 60)
print("RollingEngine 性能优势说明")
print("=" * 60)
print("""
RollingEngine 相比逐个计算的优势：

1. 减少迭代次数
   - 传统方式：计算 n 个指标需要 n 次完整遍历数据
   - RollingEngine：只需 1 次遍历计算所有指标

2. 缓存中间结果
   - 公共子表达式只计算一次
   - 滚动窗口数据复用

3. 自动处理因子收益
   - 当指定 factor_returns 时，beta 等指标自动使用
   - 无需手动对齐数据

4. 统一的输出格式
   - 所有指标返回为 DataFrame
   - 便于后续分析和可视化

适用场景：
- 多策略滚动指标对比
- 大规模回测中的性能指标监控
- 实时风控仪表盘数据准备
""")
