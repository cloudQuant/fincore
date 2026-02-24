"""
自定义组合优化示例

展示如何使用 fincore 的优化模块进行自定义组合优化：
- 目标收益优化
- 目标风险优化
- 最小方差组合
- 行业约束优化
- 卖空约束

适用场景：
- 投资组合构建
- 约束优化问题
- 风险预算管理
- 行业配置决策
"""

import numpy as np
import pandas as pd
from fincore.optimization import optimize
from fincore import sharpe_ratio, annual_volatility

print("=" * 70)
print("自定义组合优化示例")
print("=" * 70)

# 生成多资产收益数据
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=252*3, freq="B", tz="UTC")
n_assets = 8

assets = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN',  # 科技
    'JPM', 'BAC', 'WFC', 'C',           # 金融
]

# 模拟收益 - 科技股更高波动和收益
mean_returns = np.array([
    0.0010, 0.0008, 0.0007, 0.0009,  # 科技
    0.0004, 0.0003, 0.0002,           # 金融
])

# 协方差矩阵 - 8x8 对应 8 个资产
cov_matrix = np.array([
    [0.020, 0.012, 0.010, 0.011, 0.005, 0.004, 0.003, 0.003],
    [0.012, 0.018, 0.009, 0.010, 0.004, 0.003, 0.002, 0.002],
    [0.010, 0.009, 0.016, 0.009, 0.004, 0.003, 0.002, 0.002],
    [0.011, 0.010, 0.009, 0.022, 0.005, 0.004, 0.003, 0.003],
    [0.005, 0.004, 0.004, 0.005, 0.012, 0.008, 0.006, 0.005],
    [0.004, 0.003, 0.003, 0.004, 0.008, 0.010, 0.005, 0.004],
    [0.003, 0.002, 0.002, 0.003, 0.006, 0.005, 0.009, 0.004],
    [0.003, 0.002, 0.002, 0.003, 0.005, 0.004, 0.004, 0.008],
])

# 调整 mean_returns 为 8 个资产
mean_returns = np.array([
    0.0010, 0.0008, 0.0007, 0.0009,  # 科技 (4个)
    0.0004, 0.0003, 0.0002, 0.0002,   # 金融 (4个)
])

# 生成收益数据
returns = pd.DataFrame(
    np.random.multivariate_normal(mean_returns, cov_matrix, len(dates)),
    index=dates,
    columns=assets
)

# 行业映射
sector_map = {
    'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology', 'AMZN': 'Technology',
    'JPM': 'Financials', 'BAC': 'Financials', 'WFC': 'Financials', 'C': 'Financials',
}

print(f"\n数据概览:")
print(f"  资产数量: {n_assets}")
print(f"  时间范围: {dates[0].date()} 至 {dates[-1].date()}")

# 各资产统计
print(f"\n各资产年化收益与波动:")
print("-" * 50)
print(f"{'资产':<10} {'年化收益':>12} {'年化波动':>12} {'Sharpe':>10}")
print("-" * 50)

for asset in assets:
    ann_ret = returns[asset].mean() * 252
    ann_vol = returns[asset].std() * np.sqrt(252)
    sharpe = sharpe_ratio(returns[asset])
    print(f"{asset:<10} {ann_ret:>10.2%} {ann_vol:>10.2%} {sharpe:>10.4f}")

# ============================================================
# 1. 最大夏普比率组合
# ============================================================
print("\n" + "=" * 70)
print("1. 最大夏普比率组合")
print("=" * 70)

result_max_sharpe = optimize(
    returns,
    objective="max_sharpe",
    risk_free_rate=0.02,
    short_allowed=False,
    max_weight=0.4,  # 单个资产最大权重40%
)

print(f"\n优化结果:")
print(f"  最优权重:")
for asset, weight in zip(assets, result_max_sharpe['weights']):
    if weight > 0.001:
        print(f"    {asset:<10} {weight:>10.2%}")

print(f"\n  组合指标:")
print(f"    预期年化收益: {result_max_sharpe['return']:.4f}")
print(f"    预期年化波动: {result_max_sharpe['volatility']:.4f}")
print(f"    夏普比率:     {result_max_sharpe['sharpe']:.4f}")

# ============================================================
# 2. 最小方差组合
# ============================================================
print("\n" + "=" * 70)
print("2. 最小方差组合")
print("=" * 70)

result_min_var = optimize(
    returns,
    objective="min_variance",
    short_allowed=False,
    max_weight=0.5,
)

print(f"\n优化结果:")
print(f"  最优权重:")
for asset, weight in zip(assets, result_min_var['weights']):
    if weight > 0.001:
        print(f"    {asset:<10} {weight:>10.2%}")

print(f"\n  组合指标:")
print(f"    预期年化收益: {result_min_var['return']:.4f}")
print(f"    预期年化波动: {result_min_var['volatility']:.4f}")
print(f"    夏普比率:     {result_min_var['sharpe']:.4f}")

# ============================================================
# 3. 目标收益优化
# ============================================================
print("\n" + "=" * 70)
print("3. 目标收益优化")
print("=" * 70)

target_return = 0.15  # 目标年化收益15%
result_target_ret = optimize(
    returns,
    objective="target_return",
    target_return=target_return,
    short_allowed=False,
    max_weight=0.5,
)

print(f"\n目标年化收益: {target_return:.2%}")
print(f"\n优化结果:")
print(f"  最优权重:")
for asset, weight in zip(assets, result_target_ret['weights']):
    if weight > 0.001:
        print(f"    {asset:<10} {weight:>10.2%}")

print(f"\n  组合指标:")
print(f"    预期年化收益: {result_target_ret['return']:.4f}")
print(f"    预期年化波动: {result_target_ret['volatility']:.4f}")
print(f"    夏普比率:     {result_target_ret['sharpe']:.4f}")

# ============================================================
# 4. 目标风险优化
# ============================================================
print("\n" + "=" * 70)
print("4. 目标风险优化")
print("=" * 70)

target_vol = 0.18  # 目标年化波动18%
result_target_vol = optimize(
    returns,
    objective="target_risk",
    target_volatility=target_vol,
    short_allowed=False,
    max_weight=0.5,
)

print(f"\n目标年化波动: {target_vol:.2%}")
print(f"\n优化结果:")
print(f"  最优权重:")
for asset, weight in zip(assets, result_target_vol['weights']):
    if weight > 0.001:
        print(f"    {asset:<10} {weight:>10.2%}")

print(f"\n  组合指标:")
print(f"    预期年化收益: {result_target_vol['return']:.4f}")
print(f"    预期年化波动: {result_target_vol['volatility']:.4f}")
print(f"    夏普比率:     {result_target_vol['sharpe']:.4f}")

# ============================================================
# 5. 行业约束优化
# ============================================================
print("\n" + "=" * 70)
print("5. 行业约束优化")
print("=" * 70)

# 行业约束：科技股30-70%，金融股30-70%
sector_constraints = {
    'Technology': (0.30, 0.70),
    'Financials': (0.30, 0.70),
}

result_sector = optimize(
    returns,
    objective="max_sharpe",
    risk_free_rate=0.02,
    short_allowed=False,
    max_weight=0.5,
    sector_constraints=sector_constraints,
    sector_map=sector_map,
)

print(f"\n行业约束:")
print(f"  Technology: 30% ~ 70%")
print(f"  Financials: 30% ~ 70%")

print(f"\n优化结果:")
print(f"  最优权重:")
for asset, weight in zip(assets, result_sector['weights']):
    if weight > 0.001:
        print(f"    {asset:<10} {weight:>10.2%}")

# 计算行业配置
tech_alloc = sum(result_sector['weights'][i] for i, a in enumerate(assets) if sector_map.get(a) == 'Technology')
fin_alloc = sum(result_sector['weights'][i] for i, a in enumerate(assets) if sector_map.get(a) == 'Financials')

print(f"\n  行业配置:")
print(f"    Technology: {tech_alloc:.2%}")
print(f"    Financials: {fin_alloc:.2%}")

print(f"\n  组合指标:")
print(f"    预期年化收益: {result_sector['return']:.4f}")
print(f"    预期年化波动: {result_sector['volatility']:.4f}")
print(f"    夏普比率:     {result_sector['sharpe']:.4f}")

# ============================================================
# 6. 允许卖空的优化
# ============================================================
print("\n" + "=" * 70)
print("6. 允许卖空的优化")
print("=" * 70)

result_short = optimize(
    returns,
    objective="max_sharpe",
    risk_free_rate=0.02,
    short_allowed=True,  # 允许卖空
    max_weight=0.5,  # 绝对权重限制
    min_weight=-0.3,  # 最大卖空30%
)

print(f"\n允许卖空，最大多头50%，最大空头30%")
print(f"\n优化结果:")
print(f"  最优权重:")
for asset, weight in zip(assets, result_short['weights']):
    if abs(weight) > 0.001:
        pos_type = "多头" if weight > 0 else "空头"
        print(f"    {asset:<10} {weight:>9.2%}  ({pos_type})")

print(f"\n  组合指标:")
print(f"    预期年化收益: {result_short['return']:.4f}")
print(f"    预期年化波动: {result_short['volatility']:.4f}")
print(f"    夏普比率:     {result_short['sharpe']:.4f}")

# ============================================================
# 7. 优化策略对比
# ============================================================
print("\n" + "=" * 70)
print("7. 优化策略对比")
print("=" * 70)

comparison = [
    ("最大夏普", result_max_sharpe['return'], result_max_sharpe['volatility'], result_max_sharpe['sharpe']),
    ("最小方差", result_min_var['return'], result_min_var['volatility'], result_min_var['sharpe']),
    ("目标收益", result_target_ret['return'], result_target_ret['volatility'], result_target_ret['sharpe']),
    ("目标风险", result_target_vol['return'], result_target_vol['volatility'], result_target_vol['sharpe']),
    ("行业约束", result_sector['return'], result_sector['volatility'], result_sector['sharpe']),
    ("允许卖空", result_short['return'], result_short['volatility'], result_short['sharpe']),
]

comparison_df = pd.DataFrame(
    comparison,
    columns=['策略', '预期收益', '预期波动', '夏普比率']
)

print(f"\n{comparison_df.to_string(index=False)}")

# ============================================================
# 8. 有效前沿
# ============================================================
print("\n" + "=" * 70)
print("8. 有效前沿计算")
print("=" * 70)

# 计算多个目标收益下的最优组合
target_returns = np.linspace(0.05, 0.30, 10)
frontier_points = []

for target in target_returns:
    try:
        res = optimize(
            returns,
            objective="target_return",
            target_return=target,
            short_allowed=False,
            max_weight=0.5,
        )
        frontier_points.append({
            'return': res['expected_return'],
            'volatility': res['volatility'],
            'sharpe': res['sharpe'],
        })
    except:
        pass

if frontier_points:
    frontier_df = pd.DataFrame(frontier_points)
    print(f"\n有效前沿 ({len(frontier_df)} 个点):")
    print("-" * 50)
    print(f"{'预期收益':>12} {'预期波动':>12} {'夏普比率':>10}")
    print("-" * 50)
    for _, row in frontier_df.iterrows():
        print(f"{row['return']:>10.2%} {row['volatility']:>10.2%} {row['sharpe']:>10.4f}")

    # 找到最优夏普比率点
    max_sharpe_idx = frontier_df['sharpe'].idxmax()
    optimal_point = frontier_df.iloc[max_sharpe_idx]
    print(f"\n最优夏普组合:")
    print(f"  预期收益: {optimal_point['return']:.4f}")
    print(f"  预期波动: {optimal_point['volatility']:.4f}")
    print(f"  夏普比率: {optimal_point['sharpe']:.4f}")

# ============================================================
# 9. 可视化
# ============================================================
try:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 1. 权重对比
    ax = axes[0]
    x = np.arange(len(assets))
    width = 0.15

    strategies = [
        ('最大夏普', result_max_sharpe['weights']),
        ('最小方差', result_min_var['weights']),
        ('行业约束', result_sector['weights']),
    ]

    for i, (name, weights) in enumerate(strategies):
        ax.bar(x + i * width, weights, width, label=name, alpha=0.8)

    ax.set_xlabel('资产')
    ax.set_ylabel('权重')
    ax.set_title('不同优化策略的权重对比')
    ax.set_xticks(x + width)
    ax.set_xticklabels(assets)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # 2. 有效前沿
    ax = axes[1]
    if frontier_points:
        ax.plot(frontier_df['volatility'], frontier_df['return'],
                'o-', label='有效前沿', markersize=4)
        # 标记各策略
        strategies_to_plot = [
            (result_max_sharpe, '最大夏普'),
            (result_min_var, '最小方差'),
            (result_sector, '行业约束'),
        ]
        for res, name in strategies_to_plot:
            ax.plot(res['volatility'], res['expected_return'],
                    'o', label=name, markersize=8)

    ax.set_xlabel('预期波动')
    ax.set_ylabel('预期收益')
    ax.set_title('有效前沿与优化策略')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('custom_optimization.png', dpi=100)
    print("\n优化可视化已保存: custom_optimization.png")

except ImportError:
    print("\n未安装 matplotlib，跳过可视化")

print("\n" + "=" * 70)
print("自定义组合优化分析完成！")
print("=" * 70)
print("""
提示：
1. max_sharpe: 最大化风险调整后收益
2. min_variance: 最小化组合波动率
3. target_return: 在给定目标收益下最小化风险
4. target_risk: 在给定风险下最大化收益
5. sector_constraints: 可以设置行业配置约束
6. short_allowed: 允许卖空可以获得更高的风险调整收益
""")
