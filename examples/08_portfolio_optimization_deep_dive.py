"""
组合优化深度示例

展示如何使用 fincore 的组合优化功能：
- 有效前沿计算
- 风险平价组合
- 最大夏普比率组合
- 目标约束优化

适用场景：
- 投资组合构建
- 资产配置决策
- 风险预算管理
"""

import numpy as np
import pandas as pd
from fincore.optimization import efficient_frontier, risk_parity
from fincore import sharpe_ratio, max_drawdown, annual_volatility

print("=" * 70)
print("组合优化深度示例")
print("=" * 70)

# 生成多资产收益数据
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=252*3, freq="B", tz="UTC")
n_assets = 5

# 模拟5个资产的收益，具有一定的相关性
assets = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']

# 生成相关收益
mean_returns = np.array([0.0008, 0.0006, 0.0005, 0.0007, 0.0012])
# 简化：假设资产独立（实际应用中应使用真实相关性）
returns = pd.DataFrame(
    np.random.multivariate_normal(
        mean_returns,
        np.diag([0.015, 0.012, 0.018, 0.020, 0.025]),  # 对角协方差矩阵
        len(dates)
    ),
    index=dates,
    columns=assets
)

print(f"\n数据概览:")
print(f"  资产数量: {n_assets}")
print(f"  时间范围: {dates[0].date()} 至 {dates[-1].date()}")
print(f"  观测值数: {len(returns)}")

# 计算各资产的基本统计
print("\n各资产统计:")
print("-" * 50)
print(f"{'资产':<10} {'年化收益':>12} {'年化波动率':>12} {'Sharpe':>10}")
print("-" * 50)

for asset in assets:
    asset_returns = returns[asset]
    ann_ret = asset_returns.mean() * 252
    ann_vol = asset_returns.std() * np.sqrt(252)
    sharpe = sharpe_ratio(asset_returns)
    print(f"{asset:<10} {ann_ret:>10.2%} {ann_vol:>10.2%} {sharpe:>10.4f}")

# ============================================================
# 1. 有效前沿 (Efficient Frontier)
# ============================================================
print("\n" + "=" * 70)
print("1. 有效前沿 (Efficient Frontier)")
print("=" * 70)

try:
    # 计算有效前沿
    frontier = efficient_frontier(returns, n_points=20)

    print(f"\n有效前沿计算完成，包含 {len(frontier)} 个投资组合")
    print("\n前沿上的 5 个代表性组合:")
    print("-" * 70)

    # 选择5个代表性的点
    indices = np.linspace(0, len(frontier)-1, 5, dtype=int)
    for i, idx in enumerate(indices):
        portfolio = frontier[idx]
        print(f"\n  组合 {i+1}:")
        print(f"    预期收益:  {portfolio['expected_return']:.4f}")
        print(f"    预期波动:  {portfolio['volatility']:.4f}")
        print(f"    夏普比率:  {portfolio.get('sharpe', 'N/A')}")
        print(f"    权重分配:")
        for asset, weight in zip(assets, portfolio['weights']):
            if weight > 0.01:  # 只显示权重大于1%的资产
                print(f"      {asset}: {weight:.2%}")

except Exception as e:
    print(f"\n有效前沿计算失败: {e}")
    print("注意: 需要 scipy 来运行组合优化")

# ============================================================
# 2. 风险平价 (Risk Parity)
# ============================================================
print("\n" + "=" * 70)
print("2. 风险平价组合 (Risk Parity)")
print("=" * 70)

try:
    rp_weights = risk_parity(returns)

    print("\n风险平价权重:")
    print("-" * 30)
    for asset, weight in zip(assets, rp_weights):
        print(f"  {asset:<10} {weight:>10.2%}")

    # 计算风险平价组合的收益
    portfolio_returns = returns @ rp_weights
    rp_sharpe = sharpe_ratio(portfolio_returns)
    rp_vol = annual_volatility(portfolio_returns)
    rp_ret = portfolio_returns.mean() * 252
    rp_dd = max_drawdown(portfolio_returns)

    print(f"\n风险平价组合绩效:")
    print("-" * 30)
    print(f"  年化收益:     {rp_ret:.4f}")
    print(f"  年化波动率:   {rp_vol:.4f}")
    print(f"  夏普比率:     {rp_sharpe:.4f}")
    print(f"  最大回撤:     {rp_dd:.4f}")

except Exception as e:
    print(f"\n风险平价计算失败: {e}")

# ============================================================
# 3. 等权重组合对比
# ============================================================
print("\n" + "-" * 70)
print("3. 等权重组合对比")
print("-" * 70)

weight_strategies = {
    "等权重": np.ones(n_assets) / n_assets,
    "最小方差": None,  # 需要优化
    "60/40 股债": np.array([0.6, 0.4, 0, 0, 0]),
}

print("\n等权重组合:")
print("-" * 40)
ew_weights = weight_strategies["等权重"]
ew_returns = returns @ ew_weights
ew_sharpe = sharpe_ratio(ew_returns)
ew_vol = annual_volatility(ew_returns)
ew_ret = ew_returns.mean() * 252
ew_dd = max_drawdown(ew_returns)

print(f"  年化收益:   {ew_ret:.4f}")
print(f"  年化波动率: {ew_vol:.4f}")
print(f"  夏普比率:   {ew_sharpe:.4f}")
print(f"  最大回撤:   {ew_dd:.4f}")

# 60/40 股债组合
print("\n60/40 股债组合 (前两个资产):")
print("-" * 40)
stock_bond_weights = weight_strategies["60/40 股债"]
sb_returns = returns @ stock_bond_weights
sb_sharpe = sharpe_ratio(sb_returns)
sb_vol = annual_volatility(sb_returns)
sb_ret = sb_returns.mean() * 252
sb_dd = max_drawdown(sb_returns)

print(f"  年化收益:   {sb_ret:.4f}")
print(f"  年化波动率: {sb_vol:.4f}")
print(f"  夏普比率:   {sb_sharpe:.4f}")
print(f"  最大回撤:   {sb_dd:.4f}")

# ============================================================
# 4. 组合对比汇总
# ============================================================
print("\n" + "=" * 70)
print("组合策略对比汇总")
print("=" * 70)

comparison = []
if 'rp_sharpe' in locals():
    comparison.append(("风险平价", rp_ret, rp_vol, rp_sharpe, rp_dd))
comparison.append(("等权重", ew_ret, ew_vol, ew_sharpe, ew_dd))
comparison.append(("60/40 股债", sb_ret, sb_vol, sb_sharpe, sb_dd))

if comparison:
    comparison_df = pd.DataFrame(
        comparison,
        columns=['策略', '年化收益', '年化波动率', '夏普比率', '最大回撤']
    )
    print("\n" + comparison_df.to_string(index=False))

# ============================================================
# 5. 可视化
# ============================================================
try:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 资产收益散点图
    ax = axes[0]
    for asset in assets:
        ax.scatter(annual_volatility(returns[asset]),
                  returns[asset].mean() * 252,
                  label=asset, s=100, alpha=0.7)
    ax.set_xlabel('年化波动率')
    ax.set_ylabel('年化收益率')
    ax.set_title('风险-收益散点图')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 资产收益热力图
    ax = axes[1]
    corr = returns.corr()
    im = ax.imshow(corr, cmap='RdYlGn', aspect='auto', vmin=-1, vmax=1)
    ax.set_xticks(range(len(assets)))
    ax.set_yticks(range(len(assets)))
    ax.set_xticklabels(assets, rotation=45)
    ax.set_yticklabels(assets)
    ax.set_title('资产收益相关性')
    plt.colorbar(im, ax=ax, label='相关系数')
    ax.grid(False)

    plt.tight_layout()
    plt.savefig('portfolio_optimization.png', dpi=100)
    print("\n组合优化可视化已保存: portfolio_optimization.png")

except ImportError:
    print("\n未安装 matplotlib，跳过可视化")

print("\n" + "=" * 70)
print("组合优化分析完成！")
print("=" * 70)
print("""
提示：
1. 有效前沿展示了所有最优风险-收益组合
2. 风险平价提供基于风险贡献的权重分配
3. 实际应用中应考虑交易成本、流动性、约束条件等
4. 定期再平衡（如每月或每季度）以维持目标配置
""")
