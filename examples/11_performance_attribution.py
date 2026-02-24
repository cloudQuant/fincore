"""
绩效归因分析示例

展示如何使用 fincore 进行绩效归因分析：
- Brinson 归因模型
- Fama-French 三因子归因
- 行业/因子暴露分析
- 交互效应分解

适用场景：
- 投资组合绩效分解
- 识别超额收益来源
- 评估主动管理效果
- 因子暴露分析
"""

import numpy as np
import pandas as pd
from fincore import Empyrical
from fincore.attribution import brinson, fama_french

print("=" * 70)
print("绩效归因分析示例")
print("=" * 70)

# 生成模拟数据
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=252*2, freq="B", tz="UTC")
n_assets = 5
n_periods = len(dates)

# 资产名称
assets = ['Technology', 'Healthcare', 'Finance', 'Consumer', 'Energy']
sectors = assets.copy()

# 市场基准权重 (等权重)
benchmark_weights = np.array([0.20, 0.20, 0.20, 0.20, 0.20])

# 组合权重 (偏重科技和医疗)
portfolio_weights = np.array([0.35, 0.30, 0.15, 0.10, 0.10])

# 资产收益
asset_returns = pd.DataFrame(
    np.random.multivariate_normal(
        [0.0008, 0.0005, 0.0004, 0.0006, 0.0003],
        [[0.0002, 0.0001, 0.00005, 0.00008, 0.00003],
         [0.0001, 0.00015, 0.00004, 0.00006, 0.00002],
         [0.00005, 0.00004, 0.00018, 0.00007, 0.00004],
         [0.00008, 0.00006, 0.00007, 0.00012, 0.00003],
         [0.00003, 0.00002, 0.00004, 0.00003, 0.00025]],
        n_periods
    ),
    index=dates,
    columns=assets
)

print(f"\n数据概览:")
print(f"  时间范围: {dates[0].date()} 至 {dates[-1].date()}")
print(f"  资产数量: {n_assets}")
print(f"  行业: {', '.join(sectors)}")

# ============================================================
# 1. 基本收益计算
# ============================================================
print("\n" + "=" * 70)
print("1. 基本收益计算")
print("=" * 70)

# 组合收益
portfolio_returns = asset_returns @ portfolio_weights
benchmark_returns = asset_returns @ benchmark_weights

# 计算累计收益
portfolio_cum = (1 + portfolio_returns).cumprod().iloc[-1] - 1
benchmark_cum = (1 + benchmark_returns).cumprod().iloc[-1] - 1
active_cum = portfolio_cum - benchmark_cum

print(f"\n累计收益 (2年期):")
print(f"  组合:   {portfolio_cum:>8.2%}")
print(f"  基准:   {benchmark_cum:>8.2%}")
print(f"  超额:   {active_cum:>8.2%}")

# 年化收益
portfolio_ann = portfolio_returns.mean() * 252
benchmark_ann = benchmark_returns.mean() * 252
active_ann = portfolio_ann - benchmark_ann

print(f"\n年化收益:")
print(f"  组合:   {portfolio_ann:>8.2%}")
print(f"  基准:   {benchmark_ann:>8.2%}")
print(f"  超额:   {active_ann:>8.2%}")

# ============================================================
# 2. Brinson 归因分析
# ============================================================
print("\n" + "=" * 70)
print("2. Brinson 归因分析")
print("=" * 70)

# 构建分段收益数据 (简化版: 按年度分解)
years = [2020, 2021]

print(f"\n年度 Brinson 归因:")
print("-" * 70)
print(f"{'年份':<8} {'配置效应':>12} {'选股效应':>12} {'交互效应':>12} {'超额收益':>12}")
print("-" * 70)

for year in years:
    year_mask = asset_returns.index.year == year
    year_returns = asset_returns[year_mask]

    # 计算该年度的资产收益
    asset_ret = year_returns.mean() * 252  # 年化

    # Brinson 归因
    allocation = ((portfolio_weights - benchmark_weights) *
                  (asset_ret.values - asset_ret.values @ benchmark_weights))

    selection = (benchmark_weights *
                 (asset_ret.values - asset_ret.values @ benchmark_weights))

    interaction = ((portfolio_weights - benchmark_weights) *
                   (asset_ret.values - asset_ret.values @ benchmark_weights))

    total_active = allocation.sum() + selection.sum() + interaction.sum()

    print(f"{year:<8} {allocation.sum():>11.2%} {selection.sum():>11.2%} "
          f"{interaction.sum():>11.2%} {total_active:>11.2%}")

# 总体归因
overall_asset_ret = asset_returns.mean() * 252
allocation = ((portfolio_weights - benchmark_weights) *
              (overall_asset_ret.values - overall_asset_ret.values @ benchmark_weights))
selection = (benchmark_weights *
             (overall_asset_ret.values - overall_asset_ret.values @ benchmark_weights))
interaction = ((portfolio_weights - benchmark_weights) *
               (overall_asset_ret.values - overall_asset_ret.values @ benchmark_weights))

print("-" * 70)
print(f"{'总体':<8} {allocation.sum():>11.2%} {selection.sum():>11.2%} "
      f"{interaction.sum():>11.2%} {active_ann:>11.2%}")

# 归因贡献排名
print(f"\n各行业对配置效应的贡献:")
print("-" * 40)
for i, asset in enumerate(assets):
    print(f"  {asset:<12}: {allocation[i]:>7.2%}")

# ============================================================
# 3. 因子暴露分析
# ============================================================
print("\n" + "=" * 70)
print("3. 因子暴露分析")
print("=" * 70)

# 模拟因子收益
factor_returns = pd.DataFrame({
    'Market': np.random.normal(0.0004, 0.01, n_periods),
    'SMB': np.random.normal(0.0001, 0.008, n_periods),   # Small Minus Big
    'HML': np.random.normal(0.0002, 0.006, n_periods),   # High Minus Low
    'Momentum': np.random.normal(0.0003, 0.007, n_periods),
}, index=dates)

# 组合对因子的敏感性 (模拟)
factor_betas = {
    'Market': 1.05,
    'SMB': 0.30,
    'HML': -0.20,
    'Momentum': 0.15,
}

print(f"\n组合因子暴露:")
print("-" * 30)
for factor, beta in factor_betas.items():
    print(f"  {factor:<12}: {beta:>7.2f}")

# 计算因子贡献
print(f"\n因子收益贡献 (年化):")
print("-" * 40)

total_factor_return = 0
for factor, beta in factor_betas.items():
    factor_ann_return = factor_returns[factor].mean() * 252
    contribution = beta * factor_ann_return
    total_factor_return += contribution
    print(f"  {factor:<12}: {contribution:>7.2%} (β={beta:.2f})")

print(f"  {'总因子收益':<12}: {total_factor_return:>7.2%}")

# Alpha (特质收益)
alpha_ann = active_ann - total_factor_return
print(f"  {'Alpha (特质)':<12}: {alpha_ann:>7.2%}")

# ============================================================
# 4. 风险归因
# ============================================================
print("\n" + "=" * 70)
print("4. 风险归因")
print("=" * 70)

# 计算组合和基准的波动率
portfolio_vol = portfolio_returns.std() * np.sqrt(252)
benchmark_vol = benchmark_returns.std() * np.sqrt(252)

print(f"\n年化波动率:")
print(f"  组合:   {portfolio_vol:>8.2%}")
print(f"  基准:   {benchmark_vol:>8.2%}")

# 跟踪误差
tracking_error = (portfolio_returns - benchmark_returns).std() * np.sqrt(252)
print(f"  跟踪误差: {tracking_error:>8.2%}")

# 信息比率
information_ratio = active_ann / tracking_error
print(f"\n信息比率: {information_ratio:.4f}")

# 各资产对组合风险的贡献
print(f"\n各资产对组合风险的贡献 (边际风险贡献):")
print("-" * 50)

# 计算协方差矩阵
cov_matrix = asset_returns.cov() * 252  # 年化

# 组合方差
portfolio_var = float(portfolio_weights @ cov_matrix.values @ portfolio_weights)
portfolio_std = np.sqrt(portfolio_var)

for i, asset in enumerate(assets):
    # 边际风险贡献
    marginal_risk = (cov_matrix.values[i, :] @ portfolio_weights) / portfolio_std
    # 风险贡献
    risk_contrib = portfolio_weights[i] * marginal_risk
    risk_percent = risk_contrib / portfolio_var * 100
    print(f"  {asset:<12}: {risk_contrib:>7.2%} ({risk_percent:>5.1f}%)")

# ============================================================
# 5. 滚动归因分析
# ============================================================
print("\n" + "=" * 70)
print("5. 滚动超额收益分析")
print("=" * 70)

window = 126  # 半年滚动窗口
rolling_active = []
rolling_alpha = []

for i in range(window, len(portfolio_returns)):
    port_ret = portfolio_returns.iloc[i-window:i]
    bench_ret = benchmark_returns.iloc[i-window:i]

    # 滚动超额收益
    rolling_active.append((port_ret.mean() - bench_ret.mean()) * 252)

    # 滚动 Alpha
    try:
        alpha, _ = Empyrical.alpha_beta(port_ret, bench_ret)
        rolling_alpha.append(alpha * 252)
    except:
        rolling_alpha.append(np.nan)

rolling_active = pd.Series(rolling_active, index=portfolio_returns.index[window:])
rolling_alpha = pd.Series(rolling_alpha, index=portfolio_returns.index[window:])

print(f"\n滚动超额收益统计 (窗口={window}天):")
print(f"  均值:     {rolling_active.mean():.4f}")
print(f"  标准差:   {rolling_active.std():.4f}")
print(f"  最大值:   {rolling_active.max():.4f}")
print(f"  最小值:   {rolling_active.min():.4f}")

print(f"\n滚动 Alpha 统计:")
print(f"  均值:     {rolling_alpha.mean():.4f}")
print(f"  标准差:   {rolling_alpha.std():.4f}")

# ============================================================
# 6. 胜负分析
# ============================================================
print("\n" + "=" * 70)
print("6. 胜负分析")
print("=" * 70)

# 跑赢基准的天数
win_days = (portfolio_returns > benchmark_returns).sum()
total_days = len(portfolio_returns)
win_rate = win_days / total_days

print(f"\n跑赢基准统计:")
print(f"  跑赢天数:   {win_days} / {total_days}")
print(f"  胜率:       {win_rate:.2%}")

# 胜负时的平均超额收益
win_excess = (portfolio_returns - benchmark_returns)[
    portfolio_returns > benchmark_returns
].mean() * 252
lose_excess = (portfolio_returns - benchmark_returns)[
    portfolio_returns <= benchmark_returns
].mean() * 252

print(f"\n超额收益分布:")
print(f"  跑赢时平均: {win_excess:.2%}")
print(f"  跑输时平均: {lose_excess:.2%}")
print(f"  盈亏比:     {abs(win_excess / lose_excess):.2f}")

# ============================================================
# 7. 可视化
# ============================================================
try:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. 累计收益对比
    ax = axes[0, 0]
    ((1 + portfolio_returns).cumprod() * 100).plot(ax=ax, label='组合')
    ((1 + benchmark_returns).cumprod() * 100).plot(ax=ax, label='基准')
    ax.set_title('累计收益对比')
    ax.set_ylabel('收益 (%)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. 超额收益
    ax = axes[0, 1]
    excess_returns = (portfolio_returns - benchmark_returns)
    ((1 + excess_returns).cumprod() * 100).plot(ax=ax, color='green')
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax.set_title('累计超额收益')
    ax.set_ylabel('超额收益 (%)')
    ax.grid(True, alpha=0.3)

    # 3. 滚动超额收益
    ax = axes[1, 0]
    rolling_active.plot(ax=ax)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax.set_title(f'滚动超额收益 (窗口={window}天)')
    ax.set_ylabel('年化超额收益')
    ax.grid(True, alpha=0.3)

    # 4. Brinson 归因条形图
    ax = axes[1, 1]
    attribution_data = {
        '配置效应': allocation.sum(),
        '选股效应': selection.sum(),
        '交互效应': interaction.sum(),
    }
    colors = ['steelblue', 'forestgreen', 'orange']
    ax.bar(attribution_data.keys(), attribution_data.values(),
           color=colors, alpha=0.8)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_title('Brinson 归因分解')
    ax.set_ylabel('收益贡献')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('performance_attribution.png', dpi=100)
    print("\n绩效归因分析可视化已保存: performance_attribution.png")

except ImportError:
    print("\n未安装 matplotlib，跳过可视化")

print("\n" + "=" * 70)
print("绩效归因分析完成！")
print("=" * 70)
print("""
提示：
1. Brinson 归因将超额收益分解为配置、选股和交互效应
2. 配置效应反映仓位选择决策的贡献
3. 选股效应反映个股/行业选择决策的贡献
4. 交互效应是配置和选股的协同作用
5. 信息比率衡量单位跟踪误差下的超额收益
""")
