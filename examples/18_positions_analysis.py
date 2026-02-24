"""
持仓分析示例

展示如何使用 fincore 分析投资组合持仓：
- 持仓分配分析
- 多空分析
- 杠杆分析
- 行业/板块暴露
- 集中度分析

适用场景：
- 投资组合监控
- 风险暴露分析
- 合规检查
- 持仓报告生成
"""

import numpy as np
import pandas as pd
from fincore.metrics.positions import (
    get_percent_alloc,
    get_top_long_short_abs,
    get_max_median_position_concentration,
    gross_lev,
)
from fincore import max_drawdown, sharpe_ratio

print("=" * 70)
print("持仓分析示例")
print("=" * 70)

# ============================================================
# 1. 准备持仓数据
# ============================================================
print("\n数据准备...")

np.random.seed(42)
dates = pd.date_range("2023-01-01", periods=252, freq="B", tz="UTC")
n_assets = 10

assets = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META',  # 科技
    'JPM', 'BAC', 'WFC', 'GS',                 # 金融
]

# 模拟持仓数据 (单位: 元)
positions = pd.DataFrame(
    np.random.uniform(-500000, 1000000, (len(dates), n_assets)),
    index=dates,
    columns=assets
)

# 确保有正有负 (多空策略)
positions.iloc[:, :5] = np.abs(positions.iloc[:, :5]) * 1.5  # 科技多头
positions.iloc[:, 5:] = -np.abs(positions.iloc[:, 5:]) * 0.8  # 金融空头

print(f"持仓数据:")
print(f"  时间范围: {dates[0].date()} 至 {dates[-1].date()}")
print(f"  资产数量: {n_assets}")
print(f"  数据形状: {positions.shape}")

# 最新持仓
latest_pos = positions.iloc[-1]
print(f"\n最新持仓 (前5个):")
print(latest_pos.head())

# ============================================================
# 2. 持仓分配分析
# ============================================================
print("\n" + "=" * 70)
print("2. 持仓分配分析")
print("=" * 70)

# 计算持仓分配百分比
allocations = get_percent_alloc(positions.iloc[-1:])
latest_alloc = allocations.iloc[-1]

print(f"\n最新持仓分配:")
print("-" * 40)
print(f"{'资产':<10} {'持仓金额':>15} {'分配比例':>12}")
print("-" * 40)

total_value = latest_pos.sum()
for asset in assets:
    amount = latest_pos[asset]
    alloc = latest_alloc[asset]
    print(f"{asset:<10} {amount:>13,.0f}  {alloc:>10.2%}")

print("-" * 40)
print(f"{'总计':<10} {total_value:>13,.0f}  {100:>10.2%}")

# ============================================================
# 3. 多空分析
# ============================================================
print("\n" + "=" * 70)
print("3. 多空分析")
print("=" * 70)

# 计算多头和空头持仓
long_pos = positions.copy()
long_pos[long_pos < 0] = 0

short_pos = positions.copy()
short_pos[short_pos > 0] = 0

long_exposure = long_pos.sum(axis=1).iloc[-1]
short_exposure = abs(short_pos.sum(axis=1).iloc[-1])
gross_exposure = long_exposure + short_exposure
net_exposure = long_exposure - short_exposure

print(f"\n最新多空暴露:")
print(f"  多头暴露:  {long_exposure:>13,.0f}")
print(f"  空头暴露:  {short_exposure:>13,.0f}")
print(f"  总暴露:    {gross_exposure:>13,.0f}")
print(f"  净暴露:    {net_exposure:>13,.0f}")

# 杠杆倍数
equity = 1000000  # 假设权益
leverage = gross_exposure / equity
print(f"\n杠杆倍数 (假设权益={equity:,.0f}):")
print(f"  总杠杆: {leverage:.2f}x")

# ============================================================
# 4. 头寸集中度分析
# ============================================================
print("\n" + "=" * 70)
print("4. 头寸集中度分析")
print("=" * 70)

# 计算最大和中位数头寸集中度
max_conc, median_conc = get_max_median_position_concentration(positions)

print(f"\n头寸集中度统计:")
print(f"  最大头寸占比:     {max_conc:.2%}")
print(f"  中位数头寸占比:   {median_conc:.2%}")

# 头位资产集中度
top_concentration = latest_alloc.abs().sort_values(ascending=False)
print(f"\n前5大持仓占比:")
for i, (asset, alloc) in enumerate(top_concentration.head(5).items(), 1):
    print(f"  {i}. {asset:<10} {alloc:>10.2%}")

# HH指数 (赫芬达尔-赫希曼指数)
hh_index = (latest_alloc ** 2).sum()
print(f"\nHH指数 (赫芬达尔): {hh_index:.4f}")
if hh_index < 0.15:
    print(f"  集中度: 低")
elif hh_index < 0.25:
    print(f"  集中度: 中")
else:
    print(f"  集中度: 高")

# ============================================================
# 5. 顶级多头空头分析
# ============================================================
print("\n" + "=" * 70)
print("5. 顶级多头空头分析")
print("=" * 70)

top_positions = get_top_long_short_abs(positions, top=5)

print(f"\n前5大多头:")
print("-" * 40)
for asset, amount in top_positions['long'].head(5).items():
    alloc = amount / total_value
    print(f"  {asset:<10} {amount:>13,.0f}  ({alloc:>7.2%})")

print(f"\n前5大空头:")
print("-" * 40)
for asset, amount in top_positions['short'].head(5).items():
    alloc = abs(amount) / total_value
    print(f"  {asset:<10} {amount:>13,.0f}  ({alloc:>7.2%})")

print(f"\n前5大绝对持仓:")
print("-" * 40)
for asset, amount in top_positions['abs'].head(5).items():
    alloc = abs(amount) / total_value
    pos_type = "多头" if amount > 0 else "空头"
    print(f"  {asset:<10} {amount:>13,.0f}  ({alloc:>7.2%}) {pos_type}")

# ============================================================
# 6. 时间序列持仓分析
# ============================================================
print("\n" + "=" * 70)
print("6. 时间序列持仓分析")
print("=" * 70)

# 计算每日总暴露
daily_total = positions.sum(axis=1)
daily_gross = positions.abs().sum(axis=1)

print(f"\n暴露统计:")
print(f"  平均总暴露:  {daily_total.mean():>13,.0f}")
print(f"  平均总敞口:  {daily_gross.mean():>13,.0f}")

# 持仓换手率 (简化版)
pos_changes = positions.diff().abs().sum(axis=1)
avg_turnover = pos_changes.mean()
print(f"\n日均换手金额: {avg_turnover:>13,.0f}")

# 持仓稳定性
pos_stability = 1 - (pos_changes.std() / pos_changes.mean())
print(f"持仓稳定性指数: {pos_stability:.4f}")

# ============================================================
# 7. 行业暴露分析
# ============================================================
print("\n" + "=" * 70)
print("7. 行业暴露分析")
print("=" * 70)

# 行业映射
sector_map = {
    'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology',
    'AMZN': 'Consumer', 'META': 'Technology',
    'JPM': 'Financials', 'BAC': 'Financials', 'WFC': 'Financials',
    'GS': 'Financials',
}

# 计算行业暴露
sector_exposure = {}
for sector in set(sector_map.values()):
    sector_pos = 0
    for asset, exposure_amount in latest_pos.items():
        if sector_map.get(asset) == sector:
            sector_pos += exposure_amount
    sector_exposure[sector] = sector_pos

print(f"\n行业暴露:")
print("-" * 40)
for sector, exposure in sorted(sector_exposure.items(), key=lambda x: abs(x[1]), reverse=True):
    alloc = exposure / total_value
    pos_type = "多头" if exposure > 0 else "空头"
    print(f"  {sector:<12} {exposure:>10,.0f}  ({alloc:>7.2%}) {pos_type}")

# ============================================================
# 8. 杠杆历史分析
# ============================================================
print("\n" + "=" * 70)
print("8. 杠杆历史分析")
print("=" * 70)

# 计算每日杠杆
daily_lev = []
for i in range(len(positions)):
    daily_pos = positions.iloc[i]
    gross = daily_pos.abs().sum()
    lev = gross_lev(daily_pos)
    daily_lev.append(lev)

daily_lev = pd.Series(daily_lev, index=positions.index)

print(f"\n杠杆统计:")
print(f"  平均杠杆: {daily_lev.mean():.2f}x")
print(f"  最大杠杆: {daily_lev.max():.2f}x")
print(f"  最小杠杆: {daily_lev.min():.2f}x")
print(f"  杠杆标准差: {daily_lev.std():.2f}x")

# 杠杆分布
print(f"\n杠杆分布:")
print(f"  < 1x: {(daily_lev < 1).sum():>4} 天")
print(f"  1-2x: {((daily_lev >= 1) & (daily_lev < 2)).sum():>4} 天")
print(f"  2-3x: {((daily_lev >= 2) & (daily_lev < 3)).sum():>4} 天")
print(f"  > 3x: {(daily_lev >= 3).sum():>4} 天")

# ============================================================
# 9. 持仓风险评估
# ============================================================
print("\n" + "=" * 70)
print("9. 持仓风险评估")
print("=" * 70)

# 计算持仓收益 (简化)
returns = pd.Series(
    np.random.normal(0.0008, 0.015, len(dates)),
    index=dates
)

# 基于持仓的风险指标
print(f"\n持仓风险指标:")
print(f"  最大回撤: {max_drawdown(returns):.4f}")
print(f"  夏普比率: {sharpe_ratio(returns):.4f}")

# 集中度风险
if max_conc > 0.3:
    print(f"  集中度风险: 高 (单一持仓占比{max_conc:.1%})")
elif max_conc > 0.2:
    print(f"  集中度风险: 中 (单一持仓占比{max_conc:.1%})")
else:
    print(f"  集中度风险: 低 (单一持仓占比{max_conc:.1%})")

# 杠杆风险
if daily_lev.mean() > 2:
    print(f"  杠杆风险: 高 (平均{daily_lev.mean():.1f}x)")
elif daily_lev.mean() > 1.5:
    print(f"  杠杆风险: 中 (平均{daily_lev.mean():.1f}x)")
else:
    print(f"  杠杆风险: 低 (平均{daily_lev.mean():.1f}x)")

# ============================================================
# 10. 可视化
# ============================================================
try:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. 持仓分配饼图
    ax = axes[0, 0]
    alloc_abs = latest_alloc.abs()
    alloc_abs = alloc_abs[alloc_abs > 0.01]  # 只显示>1%的
    ax.pie(alloc_abs, labels=alloc_abs.index, autopct='%1.1f%%', startangle=90)
    ax.set_title('持仓分配 (绝对值)')

    # 2. 多空暴露
    ax = axes[0, 1]
    long_short_daily = pd.DataFrame({
        '多头': long_pos.sum(axis=1),
        '空头': short_pos.abs().sum(axis=1)
    })
    long_short_daily.plot(ax=ax, kind='area', stacked=True, alpha=0.7)
    ax.set_title('多空暴露时间序列')
    ax.set_ylabel('暴露金额')
    ax.grid(True, alpha=0.3)

    # 3. 杠杆时间序列
    ax = axes[1, 0]
    daily_lev.plot(ax=ax)
    ax.axhline(y=daily_lev.mean(), color='r', linestyle='--',
               label=f'平均: {daily_lev.mean():.2f}x')
    ax.set_title('杠杆倍数时间序列')
    ax.set_ylabel('杠杆倍数')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. 头寸集中度
    ax = axes[1, 1]
    sorted_alloc = latest_alloc.abs().sort_values(ascending=True)
    ax.barh(range(len(sorted_alloc)), sorted_alloc.values * 100)
    ax.set_yticks(range(len(sorted_alloc)))
    ax.set_yticklabels(sorted_alloc.index)
    ax.set_xlabel('持仓占比 (%)')
    ax.set_title('头寸集中度')
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig('positions_analysis.png', dpi=100)
    print("\n持仓分析可视化已保存: positions_analysis.png")

except ImportError:
    print("\n未安装 matplotlib，跳过可视化")

print("\n" + "=" * 70)
print("持仓分析完成！")
print("=" * 70)
print("""
分析指标说明:
1. 持仓分配: 各资产占组合的百分比
2. 多空分析: 多头和空头暴露情况
3. 头寸集中度: 最大和中位数头寸占比
4. 杠杆倍数: 总暴露与权益的比率
5. 换手率: 持仓变动频率

建议:
- 定期监控持仓集中度
- 关注杠杆水平变化
- 平衡多空暴露
- 分散行业配置
""")
