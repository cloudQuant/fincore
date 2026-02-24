"""
风险模型与极值理论示例

展示如何使用 fincore 计算高级风险指标：
- Value at Risk (VaR) 和 Conditional VaR
- 极值理论 (EVT) 风险估计
- GARCH 波动率模型
- 压力测试情景分析

适用场景：
- 风险管理和合规报告
- 监管资本计算
- 压力测试
- 期权定价和风险管理
"""

import numpy as np
import pandas as pd
from fincore import Empyrical

print("=" * 70)
print("风险模型与极值理论示例")
print("=" * 70)

# 生成模拟收益数据
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=1260, freq="B", tz="UTC")

# 模拟具有厚尾分布的收益（更接近真实市场）
returns = pd.Series(np.random.standard_t(4, 1260) * 0.01, index=dates, name="returns")

print(f"\n数据概览:")
print(f"  时间范围: {dates[0].date()} 至 {dates[-1].date()}")
print(f"  交易日数: {len(returns)}")
print(f"  年化波动率: {Empyrical.annual_volatility(returns):.2%}")
print(f"")

# ============================================================
# 1. 风险价值 (VaR) 和 条件风险价值 (CVaR)
# ============================================================
print("-" * 70)
print("1. 风险价值 (Value at Risk) 和 条件风险价值 (Conditional VaR)")
print("-" * 70)

# 不同置信水平的 VaR
confidence_levels = [0.01, 0.05, 0.10, 0.20]
print("\n不同置信水平的 VaR:")
print("-" * 40)
for cutoff in confidence_levels:
    var = Empyrical.value_at_risk(returns, cutoff=cutoff)
    print(f"  {int(cutoff*100):3d}% VaR:     {var:>8.4f}")

# 超额收益的 VaR (相对于无风险利率)
var_excess = Empyrical.var_excess_return(returns, cutoff=0.05, risk_free=0.03)
print(f"\n5% 超额收益 VaR (相对 3% 无风险利率): {var_excess:.4f}")

# 条件风险价值 (CVaR) / 期望损失
cvar = Empyrical.conditional_value_at_risk(returns)
print(f"\n条件风险价值 (CVaR): {cvar:.4f}")

# ============================================================
# 2. 下行风险
# ============================================================
print("\n" + "-" * 70)
print("2. 下行风险 (Downside Risk)")
print("-" * 70)

downside = Empyrical.downside_risk(returns, required_return=0.0)
print(f"下行风险 (相对 0%): {downside:.4f}")

downside_neg = Empyrical.downside_risk(returns, required_return=0.02)
print(f"下行风险 (相对 2%): {downside_neg:.4f}")

# 下行波动率
downside_volatility = Empyrical.downside_risk(returns)
print(f"下行波动率: {downside_volatility:.4f}")

# Sortino 比率（基于下行风险）
sortino = Empyrical.sortino_ratio(returns)
print(f"Sortino 比率: {sortino:.4f}")

# ============================================================
# 3. 极值理论 (EVT) 风险估计
# ============================================================
print("\n" + "-" * 70)
print("3. 极值理论 (Extreme Value Theory) 风险估计")
print("-" * 70)

try:
    # 需要 scipy 才能使用 EVT
    gpd_risk = Empyrical.gpd_risk_estimates(returns, p=0.05)
    print("\nGPD 风险估计:")
    print(f"  VaR (GPD):     {gpd_risk.get('var', np.nan):.4f}")
    print(f"  CVaR (GPD):    {gpd_risk.get('cvar', np.nan):.4f}")
    print(f"  形状参数 xi:    {gpd_risk.get('xi', np.nan):.4f}")
    print(f"  尺度参数 beta: {gpd_risk.get('beta', np.nan):.4f}")

    # 相对于无风险利率的超额收益 EVT 风险
    rf = 0.03
    var_excess_evt = Empyrical.gpd_risk_estimates_aligned(
        returns, benchmark_returns=None, p=0.05
    )
    if isinstance(var_excess_evt, dict):
        print(f"\n超额收益 GPD 风险估计:")
        print(f"  VaR (GPD):     {var_excess_evt.get('var', np.nan):.4f}")
        print(f"  CVaR (GPD):    {var_excess_evt.get('cvar', np.nan):.4f}")
    else:
        print(f"\n超额收益 GPD VaR (5%): {var_excess_evt:.4f}")

except Exception as e:
    print(f"\nEVT 计算（需要 scipy）: {e}")
    print("注意: 安装 scipy 后可使用极值理论功能")

# ============================================================
# 4. GARCH 波动率模型
# ============================================================
print("\n" + "-" * 70)
print("4. GARCH 波动率模型")
print("-" * 70)

try:
    from fincore.risk import garch

    # 拟合 GARCH(1,1) 模型
    garch_result = garch.fit_garch(returns, p=1, q=1)
    print("\nGARCH(1,1) 模型参数:")
    print(f"  条件均值 (mu):     {garch_result.get('mu', np.nan):.6f}")
    print(f"  条件方差 (omega):  {garch_result.get('omega', np.nan):.6f}")
    print(f"  ARCH 系数 (alpha):  {garch_result.get('alpha', np.nan):.6f}")
    print(f"  GARCH 系数 (beta):  {garch_result.get('beta', np.nan):.6f}")
    print(f"  持续方差 (persistence): {garch_result.get('persistence', np.nan):.6f}")

    # 预测波动率
    forecast = garch_result.get('forecast', None)
    if forecast is not None:
        print(f"\n未来 5 天波动率预测:")
        for i, val in enumerate(forecast[:5]):
            print(f"  第 {i+1} 天: {val**2:.4f} (日化)")

except Exception as e:
    print(f"\nGARCH 计算（需要 arch）: {e}")
    print("注意: 安装 arch 后可使用 GARCH 功能")

# ============================================================
# 5. 压力测试情景分析
# ============================================================
print("\n" + "-" * 70)
print("5. 压力测试情景分析")
print("-" * 70)

# 定义压力测试情景
scenarios = {
    "2008 金融危机": -0.05,
    "2020 疫情暴跌": -0.03,
    "极端黑天鹅": -0.10,
}

print("\n策略在压力情景下的表现:")
print("-" * 50)

for scenario_name, shock in scenarios.items():
    # 计算策略在压力情景下的单日价值变化
    # VaR 已经是负值，表示损失
    var_5 = Empyrical.value_at_risk(returns, cutoff=0.05)
    var_1 = Empyrical.value_at_risk(returns, cutoff=0.01)

    print(f"\n  {scenario_name}:")
    print(f"    单日冲击:      {shock:.1%}")
    print(f"    5% VaR:        {var_5:.4f} ({var_5*100:.2f}%)")
    print(f"    1% VaR:        {var_1:.4f} ({var_1*100:.2f}%)")

# 计算压力情景后的组合价值
initial_value = 1000000  # 100万初始投资
for scenario_name, shock in scenarios.items():
    var_5 = Empyrical.value_at_risk(returns, cutoff=0.05)
    # 使用最坏情况的 VaR
    worst_case = min(var_5, shock)
    remaining_value = initial_value * (1 + worst_case)
    print(f"    {scenario_name}后剩余: ¥{remaining_value:,.0f}")

# ============================================================
# 6. 风险汇总报告
# ============================================================
print("\n" + "=" * 70)
print("风险指标汇总报告")
print("=" * 70)

risk_metrics = {
    "95% VaR (日度)": Empyrical.value_at_risk(returns, cutoff=0.05),
    "99% VaR (日度)": Empyrical.value_at_risk(returns, cutoff=0.01),
    "CVaR (日度)": Empyrical.conditional_value_at_risk(returns),
    "下行风险": Empyrical.downside_risk(returns),
    "年化波动率": Empyrical.annual_volatility(returns),
    "最大回撤": Empyrical.max_drawdown(returns),
    "尾部比率": Empyrical.tail_ratio(returns),
}

print("\n风险指标:")
print("-" * 40)
for name, value in risk_metrics.items():
    print(f"  {name:20s}: {value:>10.4f}")

# 风险等级评估
print("\n风险等级评估:")
print("-" * 40)
annual_vol = Empyrical.annual_volatility(returns)
max_dd = Empyrical.max_drawdown(returns)

if annual_vol < 0.10:
    vol_risk = "低"
elif annual_vol < 0.20:
    vol_risk = "中"
else:
    vol_risk = "高"

if max_dd > -0.10:
    dd_risk = "低"
elif max_dd > -0.20:
    dd_risk = "中"
else:
    dd_risk = "高"

print(f"  波动率风险:    {vol_risk}")
print(f"  回撤风险:      {dd_risk}")

# ============================================================
# 7. 可视化（如果安装了 matplotlib）
# ============================================================
try:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 收益分布直方图
    ax = axes[0, 0]
    returns.hist(bins=50, ax=ax, edgecolor='black', alpha=0.7)
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax.set_title('收益分布')
    ax.set_xlabel('日收益率')
    ax.set_ylabel('频数')
    ax.grid(True, alpha=0.3)

    # Q-Q 图（检查正态性）
    ax = axes[0, 1]
    from scipy import stats
    stats.probplot(returns, dist="norm", plot=ax)
    ax.set_title('Q-Q 图（正态性检验）')
    ax.grid(True, alpha=0.3)

    # 滚动 VaR
    ax = axes[1, 0]
    from fincore.metrics.rolling import roll_max_drawdown
    rolling_var = [Empyrical.value_at_risk(returns.iloc[:i+1], cutoff=0.05) for i in range(250, len(returns), 50)]
    var_dates = returns.index[250::50][:len(rolling_var)]  # 匹配日期长度
    ax.plot(var_dates, rolling_var, label='滚动 5% VaR')
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax.set_title('滚动 VaR (5%)')
    ax.set_ylabel('VaR')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 滚动波动率
    ax = axes[1, 1]
    from fincore.metrics.rolling import roll_annual_volatility
    rolling_vol = roll_annual_volatility(returns, window=60)
    rolling_vol.plot(ax=ax, title='滚动年化波动率 (60日)')
    ax.set_ylabel('波动率')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('risk_models.png', dpi=100)
    print("\n风险模型可视化图表已保存: risk_models.png")

except ImportError:
    print("\n未安装 scipy/matplotlib，跳过可视化")

print("\n" + "=" * 70)
print("风险模型分析完成！")
print("=" * 70)
