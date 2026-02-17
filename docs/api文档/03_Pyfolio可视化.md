# Pyfolio 可视化 API

`Pyfolio` 继承自 `Empyrical`，增加 tear sheet 和绑定绘图功能。

## 构造函数

```python
from fincore import Pyfolio

pf = Pyfolio(
    returns=returns,
    factor_returns=benchmark,  # 可选
)
```

## Tear Sheet（一键报告）

| 方法 | 说明 |
|------|------|
| `create_returns_tear_sheet(returns)` | 收益分析报告（累计收益、滚动指标、月度热力图等） |
| `create_full_tear_sheet(returns, positions=None, transactions=None)` | 完整报告（收益 + 持仓 + 交易分析） |
| `create_interesting_times_tear_sheet(returns)` | 重大事件期间表现分析 |

## 收益相关绘图

| 方法 | 参数 | 说明 |
|------|------|------|
| `plot_rolling_returns(returns, factor_returns=None, ax=None)` | 累计收益曲线 |
| `plot_rolling_sharpe(returns, rolling_window=63, ax=None)` | 滚动夏普比率 |
| `plot_rolling_volatility(returns, rolling_window=63, ax=None)` | 滚动波动率 |
| `plot_rolling_beta(returns, factor_returns, rolling_window=63, ax=None)` | 滚动 Beta |
| `plot_return_quantiles(returns, ax=None)` | 收益分位数箱线图 |

## 月度/年度分析

| 方法 | 参数 | 说明 |
|------|------|------|
| `plot_monthly_returns_heatmap(returns, ax=None)` | 月度收益热力图 |
| `plot_annual_returns(returns, ax=None)` | 年度收益柱状图 |
| `plot_monthly_returns_dist(returns, ax=None)` | 月度收益分布图 |

## 回撤分析

| 方法 | 参数 | 说明 |
|------|------|------|
| `plot_drawdown_periods(returns, top=10, ax=None)` | Top N 回撤区间标注 |
| `plot_drawdown_underwater(returns, ax=None)` | 水下曲线图 |

## 持仓分析

| 方法 | 参数 | 说明 |
|------|------|------|
| `plot_exposures(returns, positions, ax=None)` | 多空暴露图 |
| `plot_gross_leverage(returns, positions, ax=None)` | 总杠杆图 |

## 交易分析

| 方法 | 参数 | 说明 |
|------|------|------|
| `plot_turnover(returns, transactions, positions, ax=None)` | 日换手率 |
| `plot_daily_volume(returns, transactions, ax=None)` | 日交易量 |

## 使用示例

```python
import matplotlib.pyplot as plt
from fincore import Pyfolio

pf = Pyfolio(returns=returns)

# 单个图表
fig, axes = plt.subplots(3, 1, figsize=(14, 16))
pf.plot_rolling_returns(returns, ax=axes[0])
pf.plot_rolling_sharpe(returns, ax=axes[1])
pf.plot_drawdown_underwater(returns, ax=axes[2])
plt.tight_layout()
plt.savefig("analysis.png")

# 完整 tear sheet
pf.create_full_tear_sheet(
    returns,
    positions=positions,
    transactions=transactions,
)
```
