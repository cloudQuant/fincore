# 顶层 API

## 包级别函数 (Flat API)

通过 `import fincore` 直接访问的常用指标函数。延迟加载，首次调用时才导入底层模块。

### 收益指标

| 函数 | 签名 | 说明 |
|------|------|------|
| `annual_return` | `(returns, period="daily", annualization=None)` | 年化收益率（CAGR） |
| `cum_returns` | `(returns, starting_value=0)` | 累计收益序列 |
| `cum_returns_final` | `(returns, starting_value=0)` | 最终累计收益（标量） |
| `simple_returns` | `(prices)` | 从价格序列计算简单收益率 |
| `aggregate_returns` | `(returns, convert_to)` | 按周/月/年聚合收益 |

### 风险指标

| 函数 | 签名 | 说明 |
|------|------|------|
| `annual_volatility` | `(returns, period="daily", alpha=2.0, annualization=None)` | 年化波动率 |
| `max_drawdown` | `(returns)` | 最大回撤（负值） |
| `downside_risk` | `(returns, required_return=0, period="daily", annualization=None)` | 下行风险 |
| `value_at_risk` | `(returns, cutoff=0.05)` | 在险价值 (VaR) |
| `tail_ratio` | `(returns)` | 尾部比率 (95th / 5th percentile) |

### 比率指标

| 函数 | 签名 | 说明 |
|------|------|------|
| `sharpe_ratio` | `(returns, risk_free=0, period="daily", annualization=None)` | 夏普比率 |
| `sortino_ratio` | `(returns, required_return=0, period="daily", annualization=None)` | 索提诺比率 |
| `calmar_ratio` | `(returns, period="daily", annualization=None)` | 卡尔玛比率 |
| `omega_ratio` | `(returns, risk_free=0, required_return=0, annualization=252)` | 欧米伽比率 |
| `information_ratio` | `(returns, factor_returns)` | 信息比率 |
| `capture` | `(returns, factor_returns, period="daily")` | 捕获率 |
| `stability_of_timeseries` | `(returns)` | 时间序列稳定性 (R²) |

### Alpha / Beta

| 函数 | 签名 | 说明 |
|------|------|------|
| `alpha` | `(returns, factor_returns, risk_free=0, period="daily", annualization=None)` | Alpha |
| `beta` | `(returns, factor_returns, risk_free=0)` | Beta |
| `alpha_beta` | `(returns, factor_returns, risk_free=0, period="daily", annualization=None)` | (Alpha, Beta) 元组 |

## 核心类

### `fincore.Empyrical`

性能分析核心类，提供 100+ 指标。详见 [Empyrical 指标](02_Empyrical指标.md)。

```python
from fincore import Empyrical

# 类级别调用
sr = Empyrical.sharpe_ratio(returns)

# 实例级别调用
emp = Empyrical(returns=returns)
wr = emp.win_rate()
```

### `fincore.Pyfolio`

继承自 `Empyrical`，增加可视化功能。详见 [Pyfolio 可视化](03_Pyfolio可视化.md)。

```python
from fincore import Pyfolio
pf = Pyfolio(returns=returns)
pf.create_returns_tear_sheet()
```

### `fincore.analyze()`

创建 `AnalysisContext`，所有指标通过 `cached_property` 惰性计算并缓存。

```python
ctx = fincore.analyze(returns, factor_returns=benchmark)
print(ctx.sharpe_ratio)   # 首次计算并缓存
print(ctx.max_drawdown)   # 首次计算并缓存
stats = ctx.perf_stats()  # 汇总所有指标
```

**AnalysisContext 属性**:

| 属性 | 类型 | 说明 |
|------|------|------|
| `annual_return` | float | 年化收益率 |
| `cumulative_returns` | float | 累计收益 |
| `annual_volatility` | float | 年化波动率 |
| `sharpe_ratio` | float | 夏普比率 |
| `calmar_ratio` | float | 卡尔玛比率 |
| `stability` | float | 稳定性 (R²) |
| `max_drawdown` | float | 最大回撤 |
| `omega_ratio` | float | 欧米伽比率 |
| `sortino_ratio` | float | 索提诺比率 |
| `skew` | float | 偏度 |
| `kurtosis` | float | 峰度 |
| `tail_ratio` | float | 尾部比率 |
| `daily_value_at_risk` | float | 日 VaR |
| `alpha` | float | Alpha（需要 factor_returns） |
| `beta` | float | Beta（需要 factor_returns） |
| `information_ratio` | float | 信息比率（需要 factor_returns） |

**AnalysisContext 方法**:

| 方法 | 返回 | 说明 |
|------|------|------|
| `perf_stats()` | `pd.Series` | 所有指标汇总 |
| `to_dict()` | `dict` | 转为字典 |
| `to_json()` | `str` | 转为 JSON |
| `to_html(path=None)` | `str` | 生成 HTML 报告 |
| `plot(backend="matplotlib")` | viz backend | 绘制核心图表 |
| `invalidate()` | None | 清除所有缓存 |

### `fincore.create_strategy_report()`

生成完整策略分析报告。

```python
fincore.create_strategy_report(
    returns,
    positions=None,        # 可选: 持仓数据
    transactions=None,     # 可选: 交易数据
    trades=None,           # 可选: 已平仓交易
    title="Strategy Report",
    output="report.html",  # .html 或 .pdf
)
```
