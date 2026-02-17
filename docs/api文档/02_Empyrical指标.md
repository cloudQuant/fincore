# Empyrical 指标 API

`Empyrical` 是 fincore 的核心分析类，提供 100+ 个性能/风险指标。所有指标通过 `_LazyMethod` 延迟加载。

## 构造函数

```python
from fincore import Empyrical

emp = Empyrical(
    returns=returns,              # pd.Series, 必需
    factor_returns=benchmark,     # pd.Series, 可选
)
```

## 基础指标

| 方法 | 模块 | 说明 |
|------|------|------|
| `sharpe_ratio(returns, risk_free=0, period="daily")` | ratios | 夏普比率 |
| `sortino_ratio(returns, required_return=0, period="daily")` | ratios | 索提诺比率 |
| `calmar_ratio(returns, period="daily")` | ratios | 卡尔玛比率 |
| `omega_ratio(returns, risk_free=0)` | ratios | 欧米伽比率 |
| `annual_return(returns, period="daily")` | yearly | 年化收益率 |
| `annual_volatility(returns, period="daily")` | risk | 年化波动率 |
| `max_drawdown(returns)` | drawdown | 最大回撤 |
| `stability_of_timeseries(returns)` | ratios | 稳定性 (R²) |

## 风险指标

| 方法 | 模块 | 说明 |
|------|------|------|
| `value_at_risk(returns, cutoff=0.05)` | risk | 在险价值 |
| `conditional_value_at_risk(returns, cutoff=0.05)` | risk | 条件 VaR (CVaR/ES) |
| `downside_risk(returns, required_return=0)` | risk | 下行风险 |
| `tail_ratio(returns)` | risk | 尾部比率 |
| `tracking_error(returns, factor_returns)` | risk | 跟踪误差 |

## Alpha / Beta

| 方法 | 模块 | 说明 |
|------|------|------|
| `alpha(returns, factor_returns, risk_free=0)` | alpha_beta | Alpha |
| `beta(returns, factor_returns, risk_free=0)` | alpha_beta | Beta |
| `alpha_beta(returns, factor_returns)` | alpha_beta | (Alpha, Beta) |
| `alpha_beta_aligned(returns, factor_returns)` | alpha_beta | 对齐后计算 |
| `up_alpha_beta(returns, factor_returns)` | alpha_beta | 上行 Alpha/Beta |
| `down_alpha_beta(returns, factor_returns)` | alpha_beta | 下行 Alpha/Beta |
| `up_capture(returns, factor_returns)` | alpha_beta | 上行捕获率 |
| `down_capture(returns, factor_returns)` | alpha_beta | 下行捕获率 |
| `capture(returns, factor_returns)` | ratios | 综合捕获率 |

## 回撤分析

| 方法 | 模块 | 说明 |
|------|------|------|
| `max_drawdown(returns)` | drawdown | 最大回撤 |
| `get_top_drawdowns(returns, top=10)` | drawdown | Top N 回撤详情 |
| `gen_drawdown_table(returns, top=10)` | drawdown | 回撤汇总表 |
| `get_max_drawdown_underwater(returns)` | drawdown | 水下回撤序列 |

## 滚动指标

| 方法 | 模块 | 说明 |
|------|------|------|
| `rolling_sharpe(returns, rolling_window=63)` | rolling | 滚动夏普 |
| `rolling_volatility(returns, rolling_window=63)` | rolling | 滚动波动率 |
| `rolling_beta(returns, factor_returns, rolling_window=63)` | rolling | 滚动 Beta |
| `rolling_alpha(returns, factor_returns, rolling_window=63)` | rolling | 滚动 Alpha |

## 统计指标

| 方法 | 模块 | 说明 |
|------|------|------|
| `skewness(returns)` | stats | 偏度 |
| `kurtosis(returns)` | stats | 峰度 |
| `hurst_exponent(returns)` | stats | Hurst 指数 |
| `serial_correlation(returns)` | stats | 序列相关性 |

## 按年聚合

| 方法 | 模块 | 说明 |
|------|------|------|
| `annual_return_by_year(returns)` | yearly | 逐年年化收益 |
| `sharpe_ratio_by_year(returns)` | yearly | 逐年夏普比率 |
| `max_drawdown_by_year(returns)` | yearly | 逐年最大回撤 |

## 连续涨跌

| 方法 | 模块 | 说明 |
|------|------|------|
| `max_consecutive_up_days(returns)` | consecutive | 最大连续上涨天数 |
| `max_consecutive_down_days(returns)` | consecutive | 最大连续下跌天数 |
| `max_single_day_gain(returns)` | consecutive | 单日最大收益 |
| `max_single_day_loss(returns)` | consecutive | 单日最大亏损 |

## 实例方法 (@_dual_method)

以下方法在实例上调用时自动填充 `returns` / `factor_returns`：

| 方法 | 自动填充参数 | 说明 |
|------|-------------|------|
| `win_rate()` | returns | 胜率 |
| `loss_rate()` | returns | 亏损率 |
| `max_drawdown_days()` | returns | 最大回撤天数 |
| `max_drawdown_recovery_days()` | returns | 最大回撤恢复天数 |
| `second_max_drawdown()` | returns | 第二大回撤 |
| `third_max_drawdown()` | returns | 第三大回撤 |
| `sterling_ratio()` | returns | 斯特林比率 |
| `burke_ratio()` | returns | 伯克比率 |
| `kappa_three_ratio()` | returns | Kappa Three 比率 |
| `common_sense_ratio()` | returns | 常识比率 |
| `treynor_ratio()` | returns + factor_returns | 特雷诺比率 |
| `m_squared()` | returns + factor_returns | M² |
| `tracking_difference()` | returns + factor_returns | 跟踪差异 |
| `residual_risk()` | returns + factor_returns | 残差风险 |

## 综合统计

```python
# 一次性获取所有核心指标
stats = Empyrical.perf_stats(returns, factor_returns=benchmark)
# 返回 pd.Series，包含年化收益、夏普、回撤等
```
