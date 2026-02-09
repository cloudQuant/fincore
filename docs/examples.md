# fincore 使用示例

本文档介绍 fincore 的三种使用方式，覆盖从快速调用到面向对象分析的完整场景。

---

## 准备工作

```python
import numpy as np
import pandas as pd
import fincore

# 生成模拟收益数据
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=500, freq="B")
returns = pd.Series(np.random.normal(0.0005, 0.02, 500), index=dates, name="strategy")
factor_returns = pd.Series(np.random.normal(0.0003, 0.015, 500), index=dates, name="benchmark")
```

---

## 方式一：Flat API（函数式调用）

最简洁的调用方式，适合快速计算单个指标。常用函数直接从 `fincore` 包级别导入。

```python
import fincore

# 直接使用 fincore.xxx()
sr = fincore.sharpe_ratio(returns)
md = fincore.max_drawdown(returns)
ar = fincore.annual_return(returns)
av = fincore.annual_volatility(returns)
cr = fincore.cum_returns_final(returns)
a, b = fincore.alpha_beta(returns, factor_returns)

print(f"夏普比率:   {sr:.4f}")
print(f"最大回撤:   {md:.4f}")
print(f"年化收益:   {ar:.4f}")
print(f"年化波动:   {av:.4f}")
print(f"累计收益:   {cr:.4f}")
print(f"Alpha:      {a:.4f}")
print(f"Beta:       {b:.4f}")
```

支持的 Flat API 函数列表：

| 函数 | 说明 |
|------|------|
| `sharpe_ratio` | 夏普比率 |
| `sortino_ratio` | 索提诺比率 |
| `max_drawdown` | 最大回撤 |
| `annual_return` | 年化收益率 |
| `annual_volatility` | 年化波动率 |
| `cum_returns` | 累计收益序列 |
| `cum_returns_final` | 最终累计收益 |
| `alpha` / `beta` / `alpha_beta` | Alpha / Beta |
| `calmar_ratio` | 卡尔玛比率 |
| `omega_ratio` | 欧米伽比率 |
| `information_ratio` | 信息比率 |
| `stability_of_timeseries` | 时间序列稳定性 |
| `tail_ratio` | 尾部比率 |
| `value_at_risk` | 在险价值 |
| `capture` | 捕获率 |
| `downside_risk` | 下行风险 |
| `simple_returns` | 简单收益 |
| `aggregate_returns` | 聚合收益 |

---

## 方式二：Empyrical 类级别调用

通过 `Empyrical` 类直接调用，无需实例化。可以访问 100+ 个指标函数，比 Flat API 覆盖面更广。

```python
from fincore import Empyrical

# 基础指标
sr = Empyrical.sharpe_ratio(returns)
md = Empyrical.max_drawdown(returns)
ar = Empyrical.annual_return(returns)

# 风险指标
var = Empyrical.value_at_risk(returns)
cvar = Empyrical.conditional_value_at_risk(returns)
dr = Empyrical.downside_risk(returns)
tr = Empyrical.tail_ratio(returns)
te = Empyrical.tracking_error(returns, factor_returns)

# 回撤分析
top_dd = Empyrical.get_top_drawdowns(returns, top=5)
dd_table = Empyrical.gen_drawdown_table(returns, top=5)
dd_underwater = Empyrical.get_max_drawdown_underwater(returns)

# Alpha / Beta 系列
a, b = Empyrical.alpha_beta(returns, factor_returns)
a_aligned, b_aligned = Empyrical.alpha_beta_aligned(returns, factor_returns)
up_a, up_b = Empyrical.up_alpha_beta(returns, factor_returns)
down_a, down_b = Empyrical.down_alpha_beta(returns, factor_returns)

# 滚动指标
rolling_sharpe = Empyrical.rolling_sharpe(returns, rolling_window=63)
rolling_vol = Empyrical.rolling_volatility(returns, rolling_window=63)
rolling_beta = Empyrical.rolling_beta(returns, factor_returns, rolling_window=63)

# 统计指标
skew = Empyrical.skewness(returns)
kurt = Empyrical.kurtosis(returns)
hurst = Empyrical.hurst_exponent(returns)

# 按年聚合
annual_by_year = Empyrical.annual_return_by_year(returns)
sharpe_by_year = Empyrical.sharpe_ratio_by_year(returns)
dd_by_year = Empyrical.max_drawdown_by_year(returns)

# 连续涨跌
up_days = Empyrical.max_consecutive_up_days(returns)
down_days = Empyrical.max_consecutive_down_days(returns)
max_gain = Empyrical.max_single_day_gain(returns)
max_loss = Empyrical.max_single_day_loss(returns)

# 收益处理
cum = Empyrical.cum_returns(returns)
agg_monthly = Empyrical.aggregate_returns(returns, "monthly")
normalized = Empyrical.normalize(returns, starting_value=100)

# 综合统计表
stats_df = Empyrical.perf_stats(returns, factor_returns=factor_returns)
print(stats_df)
```

### 延迟加载机制

`Empyrical` 类使用 `_LazyMethod` 描述符实现延迟加载。**首次**访问某个方法时，才会导入对应的底层模块并缓存：

```python
# 首次访问: 触发 _LazyMethod.__get__() → 导入 fincore.metrics.ratios → 缓存
sr = Empyrical.sharpe_ratio(returns)

# 后续访问: 直接命中缓存的 staticmethod，零开销
sr2 = Empyrical.sharpe_ratio(returns)
```

这意味着 `import fincore` 非常快（~0.05s），只有实际使用到的模块才会被加载。

---

## 方式三：Empyrical 实例调用（面向对象）

创建实例后，`returns` 和 `factor_returns` 会自动填充到方法调用中，无需每次传参。

```python
from fincore import Empyrical

# 创建实例，绑定数据
emp = Empyrical(
    returns=returns,
    factor_returns=factor_returns,
)

# 自动使用实例的 returns，无需传参
print(f"最大回撤天数:       {emp.max_drawdown_days()}")
print(f"最大回撤恢复天数:   {emp.max_drawdown_recovery_days()}")
print(f"第二大回撤:         {emp.second_max_drawdown():.4f}")
print(f"胜率:               {emp.win_rate():.4f}")
print(f"亏损率:             {emp.loss_rate():.4f}")
print(f"序列相关:           {emp.serial_correlation():.4f}")
print(f"常识比率:           {emp.common_sense_ratio():.4f}")

# 自动使用实例的 returns + factor_returns
print(f"特雷诺比率:         {emp.treynor_ratio():.4f}")
print(f"M²:                 {emp.m_squared():.4f}")
print(f"跟踪差异:           {emp.tracking_difference():.4f}")
print(f"残差风险:           {emp.residual_risk():.4f}")

# 连续涨跌日期
print(f"最大连续上涨开始:   {emp.max_consecutive_up_start_date()}")
print(f"最大连续上涨结束:   {emp.max_consecutive_up_end_date()}")
print(f"单日最大收益日期:   {emp.max_single_day_gain_date()}")

# 滚动指标（同时使用 returns + factor_returns）
roll_a = emp.roll_alpha(window=63)
roll_b = emp.roll_beta(window=63)
roll_ab = emp.roll_alpha_beta(window=63)

# 也可以显式传参覆盖实例数据
other_returns = pd.Series(np.random.normal(0.001, 0.03, 500), index=dates)
print(f"其他策略胜率: {emp.win_rate(returns=other_returns):.4f}")
```

### 实例方法 vs 类方法

| 调用方式 | 说明 | 示例 |
|----------|------|------|
| `Empyrical.sharpe_ratio(returns)` | 类级别，必须传参 | 100+ 个指标函数 |
| `emp.sharpe_ratio(returns)` | 实例级别，必须传参 | 同上（通过 `_LazyMethod` 描述符） |
| `emp.win_rate()` | 实例级别，自动填充 `returns` | `@_dual_method` 方法 |
| `emp.treynor_ratio()` | 实例级别，自动填充 `returns` + `factor_returns` | `@_dual_method` 方法 |

---

## 方式四：AnalysisContext（高级分析）

`analyze()` 创建一个带缓存的分析上下文，适合需要计算大量指标的场景。

```python
from fincore import analyze

ctx = analyze(
    returns=returns,
    factor_returns=factor_returns,
)

# cached_property: 首次计算后缓存，重复访问零开销
print(f"年化收益: {ctx.annual_return:.4f}")
print(f"夏普比率: {ctx.sharpe_ratio:.4f}")
print(f"最大回撤: {ctx.max_drawdown:.4f}")

# 综合统计
stats = ctx.perf_stats()
print(stats)
```

---

## 方式五：Pyfolio（可视化报告）

`Pyfolio` 继承自 `Empyrical`，增加了 tear sheet 可视化功能。

```python
from fincore import Pyfolio

pf = Pyfolio(
    returns=returns,
    factor_returns=factor_returns,
)

# 生成各种报告（需要 matplotlib）
pf.create_returns_tear_sheet()
pf.create_full_tear_sheet()
```

---

## 实战案例：分析 AbberationStrategy 策略

完整脚本见 `examples/011_abberation/analyze_strategy.py`，从策略回测日志中读取数据，使用 fincore 进行全面分析。

### 数据来源

策略运行后在 `logs/` 目录下生成以下日志文件（TSV 格式）：

| 文件 | 内容 | 关键列 |
|------|------|--------|
| `value.log` | 每根K线的组合净值 | `dt, value, cash` |
| `trade.log` | 每笔交易开平仓记录 | `pnl, pnlcomm, commission, dtopen, dtclose, long` |
| `order.log` | 订单记录 | `ordtype, status, size, price, dt` |
| `position.log` | 每根K线的持仓 | `dt, data_name, size, price` |

### 核心流程

```python
import pandas as pd
import fincore
from fincore import Empyrical

# 1. 读取 value.log，取每日最后一根K线的净值
value_df = pd.read_csv("logs/.../value.log", sep="\t", usecols=["dt", "value"])
value_df["dt"] = pd.to_datetime(value_df["dt"])
daily_value = value_df.groupby(value_df["dt"].dt.date)["value"].last()
daily_value.index = pd.to_datetime(daily_value.index)

# 2. 计算日收益率
daily_returns = daily_value.pct_change().dropna()

# 3. Flat API 快速看核心指标
print(f"夏普比率: {fincore.sharpe_ratio(daily_returns):.4f}")
print(f"最大回撤: {fincore.max_drawdown(daily_returns):.4f}")
print(f"年化收益: {fincore.annual_return(daily_returns):.4f}")

# 4. Empyrical 类调用扩展指标
dd_table = Empyrical.gen_drawdown_table(daily_returns, top=5)
stats = Empyrical.perf_stats(daily_returns)
yearly = pd.DataFrame({
    "年化收益": Empyrical.annual_return_by_year(daily_returns),
    "夏普比率": Empyrical.sharpe_ratio_by_year(daily_returns),
    "最大回撤": Empyrical.max_drawdown_by_year(daily_returns),
})

# 5. 实例方法获取回撤细节
emp = Empyrical(returns=daily_returns)
print(f"最大回撤天数: {emp.max_drawdown_days()}")
print(f"胜率:         {emp.win_rate():.4f}")
print(f"斯特林比率:   {emp.sterling_ratio():.4f}")

# 6. 读取 trade.log 做交易统计
trade_df = pd.read_csv("logs/.../trade.log", sep="\t")
closed = trade_df[trade_df["status"] == "Closed"]
win_rate = len(closed[closed["pnlcomm"] > 0]) / len(closed)
avg_win = closed[closed["pnlcomm"] > 0]["pnlcomm"].mean()
avg_loss = closed[closed["pnlcomm"] <= 0]["pnlcomm"].mean()
print(f"交易胜率: {win_rate:.2%}, 盈亏比: {abs(avg_win/avg_loss):.2f}")
```

### 运行结果摘要（AbberationStrategy on RB 螺纹钢 2018-07 → 2021-08）

```
策略: AbberationStrategy
品种: RB (螺纹钢)
区间: 2018-07-02 → 2021-08-03
年化收益: -0.53%
夏普比率: 0.0602
最大回撤: -27.07%
交易次数: 244
胜率:     37.70%
盈亏比:   1.59

按年统计:
         年化收益  夏普比率  最大回撤
2018     -0.1131   -0.4429   -0.1728
2019     -0.0119   -0.0090   -0.1437
2020     -0.0378   -0.1357   -0.1346
2021      0.1772    0.8931   -0.1205
```

运行方式：
```bash
cd examples/011_abberation
python analyze_strategy.py
```

---

## 架构总览

```
fincore
├── fincore.sharpe_ratio()              # Flat API（方式一）
├── Empyrical.sharpe_ratio(returns)     # 类级别调用（方式二）
├── emp.win_rate()                      # 实例调用，自动填充（方式三）
├── analyze(returns)                    # 分析上下文（方式四）
└── Pyfolio(returns).create_*()         # 可视化报告（方式五）
```

### 内部机制

```
@_populate_from_registry 装饰器
  │
  ├─ 遍历 CLASSMETHOD_REGISTRY (100+ 条目)
  │   └─ setattr(Empyrical, name, _LazyMethod(...))
  │
  └─ 遍历 STATIC_METHODS (6 条目)
      └─ setattr(Empyrical, name, _LazyMethod(...))

_LazyMethod 描述符（首次访问时）
  │
  ├─ Empyrical.sharpe_ratio
  │   └─ __get__(None, Empyrical)
  │       ├─ func = fincore.metrics.ratios.sharpe_ratio  (导入模块)
  │       ├─ setattr(Empyrical, 'sharpe_ratio', staticmethod(func))  (自替换)
  │       └─ return func
  │
  └─ emp.sharpe_ratio  (同上，描述符协议对实例和类访问都生效)

后续访问
  └─ 直接命中 staticmethod(func)，零开销
```
