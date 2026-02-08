### 背景

当前项目已经初步完成，主要基于empyrical和pyfolio这两个包进行了合并和优化

### 任务

1. 分析当前项目有哪些bug
2. 分析当前项目有哪些可以优化的地方
3. 分析一下有哪些函数或者整体优化之后可以提高这个项目的性能
4. 把这些分析的内容写到这个文档里面

### 当前状态

- 全部 1233 个测试用例通过（103 warnings）
- 前两轮（0007、0008）已发现 19 个 Bug 和 22 个优化/性能点，本轮在此基础上继续深入分析

---

## 一、Bug 分析

### Bug 1：`consecutive.py` 中 `resample("M")` 在 pandas 2.2+ 中已弃用

**文件**: `fincore/metrics/consecutive.py` 第 81、265、285 行

```python
monthly_returns = returns.resample("M").apply(lambda g: cum_returns_final(g))
```

项目在 `fincore/constants/periods.py` 中已经做了版本感知的频率映射（pandas 2.2+ 使用 `"ME"` 替代 `"M"`），但 `consecutive.py` 中直接硬编码了 `"M"`，绕过了兼容层。在 pandas 2.2+ 环境下会触发 `FutureWarning`，后续版本可能直接报错。

**修复建议**: 从 `fincore.constants` 导入 `PERIOD_TO_FREQ` 并使用 `PERIOD_TO_FREQ[MONTHLY]` 替代硬编码的 `"M"`。同样，`returns.resample("W")` 也应检查是否需要兼容（`"W"` 目前未变但应统一使用常量）。

---

### Bug 2：`up_alpha_beta` 和 `down_alpha_beta` 在 `alpha_beta.py` 中被定义了两次

**文件**: `fincore/metrics/alpha_beta.py`

- `up_alpha_beta`: 第 345-393 行（第一次），第 598-674 行（第二次）
- `down_alpha_beta`: 第 396-444 行（第一次），第 677-753 行（第二次）

两次定义的实现完全相同（代码级重复），后面的定义会覆盖前面的。虽然不会导致运行错误，但：
1. 增加了维护负担，修改一处容易遗漏另一处
2. 文件体积膨胀（每个函数重复约 50 行）

**修复建议**: 删除第一次定义（345-444 行），只保留带有完整文档字符串的第二次定义。

---

### Bug 3：`max_drawdown_weeks` 和 `max_drawdown_months` 返回的是天数而非对应单位

**文件**: `fincore/metrics/drawdown.py` 第 443-474 行

```python
def max_drawdown_weeks(returns):
    return max_drawdown_days(returns)

def max_drawdown_months(returns):
    return max_drawdown_days(returns)
```

这两个函数声称返回"周数"和"月数"，但实际只是直接委托给 `max_drawdown_days()`，返回的仍然是天数/索引位置差。同样的问题也出现在 `max_drawdown_recovery_weeks` 和 `max_drawdown_recovery_months` 中。

**影响**: 用户调用 `max_drawdown_weeks()` 期望得到周数，实际得到的是天数，结果会差约 7 倍。

**修复建议**: `max_drawdown_weeks` 应返回 `max_drawdown_days(returns) / 7`（或按交易日 5 天/周计算），`max_drawdown_months` 应返回 `max_drawdown_days(returns) / 21`。`recovery` 版本同理。

---

### Bug 4：`second_max_drawdown` 返回的是第二小（最不严重）的回撤而非第二大

**文件**: `fincore/metrics/drawdown.py` 第 555-578 行

```python
sorted_drawdowns = np.sort(drawdown_periods)
return sorted_drawdowns[-2]
```

`get_all_drawdowns` 返回的回撤值是负数（如 -0.1, -0.3, -0.05）。`np.sort` 按升序排列，所以 `sorted_drawdowns[-2]` 取的是倒数第二大的值（即第二个最不负的值），而不是第二严重的回撤。

例如：回撤为 `[-0.3, -0.1, -0.05]`，排序后为 `[-0.3, -0.1, -0.05]`，`[-2]` 是 `-0.1`，这确实是第二大（第二严重）的。但如果我们期望"第二大回撤"是按绝对值排序的第二个，则需要确认语义。

**实际上**: `np.sort([-0.3, -0.1, -0.05])` = `[-0.3, -0.1, -0.05]`，`[-2]` = `-0.1`。这个逻辑是正确的（第二严重）。但 `third_max_drawdown` 中 `sorted_drawdowns[-3]` 对应的是最严重的回撤，而非第三严重。

对于 `third_max_drawdown`：`[-3]` = `-0.3`，即最大回撤而非第三大。

**修复建议**: 应使用 `sorted_drawdowns[0]` 为最大回撤、`sorted_drawdowns[1]` 为第二大、`sorted_drawdowns[2]` 为第三大。或者直接 `sorted(drawdown_periods)` 后取 `[0]`, `[1]`, `[2]`。

---

### Bug 5：`Empyrical.common_sense_ratio` 委托给 `_stats.common_sense_ratio` 而非 `_ratios.common_sense_ratio`

**文件**: `fincore/empyrical.py` 第 750-753 行

```python
@classmethod
def common_sense_ratio(cls, returns=None):
    returns = cls._get_returns(returns)
    return _stats.common_sense_ratio(returns)
```

`stats.common_sense_ratio` 内部又调用 `ratios.common_sense_ratio`，形成了不必要的间接调用链。虽然结果正确，但增加了调用栈深度和维护复杂度。此外，`ratios.py` 的 `__all__` 中同时包含了从 `risk.py` 导入的 `tail_ratio`（第 50 行），这意味着 `from fincore.metrics.ratios import *` 会导出一个并非在 `ratios.py` 中定义的函数。

**修复建议**: `Empyrical.common_sense_ratio` 应直接委托给 `_ratios.common_sense_ratio`。同时，`ratios.py` 的 `__all__` 中应移除 `'tail_ratio'`，因为它属于 `risk.py`。

---

### Bug 6：`timing.py` 中多个函数仍有冗余的生成器类型检查代码

**文件**: `fincore/metrics/timing.py` 第 58-61、105-108、156-157、192-195 行

```python
if not isinstance(returns_aligned, (pd.Series, np.ndarray)):
    returns_aligned = pd.Series(list(returns_aligned))
if not isinstance(factor_aligned, (pd.Series, np.ndarray)):
    factor_aligned = pd.Series(list(factor_aligned))
```

在 0007 轮已修复 `aligned_series()` 返回 `tuple`，不再返回生成器。但 `timing.py` 中的 `treynor_mazuy_timing`、`henriksson_merton_timing`、`market_timing_return`、`cornell_timing` 四个函数仍保留了这些冗余检查。

同样的问题出现在 `fincore/metrics/yearly.py` 第 225-228 行的 `annual_active_return` 和第 314-317 行的 `information_ratio_by_year` 中。

**修复建议**: 移除这些冗余的类型检查代码。

---

### Bug 7：`residual_risk` 使用 `np.var(ddof=1)` 但 `beta_aligned` 使用 `nanmean` 计算方差

**文件**: `fincore/metrics/risk.py` 第 321 行

```python
beta_val = np.cov(excess_returns, excess_factor)[0, 1] / np.var(excess_factor, ddof=1)
```

`residual_risk` 中的 beta 计算使用 `np.cov`（默认 ddof=1）和 `np.var(ddof=1)`，而主 `beta_aligned` 函数使用 `nanmean` 的自定义实现（等效于 ddof=0）。这两种计算方式对于 beta 给出略有不同的结果，尤其在小样本时差异明显。

**修复建议**: `residual_risk` 应复用 `beta_aligned` 来计算 beta，确保一致性。

---

### Bug 8：`Empyrical` 类中 `_get_returns` 和 `_get_factor_returns` 方法无实际用途

**文件**: `fincore/empyrical.py` 第 77-85 行

```python
@classmethod
def _get_returns(cls, returns):
    return returns

@classmethod
def _get_factor_returns(cls, factor_return):
    return factor_return
```

这两个方法只是原样返回输入参数，没有任何实际处理。但类中大量方法都通过 `cls._get_returns(returns)` 调用，增加了不必要的函数调用开销。在原始设计中，这些方法可能用于从实例中获取存储的数据，但当前作为 `@classmethod` 实现时完全无效。

**影响**: 当用户通过 `Empyrical.some_method(returns=None)` 调用时，`_get_returns(None)` 返回 `None`，传入下游计算函数后会抛出错误。这些方法应当在 `returns is None` 时从 `self` 中获取存储的数据，但 `@classmethod` 无法访问 `self`。

**修复建议**: 将这些方法从 `@classmethod` 改为普通实例方法，并在 `returns is None` 时从 `self.returns` 获取数据，或者直接移除这些辅助方法。

---

### Bug 9：`__init__.py` 中无条件导入 `Pyfolio` 会导致缺少可视化依赖时整个包不可用

**文件**: `fincore/__init__.py` 第 13 行

```python
from .pyfolio import Pyfolio
```

`pyfolio.py` 在模块顶层导入了 `seaborn`, `matplotlib`, `IPython.display`, `scipy`, `pytz` 等重型依赖。如果用户只需要 `Empyrical` 的纯计算功能，但环境中未安装 `seaborn` 或 `IPython`，则 `import fincore` 会直接失败。

**修复建议**: 将 `Pyfolio` 改为延迟导入，或在 `__init__.py` 中使用 `try/except` 包裹：

```python
try:
    from .pyfolio import Pyfolio
    __all__ = ["Empyrical", "Pyfolio"]
except ImportError:
    __all__ = ["Empyrical"]
```

---

### Bug 10：`interesting_periods.py` 中 `New Normal` 的结束日期在模块导入时固定

**文件**: `fincore/constants/interesting_periods.py` 第 76-77 行

```python
PERIODS['New Normal'] = (pd.Timestamp('20130101'),
                         pd.Timestamp.now().normalize())
```

`pd.Timestamp.now()` 在模块**首次导入时**求值并永久缓存。如果程序在 2026-01-01 导入 fincore，之后运行到 2026-06-01，`New Normal` 的结束日期仍然是 2026-01-01。

**修复建议**: 将 `PERIODS` 改为属性/函数，或在 `extract_interesting_date_ranges` 中动态更新这个日期。

---

## 二、优化建议

### 优化 1：`consecutive.py` 中的 `resample` 频率应使用常量

如 Bug 1 所述，`consecutive.py` 中硬编码了 `"M"` 和 `"W"` 频率字符串，应统一使用 `fincore.constants.PERIOD_TO_FREQ` 中的版本感知值。同样，`tearsheets/returns.py` 第 670 行也硬编码了 `resample('M')`。

---

### 优化 2：`alpha_beta.py` 中 `up_alpha_beta` 和 `down_alpha_beta` 代码高度重复

这两个函数的实现几乎完全相同，唯一的区别是：
- `up_alpha_beta` 使用 `factor_array > 0` 过滤
- `down_alpha_beta` 使用 `factor_array <= 0` 过滤

应提取为一个通用的 `_conditional_alpha_beta(returns, factor_returns, condition_func, ...)` 私有函数，然后两个公开函数各自传入不同的条件。这可以减少约 100 行重复代码。

---

### 优化 3：`ratios.py` 的 `__all__` 不应导出非本模块定义的函数

**文件**: `fincore/metrics/ratios.py` 第 50 行

```python
'tail_ratio',
```

`tail_ratio` 是从 `risk.py` 导入的（第 27 行 `from fincore.metrics.risk import tail_ratio`），但被包含在 `ratios.py` 的 `__all__` 中。这违反了 Python 的模块导出约定，容易造成维护混乱。

**修复建议**: 从 `ratios.py` 的 `__all__` 中移除 `'tail_ratio'`。

---

### 优化 4：`timing.py` 和 `yearly.py` 中多余的类型检查代码应清理

如 Bug 6 所述，`aligned_series()` 已返回 `tuple`，不再返回生成器。`timing.py` 中 4 个函数和 `yearly.py` 中 2 个函数仍保留冗余的 `isinstance` 检查和 `list()` 转换。这些应移除以提升代码整洁度和微小的性能提升。

---

### 优化 5：`Empyrical` 类方法中存在不一致的设计模式

类中的方法混合使用了两种模式：
1. **直接传参**: 如 `sharpe_ratio(cls, returns, ...)` — 必须显式传入 `returns`
2. **可选传参**: 如 `win_rate(cls, returns=None)` — `returns` 默认为 `None`，通过 `cls._get_returns(returns)` 获取

但由于 `_get_returns` 只是原样返回参数（Bug 8），`returns=None` 的情况会传 `None` 给下游函数并报错。应统一为一种模式：要么所有方法都要求显式传参，要么正确实现实例方法支持从 `self` 获取数据。

---

### 优化 6：`perf_stats_bootstrap` 中 `SIMPLE_STAT_FUNCS` 使用字符串标识函数

**文件**: `fincore/constants/style.py` 和 `fincore/metrics/perf_stats.py`

`SIMPLE_STAT_FUNCS` 用字符串（如 `"annual_return"`, `"stats.skew"`）标识统计函数，然后在 `perf_stats_bootstrap` 中通过字典映射和字符串解析来查找实际函数。这种间接方式：
1. 容易因为字符串拼写错误而静默跳过某个指标
2. 难以通过 IDE 进行重构和跳转
3. `"stats.skew"` 和 `"stats.kurtosis"` 使用了 scipy 的原始函数，与项目中定义的 `skewness`/`kurtosis` 不一致

**修复建议**: 将 `SIMPLE_STAT_FUNCS` 改为直接引用函数对象。

---

### 优化 7：`drawdown.py` 中 `second_max_drawdown_days` 等函数重复调用 `get_all_drawdowns_detailed`

`second_max_drawdown_days`、`second_max_drawdown_recovery_days`、`third_max_drawdown_days`、`third_max_drawdown_recovery_days` 各自独立调用 `get_all_drawdowns_detailed(returns)`，每次都重新计算整个回撤时间序列。如果用户同时需要这些指标（如在 `perf_stats` 中），会导致 4 次重复计算。

**修复建议**: 提供一个 `top_n_drawdown_stats(returns, n=3)` 函数，一次性返回前 N 个回撤的详细信息。

---

### 优化 8：`pyfolio.py` 中 `matplotlib.use('Agg')` 应在条件下使用

**文件**: `fincore/pyfolio.py` 第 22 行

```python
matplotlib.use('Agg')
```

这会强制将 matplotlib 后端设置为非交互式的 `Agg`。如果用户在 Jupyter notebook 中使用 fincore，这会覆盖 notebook 的交互式后端（如 `%matplotlib inline`），导致图表无法在 notebook 中内联显示。

**修复建议**: 只在非交互式环境中设置 `Agg` 后端，或将此设置移至 `create_*_tear_sheet` 函数内部。

---

## 三、性能提升分析

### 性能优化 1：`rolling.py` 中 `roll_*` 函数仍使用 Python 循环

**文件**: `fincore/metrics/rolling.py`

**当前问题**: 上一轮分析已指出此问题，但尚未修复。所有 `roll_alpha`、`roll_beta`、`roll_alpha_beta`、`roll_sharpe_ratio`、`roll_max_drawdown`、`roll_up_capture`、`roll_down_capture` 仍使用 Python `for` 循环逐窗口计算。

**优化方案（分级）**:
1. **roll_sharpe_ratio**: 可完全向量化为 `returns.rolling(window).mean() / returns.rolling(window).std() * np.sqrt(ann_factor)`，类似于已有的 `rolling_sharpe` 函数
2. **roll_beta**: 可使用 `pd.rolling.cov()` 和 `pd.rolling.var()` 组合计算
3. **roll_max_drawdown**: 可使用 `numba` JIT 加速或自定义 numpy stride 操作
4. **roll_up_capture / roll_down_capture**: 需要条件筛选，较难完全向量化，但可通过预分配和减少 Python 层面开销来优化

**预期收益**: 对于 `roll_sharpe_ratio`，向量化后预计速度提升 **50-100 倍**。对于 `roll_beta`，使用 rolling cov/var 预计提升 **20-50 倍**。

---

### 性能优化 2：`consecutive.py` 中各函数独立 `resample` 造成重复计算

**文件**: `fincore/metrics/consecutive.py`

虽然已提供了 `consecutive_stats()` 函数（第 53 行），但各独立函数（如 `max_consecutive_up_weeks`、`max_consecutive_down_weeks`）仍然各自独立进行 `resample("W")` 操作。如果用户依次调用这些函数，`resample` 会被重复执行。

**已有方案**: `consecutive_stats()` 已实现合并 resample，但 `Empyrical` 类中并未暴露此函数，类中的方法仍逐个调用独立函数。

**修复建议**: 在 `Empyrical` 类中添加 `consecutive_stats` 方法，并在 `perf_stats` 等批量统计场景中优先使用。

---

### 性能优化 3：`calmar_ratio` 在 `perf_stats` 中被重复计算

**文件**: `fincore/metrics/perf_stats.py` 第 67-75 行

```python
mdd = max_drawdown(returns)
ann_ret = annual_return(returns, period=period)
stats['Calmar ratio'] = ann_ret / abs(mdd) if mdd != 0 else np.nan
```

`perf_stats` 已优化了 calmar 的计算方式（复用预计算的 `mdd` 和 `ann_ret`），但仍然在计算 `annual_return` 和 `sharpe_ratio` 时分别计算均值和标准差。

**进一步优化**: 预计算 `returns_risk_adj = returns - risk_free` 以及 `std = nanstd(returns)`、`mean = nanmean(returns)`，然后将这些传入 `sharpe_ratio`（通过内部参数）和 `annual_volatility`（通过预计算的 std）。

**预期收益**: 在批量计算多策略绩效时减少约 **20-30%** 的计算时间。

---

### 性能优化 4：`beta_fragility_heuristic` 中不必要的 `list()` 转换

**文件**: `fincore/metrics/risk.py` 第 576-577 行

```python
returns_aligned = list(returns_aligned)
factor_aligned = list(factor_aligned)
```

对齐后的 Series 被转换为 `list`，然后立即重新构建为 `pd.Series`（第 582-583 行）。这个中间的 `list()` 转换完全没有必要，浪费内存和 CPU。

**修复建议**: 直接使用 `aligned_series` 返回的结果，跳过 `list()` 转换。

---

### 性能优化 5：`compute_bayes_cone` 中 `scoreatpercentile` 使用 Python 列表推导

**文件**: `fincore/metrics/bayesian.py` 第 263-264 行

```python
def scoreatpercentile(cum_predictions, p):
    return [np.percentile(cum_predictions[:, i], p) for i in range(cum_predictions.shape[1])]
```

这个列表推导逐列计算百分位数。numpy 的 `np.percentile` 支持 `axis` 参数，可以一次性计算所有列。

**修复建议**:
```python
def scoreatpercentile(cum_predictions, p):
    return np.percentile(cum_predictions, p, axis=0).tolist()
```

**预期收益**: 对于大矩阵，速度提升 **5-10 倍**。

---

### 性能优化 6：`simulate_paths` 中逐行采样效率低

**文件**: `fincore/metrics/bayesian.py` 第 365-370 行

```python
samples = np.empty((num_samples, num_days))
seed = np.random.RandomState(seed=random_seed)
for i in range(num_samples):
    samples[i, :] = is_returns.sample(num_days, replace=True, random_state=seed)
```

使用 Python 循环逐行从 Series 中采样。可以使用 numpy 的随机选择一次性生成所有样本。

**修复建议**:
```python
rng = np.random.RandomState(seed=random_seed)
indices = rng.randint(0, len(is_returns), size=(num_samples, num_days))
samples = np.asarray(is_returns)[indices]
```

**预期收益**: 对于 1000 个样本 × 252 天，预计速度提升 **10-30 倍**。

---

### 性能优化 7：`calc_bootstrap` 中使用 Python 循环进行自助法采样

**文件**: `fincore/metrics/perf_stats.py` 第 218-227 行

```python
for i in range(n_samples):
    idx = np.random.randint(len(returns), size=len(returns))
    returns_i = returns.iloc[idx].reset_index(drop=True)
    ...
    out[i] = func(returns_i, *args, **kwargs)
```

1000 次迭代 × 每次构建新 Series + 调用统计函数，开销很大。对于简单函数（如 `sharpe_ratio`），可以批量预生成所有样本索引，然后利用向量化操作。

**修复建议**: 预生成所有随机索引矩阵 `all_idx = np.random.randint(...)` 形状为 `(n_samples, len(returns))`，对于支持 2D 输入的函数（如 `sharpe_ratio`），直接传入 2D 数组一次性计算。

**预期收益**: 对于简单指标，速度提升 **5-20 倍**。

---

## 四、总结

### Bug 数量：10 个

| 编号 | 严重程度 | 简述 |
|------|---------|------|
| 1 | 中 | `consecutive.py` 硬编码 `resample("M")`，pandas 2.2+ 弃用 |
| 2 | 低 | `up_alpha_beta` 和 `down_alpha_beta` 被定义两次 |
| 3 | 高 | `max_drawdown_weeks/months` 返回天数而非对应单位 |
| 4 | 中 | `third_max_drawdown` 返回最大回撤而非第三大 |
| 5 | 低 | `Empyrical.common_sense_ratio` 间接调用链 + `ratios.__all__` 导出非本模块函数 |
| 6 | 低 | `timing.py` 和 `yearly.py` 中遗留冗余类型检查代码 |
| 7 | 中 | `residual_risk` 中 beta 计算方式与 `beta_aligned` 不一致 |
| 8 | 高 | `_get_returns` 作为 classmethod 无法从实例获取数据，`returns=None` 时传 None 给下游 |
| 9 | 高 | `__init__.py` 无条件导入 Pyfolio，缺少可视化依赖时整个包不可用 |
| 10 | 低 | `interesting_periods.py` 中 `New Normal` 日期在导入时固定 |

### 优化点数量：8 个

主要集中在：
- **频率常量统一**（优化 1）
- **代码去重**（优化 2, 4）
- **模块导出规范**（优化 3）
- **设计模式统一**（优化 5）
- **配置改进**（优化 6, 7, 8）

### 性能提升点数量：7 个

| 编号 | 预期收益 | 简述 |
|------|---------|------|
| 1 | 20-100x | rolling 函数向量化（分级实现） |
| 2 | 2-3x | consecutive 函数合并 resample |
| 3 | 20-30% | perf_stats 共享中间计算结果 |
| 4 | 微量 | beta_fragility_heuristic 移除不必要的 list 转换 |
| 5 | 5-10x | compute_bayes_cone 使用 numpy axis 参数 |
| 6 | 10-30x | simulate_paths 批量采样替代循环 |
| 7 | 5-20x | calc_bootstrap 批量预生成样本索引 |