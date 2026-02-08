### 背景

当前项目已经初步完成，主要基于empyrical和pyfolio这两个包进行了合并和优化

### 任务

1. 分析当前项目有哪些bug
2. 分析当前项目有哪些可以优化的地方
3. 分析一下有哪些函数或者整体优化之后可以提高这个项目的性能
4. 把这些分析的内容写到这个文档里面

---

## 一、Bug分析

### Bug 1: `beta_aligned` 忽略了 `risk_free` 参数

**文件**: `fincore/metrics/alpha_beta.py` 第71行

```python
_ = risk_free  # API compatibility; not used
```

**问题**: `beta_aligned` 接受 `risk_free` 参数但显式地忽略了它。在CAPM模型中，beta应该使用超额收益（returns - risk_free）来计算。当 `risk_free != 0` 时，beta计算结果不正确。

**对比**: `_conditional_alpha_beta` 函数（第400-410行）正确地使用了 `risk_free` 来计算超额收益后再计算beta，两者行为不一致。

**修复建议**: 在 `beta_aligned` 中，如果 `risk_free != 0`，应当先计算超额收益再进行beta估计，或者在文档中明确说明该函数不使用risk_free参数。

---

### Bug 2: `beta_aligned` 长度检查不一致

**文件**: `fincore/metrics/alpha_beta.py` 第90行

```python
if len(returns) < 1 or len(factor_returns) < 2:
```

**问题**: 对 `returns` 的检查是 `< 1`，但对 `factor_returns` 的检查是 `< 2`。计算有意义的beta需要两个序列都至少有2个数据点。当 `len(returns) == 1` 且 `len(factor_returns) >= 2` 时，函数不会提前返回NaN，而是在后续计算中产生不正确的结果。

**修复建议**: 统一为 `if len(returns) < 2 or len(factor_returns) < 2:`

---

### Bug 3: `omega_ratio` 使用Python内置 `sum()` 而非 `np.sum()`

**文件**: `fincore/metrics/ratios.py` 第389-390行

```python
numer = sum(returns_less_thresh[returns_less_thresh > 0.0])
denom = -1.0 * sum(returns_less_thresh[returns_less_thresh < 0.0])
```

**问题**: 使用Python内置 `sum()` 而不是 `np.sum()`。对于包含NaN的数据，`sum()` 会传播NaN（与numpy行为一致），但对于空数组，`sum()` 返回0而 `np.sum()` 也返回0。主要问题在于效率和一致性——项目中其他地方都使用 `np.sum()` 或 `nansum()`。

**修复建议**: 替换为 `np.nansum()` 以保持一致性和处理NaN值。

---

### Bug 4: `conditional_sharpe_ratio` 硬编码年化因子为252

**文件**: `fincore/metrics/ratios.py` 第307行

```python
return mean_ret / std_ret * np.sqrt(252)
```

**问题**: 与其他ratio函数不同（这些函数都接受 `period` 和 `annualization` 参数），此函数硬编码了252作为年化因子。如果输入数据不是日频率（例如周频或月频），结果将不正确。

**修复建议**: 添加 `period` 和 `annualization` 参数，使用 `annualization_factor()` 函数获取正确的年化因子。

---

### Bug 5: `information_ratio` 未处理零标准差情况

**文件**: `fincore/metrics/ratios.py` 第428-430行

```python
mean_excess_return = super_returns.mean()
std_excess_return = super_returns.std(ddof=1)
ir = (mean_excess_return * ann_factor) / (std_excess_return * np.sqrt(ann_factor))
```

**问题**: 当 `std_excess_return` 为0时，会产生inf或nan，但没有显式处理。其他ratio函数（如 `sharpe_ratio`、`sortino_ratio`）都使用了 `np.errstate` 来抑制除零警告。

**修复建议**: 添加零标准差的显式检查，或使用 `np.errstate` 包装除法运算。

---

### Bug 6: `capture` 函数未对齐 returns 和 factor_returns

**文件**: `fincore/metrics/ratios.py` 第863-903行

**问题**: `capture` 函数分别对 `returns` 和 `factor_returns` 计算年化收益，使用各自的长度来计算 `num_years`。如果两个序列的日期范围不同，年化计算将不一致。虽然 `up_capture` 和 `down_capture` 会先调用 `aligned_series`，但直接调用 `capture` 时不会对齐。

**修复建议**: 在 `capture` 函数开头添加 `returns, factor_returns = aligned_series(returns, factor_returns)`。

---

### Bug 7: `extract_round_trips` 截断小数股数

**文件**: `fincore/metrics/round_trips.py` 第204行

```python
remaining = abs(int(row.amount))
```

**问题**: 使用 `int()` 将交易数量截断为整数，对于支持小数股交易的市场（如加密货币、某些券商的股票拆分交易），这会导致PnL计算不准确。

**修复建议**: 使用 `abs(row.amount)` 并在后续逻辑中支持浮点数量，或者使用 `round()` 进行四舍五入。

---

### Bug 8: `add_closing_transactions` 除零风险

**文件**: `fincore/metrics/round_trips.py` 第307-309行

```python
ending_amount = txn_sym.amount.sum()
ending_price = ending_val / ending_amount
```

**问题**: 如果某个symbol的所有交易amount之和为0（例如买入和卖出完全对冲），则 `ending_amount` 为0，导致 `ZeroDivisionError`。

**修复建议**: 添加 `if ending_amount == 0: continue` 的检查。

---

### Bug 9: `adjust_returns_for_slippage` 除零风险

**文件**: `fincore/metrics/transactions.py` 第362行

```python
adjusted_returns = returns * adjusted_pnl / pnl
```

**问题**: 当 `pnl` 中包含0值时（某天盈亏为零），会导致除零错误。

**修复建议**: 使用 `pnl.replace(0, np.nan)` 或添加零值保护。

---

### Bug 10: `perf_stats` 中 Calmar ratio 计算逻辑与 `calmar_ratio` 函数不一致

**文件**: `fincore/metrics/perf_stats.py` 第75行

```python
stats['Calmar ratio'] = ann_ret / abs(mdd) if mdd != 0 else np.nan
```

**问题**: 这里检查 `mdd != 0`，但 `calmar_ratio()` 函数检查的是 `max_dd < 0`。当 `mdd > 0`（理论上不会发生，但如果数据有问题可能出现）时，`perf_stats` 会计算出一个值，而 `calmar_ratio` 会返回NaN。

**修复建议**: 直接调用 `calmar_ratio(returns, period=period)` 以保持一致性。

---

### Bug 11: 代码重复定义 - `nanmean`/`nanstd` 等函数

**文件**: `fincore/utils/math_utils.py` 和 `fincore/utils/common_utils.py`

**问题**: bottleneck包装函数 (`nanmean`, `nanstd`, `nansum`, `nanmax`, `nanmin`, `nanargmax`, `nanargmin`) 在两个文件中完全重复定义。`math_utils.py` 通过 `__init__.py` 导出，而 `common_utils.py` 也定义了相同的函数。这可能导致导入混乱。

**修复建议**: 在 `common_utils.py` 中删除重复定义，改为从 `math_utils` 导入。

---

### Bug 12: `roll`, `up`, `down`, `rolling_window` 函数重复定义

**文件**: `fincore/utils/data_utils.py` 和 `fincore/utils/common_utils.py`

**问题**:
- `roll`, `up`, `down` 在两个文件中完全重复。
- `rolling_window` 在两个文件中有不同的实现：`data_utils.py` 中仅支持1D数组，`common_utils.py` 中支持多维数组。由于 `__init__.py` 同时导入两者，后导入的会覆盖前者，可能导致意外行为。

**修复建议**: 整合到一个文件中，删除另一个的重复定义。

---

### Bug 13: `common_utils.py` 中 `AssertionError` 拼写错误

**文件**: `fincore/utils/common_utils.py` 第80行和第155行

```python
except AssertionError as e:
```

**问题**: Python内置异常名为 `AssertionError`（A-s-s-e-r-t-i-o-n-E-r-r-o-r），需确认代码中的拼写是否正确。如果拼写为 `AssertionError`（缺少字母），则该except子句永远不会捕获到断言错误，错误会直接抛出而非被优雅处理。

**修复建议**: 确保拼写为 `AssertionError`。

---

## 二、优化机会分析

### 优化 1: `rolling.py` 中的滚动计算使用Python循环

**文件**: `fincore/metrics/rolling.py` 所有 `roll_*` 函数

**问题**: 所有 `roll_alpha`, `roll_beta`, `roll_sharpe_ratio`, `roll_max_drawdown` 等函数都使用Python `for` 循环逐窗口调用指标函数。对于大数据集（例如10年日度数据约2520个数据点），这非常缓慢。

```python
for i in range(n):
    out[i] = alpha(returns_aligned.iloc[i:i + window], ...)
```

**优化建议**:
- 对于可以增量计算的指标（如rolling mean, rolling std），使用 `pandas.rolling()` 或 `bottleneck.move_mean()`/`move_std()`。
- 对于复杂指标（如alpha, beta），考虑使用矩阵化的滑动窗口运算或 `numba` JIT编译。

---

### 优化 2: `consecutive.py` 中重复的resample操作

**文件**: `fincore/metrics/consecutive.py`

**问题**: `max_consecutive_up_weeks`, `max_consecutive_down_weeks`, `max_consecutive_up_months`, `max_consecutive_down_months` 各自独立调用 `resample()`。在常见使用场景中（例如生成完整的统计报告），相同数据会被重复resample。

**优化建议**: 已有 `consecutive_stats()` 函数可以批量计算，建议在 `Empyrical` 类中优先调用批量版本，或在个别函数中缓存resample结果。

---

### 优化 3: `sterling_ratio` 和 `burke_ratio` 高度重复的代码

**文件**: `fincore/metrics/ratios.py` 第605-725行

**问题**: 两个函数都包含几乎相同的：
1. `get_all_drawdowns(returns)` 调用
2. 空drawdown处理逻辑
3. 年化收益计算代码（`ann_factor`, `num_years`, `ending_value`, `ann_ret`）

```python
# 重复出现在两个函数中：
ann_factor = annualization_factor(period, annualization)
num_years = len(returns) / ann_factor
ending_value = cum_returns_final(returns, starting_value=1)
ann_ret = ending_value ** (1 / num_years) - 1
```

**优化建议**: 提取共享的 `_compute_annualized_return` 辅助函数和 `_get_drawdown_risk` 辅助函数。

---

### 优化 4: `kappa_three_ratio` 重复计算年化收益3次

**文件**: `fincore/metrics/ratios.py` 第728-795行

**问题**: 在同一个函数中，年化收益的计算逻辑出现了3次（第776-778行、第786-788行、第791-793行）。

**优化建议**: 将年化收益计算提到函数开头，只计算一次。

---

### 优化 5: `up_down_capture` 重复调用 `aligned_series`

**文件**: `fincore/metrics/ratios.py` 第976-1003行

**问题**: `up_down_capture` 调用 `up_capture` 和 `down_capture`，两者各自独立调用 `aligned_series`。同一数据对齐了两次。

**优化建议**: 在 `up_down_capture` 中先对齐一次，然后传递已对齐的数据给 `up_capture` 和 `down_capture`。

---

### 优化 6: `annual_alpha` 和 `annual_beta` 共享相同的循环结构

**文件**: `fincore/metrics/alpha_beta.py` 第491-586行

**问题**: 两个函数有完全相同的结构：按年分组、遍历、调用单个指标函数。如果需要同时计算两者（常见需求），数据被遍历两次。

**优化建议**: 提供一个 `annual_alpha_beta` 函数，一次遍历同时计算alpha和beta。

---

### 优化 7: `perf_stats_bootstrap` 中bootstrap采样效率低

**文件**: `fincore/metrics/perf_stats.py` 第191-227行

**问题**: `calc_bootstrap` 使用Python循环进行1000次bootstrap采样。

```python
for i in range(n_samples):
    idx = np.random.randint(len(returns), size=len(returns))
    returns_i = returns.iloc[idx].reset_index(drop=True)
    ...
```

**优化建议**:
- 对于简单统计量（如mean, std），可以使用向量化的bootstrap：一次性生成所有随机索引矩阵，然后使用numpy向量化运算。
- 对于复杂统计量，考虑使用 `joblib.Parallel` 进行并行计算。

---

### 优化 8: `gen_drawdown_table` 使用 `inplace=True` 修改数据

**文件**: `fincore/metrics/drawdown.py` 第303行

```python
underwater.drop(underwater[peak:recovery].index[1:-1], inplace=True)
```

**问题**: `inplace=True` 是pandas已弃用的模式，且在循环中修改Series效率较低。

**优化建议**: 使用非inplace操作或重写逻辑以避免循环中修改Series。

---

## 三、性能提升分析

### 性能 1: 为频繁调用的函数添加缓存

**涉及函数**: `annualization_factor()`

**说明**: `annualization_factor()` 在几乎每个指标函数中都被调用，但输入参数通常相同（如 `period="daily"`, `annualization=None`）。

**建议**: 使用 `functools.lru_cache` 缓存结果。由于参数是简单的字符串和数值，缓存非常有效。

```python
from functools import lru_cache

@lru_cache(maxsize=32)
def annualization_factor(period=DAILY, annualization=None):
    ...
```

---

### 性能 2: 使用stride tricks向量化滚动窗口计算

**涉及模块**: `fincore/metrics/rolling.py`

**说明**: 当前滚动计算使用Python循环，对于252天窗口和2520天数据，需要执行约2268次循环迭代，每次迭代都调用完整的指标函数。

**建议**: 对于 `rolling_volatility` 和 `rolling_sharpe` 这类基于简单统计量的函数，使用 `numpy.lib.stride_tricks.as_strided` 创建滑动窗口视图，然后对整个矩阵进行向量化运算。

```python
# 示例：向量化rolling volatility
from numpy.lib.stride_tricks import sliding_window_view
windows = sliding_window_view(returns.values, window)
rolling_vol = np.std(windows, axis=1, ddof=1) * np.sqrt(252)
```

---

### 性能 3: 使用numba JIT编译热点函数

**涉及函数**: `max_drawdown`, `beta_aligned`, `sharpe_ratio`

**说明**: 这些函数是纯数值计算（numpy操作），非常适合numba JIT编译。对于大型数据集或频繁调用的场景（如bootstrap、rolling计算），JIT编译可以提升5-50倍性能。

**建议**: 为核心计算逻辑创建numba版本，保留原始版本作为fallback。

```python
try:
    from numba import njit
    
    @njit
    def _max_drawdown_numba(returns_array):
        cumulative = 1.0
        peak = 1.0
        max_dd = 0.0
        for r in returns_array:
            cumulative *= (1 + r)
            if cumulative > peak:
                peak = cumulative
            dd = (cumulative - peak) / peak
            if dd < max_dd:
                max_dd = dd
        return max_dd
except ImportError:
    pass
```

---

### 性能 4: `perf_stats` 中批量计算共享中间结果

**文件**: `fincore/metrics/perf_stats.py`

**说明**: `perf_stats` 调用多个独立的指标函数，但许多函数共享相同的中间结果：
- `cum_returns` 被 `annual_return`, `calmar_ratio`, `cum_returns_final` 分别计算
- `annualization_factor` 被多个函数重复调用
- 标准差被 `sharpe_ratio` 和 `annual_volatility` 分别计算

**建议**: 预计算共享中间结果，传递给需要它们的函数。当前代码已经部分做了这点（预计算 `mdd` 和 `ann_ret`），可以扩展到更多指标。

---

### 性能 5: 预计算和缓存drawdown序列

**涉及模块**: `fincore/metrics/drawdown.py`

**说明**: 累积收益和drawdown序列在 `max_drawdown`, `get_all_drawdowns`, `gen_drawdown_table`, `get_max_drawdown_underwater`, `get_top_drawdowns` 等多个函数中被重复计算。

**建议**: 提供一个 `DrawdownAnalysis` 类或缓存机制，一次计算drawdown序列，多次使用。

---

### 性能 6: 使用 `bottleneck` 的移动窗口函数

**说明**: 项目已经使用了 `bottleneck` 库的 `nanmean`/`nanstd`，但没有使用其移动窗口函数（`move_mean`, `move_std`, `move_sum`）。这些函数比pandas的rolling实现快10-100倍。

**建议**: 在 `rolling_volatility` 和 `rolling_sharpe` 中使用 `bottleneck.move_std()` 和 `bottleneck.move_mean()`。

```python
try:
    import bottleneck as bn
    def rolling_volatility(returns, window):
        return bn.move_std(returns.values, window, ddof=1) * np.sqrt(252)
except ImportError:
    # fallback to pandas
    ...
```

---

### 性能 7: 避免 `pd.Series.apply` 中使用lambda

**涉及模块**: `fincore/metrics/consecutive.py`, `fincore/metrics/yearly.py`

**说明**: `.resample().apply(lambda g: cum_returns_final(g))` 中的lambda函数会阻止pandas的内部优化。

```python
# 当前代码
weekly_returns = returns.resample(...).apply(lambda g: cum_returns_final(g))

# 更快的写法
weekly_returns = returns.resample(...).apply(cum_returns_final)
```

**建议**: 直接传递函数引用而不是lambda包装。

---

### 性能 8: `Empyrical` 类使用 `@classmethod` 无必要

**文件**: `fincore/empyrical.py`

**说明**: `Empyrical` 类的所有方法都是 `@classmethod`，没有实例状态。使用类方法会增加方法查找的开销（虽然很小），且从设计角度看，使用模块级函数更符合Python惯例。

**建议**: 如果不需要继承/多态（`Pyfolio` 继承了 `Empyrical`），可以考虑将其改为模块级函数。但由于 `Pyfolio` 继承关系，当前设计有其合理性，此项优先级较低。

---

### 性能 9: `aligned_series` 对已对齐数据的冗余操作

**文件**: `fincore/metrics/basic.py`

**说明**: 许多内部调用传递的数据已经对齐（例如在 `_conditional_alpha_beta` 中先对齐再调用 `beta`），但被调用函数会再次调用 `aligned_series`。

**建议**: 为内部调用路径提供 `_aligned` 版本的函数（类似已有的 `beta_aligned`、`alpha_aligned` 模式），跳过不必要的对齐操作。

---

### 性能 10: 统一惰性导入策略

**涉及**: 全项目

**说明**: 项目中部分函数使用惰性导入（函数内import），部分在文件顶部导入。例如：
- `bayesian.py` 在函数内导入 `pymc`（正确，因为是可选依赖）
- `ratios.py` 中的 `sortino_ratio` 在函数内导入 `downside_risk`（避免循环导入）
- `stats.py` 在顶部导入 `stability_of_timeseries`

**建议**: 制定统一的导入策略：
1. 可选重型依赖（pymc, scipy.optimize）：惰性导入
2. 内部模块避免循环导入：惰性导入
3. 其他内部模块：顶部导入

---

## 四、总结

| 类别 | 数量 | 严重程度 |
|------|------|----------|
| Bug | 13 | 高：Bug 1, 7, 8, 9；中：Bug 2-6, 10-13 |
| 优化机会 | 8 | 中：代码质量和可维护性改进 |
| 性能提升 | 10 | 高：性能 1-3, 6；中：性能 4-5, 7-10 |

**优先修复建议**:
1. **最高优先级**: Bug 1（beta计算错误）、Bug 7-9（除零和数据截断）
2. **高优先级**: 性能1（annualization_factor缓存）、性能6（bottleneck移动窗口）、Bug 11-12（代码重复）
3. **中优先级**: 优化1（rolling计算向量化）、性能3（numba JIT）、Bug 4-6（边界情况处理）
4. **低优先级**: 性能8（类设计）、性能10（导入策略统一）

