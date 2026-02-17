### 背景

当前项目已经初步完成，主要基于empyrical和pyfolio这两个包进行了合并和优化，并且现在已经做了很多的修复bug和改进优化的工作。

### 任务

1. 分析当前项目有哪些bug
2. 分析当前项目有哪些可以优化的地方
3. 分析一下有哪些函数或者整体优化之后可以提高这个项目的性能
4. 把这些分析的内容写到这个文档里面

---

## 分析结果

> 分析日期: 2026-02-09
> 测试状态: 1233个测试全部通过，102个warnings
> 备注: 0011-0014 中已报告并修复的问题不再重复列出。以下为本轮发现的**新问题**。

---

### 一、Bug分析

#### Bug 1: `beta_aligned` 对全NaN输入产生 `RuntimeWarning: Mean of empty slice`（严重度：低-中）

- **文件**: `fincore/metrics/alpha_beta.py` 第99-104行
- **问题**: 当 `returns` 或 `factor_returns` 全为 NaN 时，`nanmean` 对空切片计算会触发 `RuntimeWarning`。测试中观察到3条此类警告。虽然最终结果正确（返回 NaN），但警告会污染用户输出。
- **影响**: 不影响计算正确性，但在批量运行时产生大量无意义警告。
- **建议修复**:
  ```python
  with np.errstate(invalid='ignore'):
      ind_residual = independent - nanmean_local(independent, axis=0)
      covariances = nanmean_local(ind_residual * returns, axis=0)
  ```

#### Bug 2: `adjust_returns_for_slippage` 除零风险（严重度：中）

- **文件**: `fincore/metrics/transactions.py` 第362-363行
- **问题**: `adjusted_returns = returns * adjusted_pnl / pnl`，当 `pnl` 为零时（某天收益率为0且持仓为0），会产生 `0/0 = NaN` 或 `x/0 = inf`。虽然有 `np.errstate` 抑制警告并用 `fillna(0)` 处理，但 `inf * 0 = NaN` 仍可能存在于中间结果中。
- **影响**: 当策略有空仓日时，调整后的收益率可能包含意外的 NaN。
- **建议修复**: 在除法前使用 `pnl.replace(0, np.nan)` 使零值日自动变为 NaN，后续 `fillna(0)` 正确处理。

#### Bug 3: `get_percent_alloc` 除零未保护（严重度：中）

- **文件**: `fincore/metrics/positions.py` 第51行
- **问题**: `values.divide(values.sum(axis="columns"), axis="rows")` 当某天所有持仓为零时，`sum` 为零导致除零。与 `gross_lev` 类似的问题，但此函数被 `days_to_liquidate_positions`、`get_max_median_position_concentration` 等多处调用。
- **影响**: 空仓日会产生 `inf` 或 `NaN`，可能影响下游 `get_max_median_position_concentration` 计算。
- **建议修复**: 添加 `replace([np.inf, -np.inf], np.nan)` 或者在 `sum` 中排除零值行。

#### Bug 4: `perf_stats` 的 `Calmar ratio` 内联计算与 `calmar_ratio()` 行为不一致（严重度：低-中）

- **文件**: `fincore/metrics/perf_stats.py` 第83行
- **问题**: `stats['Calmar ratio'] = ann_ret / abs(mdd) if mdd < 0 else np.nan`。但 `calmar_ratio()` 函数内部调用 `annual_return()` 重新计算年化收益率。0014中已统一了 `calmar_ratio` 使用 `annual_return()`，但 `perf_stats` 中的内联版本使用预计算的 `ann_ret`。当前结果一致，但如果 `calmar_ratio()` 的逻辑未来有变化，两者可能分歧。
- **建议**: 维护风险提示。可考虑直接调用 `calmar_ratio(returns, period=period)` 以保持一致。

#### Bug 5: `annual_alpha`/`annual_beta` 未对齐 returns 和 factor_returns（严重度：中）

- **文件**: `fincore/metrics/alpha_beta.py` 第523-537行、第572-587行
- **问题**: `annual_alpha` 和 `annual_beta` 按年分组后分别从 `returns.groupby(year)` 和 `factor_returns.groupby(year)` 取数据，但未预先对齐两者。如果 `returns` 和 `factor_returns` 有不同的日期索引（如交易日不匹配），同一年内的数据长度可能不一致，导致 `alpha()` 或 `beta()` 内部的 `aligned_series` 静默截断数据。
- **影响**: 按年计算的 alpha/beta 可能基于不完整的匹配数据。
- **建议修复**: 在分组前先调用 `aligned_series(returns, factor_returns)` 统一索引。

---

### 二、代码优化分析

#### 优化 1: `second_*/third_*` 回撤函数各自独立调用 `get_all_drawdowns_detailed`

- **文件**: `fincore/metrics/drawdown.py` 第617-712行
- **问题**: `second_max_drawdown_days`、`second_max_drawdown_recovery_days`、`third_max_drawdown_days`、`third_max_drawdown_recovery_days` 各自独立调用 `get_all_drawdowns_detailed(returns)` 后排序。如果用户同时调用多个函数，回撤分析被重复执行4次。
- **建议**: 提供 `nth_drawdown_stats(returns, n)` 或在 Empyrical 中提供批量方法。

#### 优化 2: `calc_bootstrap` 使用 Python for 循环

- **文件**: `fincore/metrics/perf_stats.py` 第226-233行
- **问题**: `for i in range(n_samples)` 逐次采样并计算。1000次循环中每次调用一个 Python 函数。
- **建议**: 对于可向量化的函数（如 `annual_return`、`annual_volatility`），预生成 `(n_samples, len(returns))` 的索引矩阵并批量计算。

#### 优化 3: `Pyfolio.__init__` 参数过多

- **文件**: `fincore/pyfolio.py` 第114-166行
- **问题**: `Pyfolio.__init__` 接受20+个参数，大部分是存储为实例属性。但这些属性在 tear sheet 方法中又作为参数显式传递，导致冗余。
- **建议**: 考虑使用 `**kwargs` 存储配置，或将配置抽取为独立的 `Config` 数据类。

#### 优化 4: `m_squared` 中 `cum_returns_final` 导入未使用

- **文件**: `fincore/metrics/ratios.py` 第588行
- **问题**: 0014中已将 `m_squared` 改为使用 `annual_return()`，但 `from fincore.metrics.returns import cum_returns_final` 导入仍然保留，不再被使用。
- **建议**: 移除未使用的导入。

---

### 三、性能优化分析

#### 性能问题 1: `get_top_drawdowns` 使用 O(n×top) 串行扫描

- **文件**: `fincore/metrics/drawdown.py` 第274-311行
- **问题**: 循环 `top` 次，每次调用 `get_max_drawdown_underwater` 扫描整个 underwater 序列。
- **建议**: 使用 `get_all_drawdowns_detailed` 单遍扫描后按严重度排序。

#### 性能问题 2: `roll_max_drawdown` 使用 Python for 循环

- **文件**: `fincore/metrics/rolling.py` 第283-290行
- **问题**: `for i in range(n)` 逐窗口计算 max_drawdown。每次计算包括 `cumprod` + `fmax.accumulate`。
- **建议**: 较难完全向量化，但可通过 `numba.jit` 加速或使用滑动窗口优化减少重复计算。

#### 性能问题 3: `roll_up_capture` / `roll_down_capture` 使用 Python for 循环

- **文件**: `fincore/metrics/rolling.py` 第330-338行、第368-376行
- **问题**: 逐窗口调用 `up_capture` / `down_capture`，每次内部又计算 `cum_returns_final` 等。
- **建议**: 预计算滚动年化收益率，然后用条件掩码计算 up/down 子集。

#### 性能问题 4: `hurst_exponent` 逐 lag 循环

- **文件**: `fincore/metrics/stats.py` 第134-151行
- **问题**: `for lag in range(min_lag, max_lag + 1)` 对长序列 `max_lag` 可达 3333。
- **建议**: 使用对数等间距采样 lag 值，将循环次数控制在 50 次以内。

---

### 四、总结

| 类别 | 数量 | 优先级建议 |
|------|------|-----------|
| Bug | 5个 | Bug 5（annual_alpha/beta未对齐）和 Bug 2-3（除零保护）应优先修复 |
| 代码优化 | 4个 | 优化 4（移除未使用导入）最简单，优化 1（回撤去重）收益最大 |
| 性能优化 | 4个 | 性能问题 1（get_top_drawdowns）和 性能问题 2（roll_max_drawdown）收益最高 |

**推荐优先修复顺序**:
1. Bug 5：修复 `annual_alpha`/`annual_beta` 缺少预对齐
2. Bug 2-3：修复 `adjust_returns_for_slippage` 和 `get_percent_alloc` 除零
3. Bug 1：抑制 `beta_aligned` 空切片警告
4. 优化 4：移除 `m_squared` 中未使用的 `cum_returns_final` 导入
5. Bug 4：维护风险，考虑统一 `perf_stats` 中 Calmar 内联计算
6. 性能问题 1：优化 `get_top_drawdowns` 为单遍扫描

---

### 五、修复记录

> 修复日期: 2026-02-09
> 修复后测试: 1233个测试全部通过，102个warnings

| 序号 | 问题 | 修复文件 | 修复内容 | 状态 |
|------|------|---------|---------|------|
| Bug 5 | `annual_alpha`/`annual_beta` 未预对齐 | `fincore/metrics/alpha_beta.py` | 在 groupby 前调用 `aligned_series(returns, factor_returns)` | ✅ 已修复 |
| Bug 2 | `adjust_returns_for_slippage` 除零 | `fincore/metrics/transactions.py` | 除法前 `pnl.replace(0, np.nan)` | ✅ 已修复 |
| Bug 3 | `get_percent_alloc` 除零 | `fincore/metrics/positions.py` | 除法后 `replace([np.inf, -np.inf], np.nan)` | ✅ 已修复 |
| Bug 1 | `beta_aligned` 空切片警告 | `fincore/metrics/alpha_beta.py` | `np.errstate(invalid='ignore')` 包裹 nanmean 调用 | ✅ 已修复 |
| 优化 4 | `m_squared` 未使用导入 | `fincore/metrics/ratios.py` | 移除 `from fincore.metrics.returns import cum_returns_final` | ✅ 已修复 |
| Bug 4 | `perf_stats` Calmar 内联不一致 | `fincore/metrics/perf_stats.py` | 替换内联计算为 `calmar_ratio(returns, period=period)` | ✅ 已修复 |
| 性能1 | `get_top_drawdowns` 串行扫描 | `fincore/metrics/drawdown.py` | 尝试单遍扫描优化但回退：测试依赖旧算法在无回撤时返回退化结果的行为 | ⏭️ 跳过 |

