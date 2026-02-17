### 背景

当前项目已经初步完成，主要基于empyrical和pyfolio这两个包进行了合并和优化

### 任务

1. 分析当前项目有哪些bug
2. 分析当前项目有哪些可以优化的地方
3. 分析一下有哪些函数或者整体优化之后可以提高这个项目的性能
4. 把这些分析的内容写到这个文档里面

---

## 分析结果

> 分析日期: 2026-02-09
> 测试状态: 1233个测试全部通过，102个warnings
> 备注: 0011和0012中已报告的多个bug已修复（`beta()`传递`risk_free`、`get_max_drawdown_underwater`使用`pd.NaT`、`max_drawdown_days`区分日历日和交易日、`Pyfolio.__init__`调用`super().__init__`、`_get_returns`改用`_dual_method`描述器、`normalize`除零检查、`aggregate_returns`类型检查和错误信息修正等）。以下为本轮发现的**新问题**。

---

### 一、Bug分析

#### Bug 1: `annual_return` 对 DataFrame 输入时 `ending_value <= 0` 检查会崩溃（严重度：高）

- **文件**: `fincore/metrics/yearly.py` 第71-73行
- **问题**: 当 `returns` 为 DataFrame 时，`cum_returns_final(returns, starting_value=1)` 返回一个 Series（每列一个值）。后续 `if ending_value <= 0:` 对 Series 执行布尔判断，会抛出 `ValueError: The truth value of a Series is ambiguous`。
- **复现**: `Empyrical.annual_return(pd.DataFrame({'a': [-0.5, -0.8], 'b': [0.1, 0.2]}))`
- **影响**: 任何包含"累计亏损超过100%"列的 DataFrame 都会导致崩溃。
- **建议修复**:
  ```python
  if isinstance(ending_value, (pd.Series, np.ndarray)):
      result = ending_value.copy()
      mask = ending_value <= 0
      result[mask] = -1.0
      result[~mask] = result[~mask] ** (1 / num_years) - 1
      return result
  else:
      if ending_value <= 0:
          return -1.0
      return ending_value ** (1 / num_years) - 1
  ```

#### Bug 2: `calmar_ratio` 不处理 `ending_value <= 0` 的情况（严重度：中）

- **文件**: `fincore/metrics/ratios.py` 第340-356行
- **问题**: `calmar_ratio` 内部计算 `ending_value = cum_returns_final(returns, starting_value=1)`，然后直接执行 `ending_value ** (1 / num_years) - 1`。当 `ending_value <= 0` 时（策略亏损100%+），负数的非整数次方会产生 `nan`，导致 calmar_ratio 为 `nan`。而同样条件下 `annual_return()` 会返回 `-1.0`。
- **影响**: 直接调用 `Empyrical.calmar_ratio()` 和通过 `perf_stats()` 间接计算的 calmar_ratio 结果不一致。`perf_stats` 使用 `ann_ret / abs(mdd)` 绕过了这个问题。
- **建议修复**: 在 `calmar_ratio` 内部复用 `annual_return()` 而不是重新计算年化收益率。

#### Bug 3: `perf_attrib` 实例方法不使用实例的 `positions`/`factor_returns`/`factor_loadings`（严重度：中）

- **文件**: `fincore/empyrical.py` 第675-679行
- **问题**: `perf_attrib` 方法使用 `self._get_returns(returns)` 对 `returns` 进行实例回退，但 `positions`、`factor_returns`、`factor_loadings` 参数直接传递给底层函数，不会回退到实例存储的数据。用户在 `__init__` 中设置了这些数据后，调用 `emp.perf_attrib()` 仍需手动传入。
- **影响**: 违背面向对象设计意图，`positions`、`factor_returns`、`factor_loadings` 在实例中存储了但不被自动使用。
- **建议修复**: 添加类似 `_get_positions`、`_get_factor_loadings` 的回退方法，或在 `perf_attrib` 内部手动检查：
  ```python
  if positions is None and hasattr(self, 'positions'):
      positions = self.positions
  ```

#### Bug 4: `r_cubed` 不处理 NaN 值（严重度：中）

- **文件**: `fincore/metrics/stats.py` 第460-465行
- **问题**: `r_cubed` 调用 `np.corrcoef(returns_aligned, factor_aligned)[0, 1]`，但 `np.corrcoef` 不忽略 NaN。当 `aligned_series` 返回的序列包含 NaN（因索引不完全匹配时的 outer join），相关系数将为 NaN。
- **影响**: 当策略和基准的交易日不完全重合时，`r_cubed` 总是返回 NaN。
- **建议修复**: 在计算前过滤 NaN：
  ```python
  mask = ~(np.isnan(np.asarray(returns_aligned)) | np.isnan(np.asarray(factor_aligned)))
  if mask.sum() < 2:
      return np.nan
  correlation = np.corrcoef(np.asarray(returns_aligned)[mask], np.asarray(factor_aligned)[mask])[0, 1]
  ```

#### Bug 5: `rolling_regression` 返回非年化 alpha，与 `roll_alpha` 返回年化 alpha 不一致（严重度：中）

- **文件**: `fincore/metrics/rolling.py`
- **问题**:
  - `roll_alpha`（第84行）调用 `alpha()` 返回**年化** alpha
  - `rolling_regression`（第514行）计算 `rolling_mean_ret - rolling_beta * rolling_mean_fac` 返回**日频** alpha（未年化）
- **影响**: 用户调用两个函数期望得到可比较的 alpha 值，但实际数值差异巨大（年化因子 252 倍量级）。API 文档未明确说明此区别。
- **建议修复**: 在 `rolling_regression` 的 docstring 中明确标注返回的是日频 alpha，或添加 `annualize` 参数。

#### Bug 6: `residual_risk` 硬编码 `APPROX_BDAYS_PER_YEAR` 不支持非日频数据（严重度：中）

- **文件**: `fincore/metrics/risk.py` 第326行
- **问题**: `residual_risk` 函数使用 `np.std(residuals, ddof=1) * np.sqrt(APPROX_BDAYS_PER_YEAR)` 年化残差风险，但不接受 `period` 参数。当输入为周频或月频数据时，年化结果错误。
- **影响**: 使用非日频数据时，残差风险被错误地按日频年化。
- **建议修复**: 添加 `period=DAILY` 和 `annualization=None` 参数，使用 `annualization_factor()` 获取正确的年化因子。

#### Bug 7: `rolling_volatility` 和 `rolling_sharpe` 不接受 `period` 参数，硬编码假设日频数据（严重度：低-中）

- **文件**: `fincore/metrics/rolling.py` 第409-446行
- **问题**: 两个函数都使用 `np.sqrt(APPROX_BDAYS_PER_YEAR)` 进行年化，但不提供 `period` 参数让用户指定数据频率。
- **影响**: 对周频或月频数据使用这些函数会得到错误的年化波动率和夏普比率。
- **建议修复**: 添加 `period=DAILY, annualization=None` 参数。

#### Bug 8: `aligned_series` 对齐后仍保留 NaN 行（从0012延续，未修复）（严重度：低-中）

- **文件**: `fincore/metrics/basic.py` 第226-230行
- **问题**: 两个 Series 索引不完全匹配时，`pd.concat([head, tail[0]], axis=1)` 执行 outer join，不匹配的行填充 NaN。后续依赖函数（如 `r_cubed`、`tracking_difference` 等）可能因此获得意外的 NaN 值。
- **建议修复**: 添加 `.dropna()` 或改用 `head.align(tail[0], join='inner')`：
  ```python
  combined = pd.concat([head, tail[0]], axis=1).dropna()
  return tuple(combined.iloc[:, i] for i in range(2))
  ```

#### Bug 9: `Pyfolio.create_full_tear_sheet` 缺少 `@customize` 装饰器（严重度：低）

- **文件**: `fincore/pyfolio.py` 第268行
- **问题**: 所有其他 tear sheet 方法（`create_simple_tear_sheet`、`create_returns_tear_sheet` 等）都使用了 `@customize` 装饰器，但 `create_full_tear_sheet` 没有。
- **影响**: `create_full_tear_sheet` 不会应用 `customize` 装饰器提供的通用功能（如 context manager、错误处理等），与其他 tear sheet 方法行为不一致。

#### Bug 10: `get_max_drawdown_underwater` 对全正收益序列返回误导性结果（从0012延续，未修复）（严重度：低）

- **文件**: `fincore/metrics/drawdown.py` 第262-271行
- **问题**: 当所有收益为正时，underwater 全为0，函数返回 `(first_date, first_date, first_date)` 而非表示"无回撤"的结果。
- **建议修复**: 在 `underwater.min() == 0` 时返回 `(pd.NaT, pd.NaT, pd.NaT)`。

---

### 二、代码优化分析

#### 优化 1: `Empyrical` 类中多个方法通过类方法间接调用，增加不必要的分发开销

- **文件**: `fincore/empyrical.py`
- **涉及方法**:
  - `annual_active_risk`（第912行）调用 `Empyrical.tracking_error(...)` 而非直接调用 `_risk.tracking_error(...)`
  - `regression_annual_return`（第939-946行）调用 `Empyrical.alpha()`、`Empyrical.beta()`、`Empyrical.annual_return()` 而非直接调用模块函数
  - `annualized_cumulative_return`（第952行）调用 `Empyrical.annual_return(...)` 而非 `_yearly.annual_return(...)`
- **影响**: 每次间接调用都经过 `_dual_method.__get__` 创建新的 wrapper 函数，再调用实际函数，增加了约 2-3 微秒/次的开销。在批量计算场景下累积效果显著。
- **建议**: 统一改为直接调用对应的模块函数。

#### 优化 2: `_dual_method` 描述器每次属性访问都创建新的闭包对象

- **文件**: `fincore/empyrical.py` 第46-56行
- **问题**: `__get__` 方法在每次被访问时都创建一个新的 `wrapper` 函数。这意味着 `emp.method is emp.method` 返回 `False`，并且每次方法调用都有额外的对象创建开销。
- **影响**: 对于频繁调用的方法（如在循环中调用），性能开销可测量。也破坏了基于 `is` 的方法比较。
- **建议**: 使用 `functools.lru_cache` 缓存绑定方法，或在实例上缓存绑定的方法对象：
  ```python
  def __get__(self, obj, objtype=None):
      if obj is None:
          # Class-level access: always create wrapper (infrequent)
          @functools.wraps(self.func)
          def wrapper(*args, **kwargs):
              return self.func(objtype, *args, **kwargs)
          return wrapper
      else:
          # Instance-level access: cache on instance
          attr_name = f'_bound_{self.__name__}'
          try:
              return obj.__dict__[attr_name]
          except KeyError:
              @functools.wraps(self.func)
              def wrapper(*args, **kwargs):
                  return self.func(obj, *args, **kwargs)
              obj.__dict__[attr_name] = wrapper
              return wrapper
  ```

#### 优化 3: `perf_stats` 中 `std(returns)` 被多个指标重复计算

- **文件**: `fincore/metrics/perf_stats.py` 第64-91行
- **问题**: `perf_stats` 已经优化了 `max_drawdown` 和 `annual_return` 的预计算，但 `sharpe_ratio`、`annual_volatility`、`sortino_ratio` 各自内部都独立计算 `nanstd(returns)`。三个函数共享相同的标准差计算。
- **建议**: 预计算 `std_ret = nanstd(returns, ddof=1)` 和 `mean_ret = nanmean(returns)`，然后：
  ```python
  ann_vol = std_ret * np.sqrt(ann_factor)
  sharpe = (mean_ret / std_ret) * np.sqrt(ann_factor)
  ```

#### 优化 4: `get_top_drawdowns` 使用 O(n×top) 的串行删除策略

- **文件**: `fincore/metrics/drawdown.py` 第274-311行
- **问题**: `get_top_drawdowns` 循环 `top` 次，每次调用 `get_max_drawdown_underwater` 扫描整个 underwater 序列，然后删除已找到的回撤段，再重新扫描。总时间复杂度为 O(n × top)。
- **建议**: 使用单遍扫描识别所有回撤周期（类似 `get_all_drawdowns_detailed` 的逻辑），然后按严重度排序取前 top 个。时间复杂度降为 O(n + top × log(top))。

#### 优化 5: `up_down_capture` 和相关捕获率函数重复对齐和过滤

- **文件**: `fincore/metrics/ratios.py` 第982-1023行
- **问题**: `up_down_capture` 调用 `aligned_series` 后分别过滤上行和下行，然后调用 `_capture_aligned` 两次。但 `up_capture` 和 `down_capture` 各自也独立调用 `aligned_series`。如果用户同时需要三个捕获率指标，`aligned_series` 会被调用三次。
- **建议**: 提供一个内部函数 `_compute_all_captures(returns, factor_returns, period)` 一次性计算三个指标，避免重复对齐。

#### 优化 6: `Pyfolio` 多个绘图方法的 `self` 传递不一致

- **文件**: `fincore/pyfolio.py`
- **问题**: 大部分绘图方法传递 `self` 作为第一个参数（如 `_plot_monthly_returns_heatmap(self, returns, ...)`），但以下方法不传递 `self`：
  - `plot_long_short_holdings`（第491行）
  - `plot_exposures`（第590行）
  - `plot_returns`（第535行）
  - `plot_sector_allocations`（第618行）
  - `plot_txn_time_hist`（第685行）
  - `plot_style_factor_exposures`（第207行）
  - 所有 `plot_sector_exposures_*`、`plot_cap_exposures_*`、`plot_volume_exposures_*` 方法
- **影响**: 如果底层 tearsheet 函数签名变更（如添加 `emp` 参数），这些方法将立即出错。同时增加了代码维护的心智负担。
- **建议**: 逐一检查底层函数签名，统一传递模式。

#### 优化 7: `drawdown.py` 中 `second_*` 和 `third_*` 系列函数各自独立调用 `get_all_drawdowns_detailed`

- **文件**: `fincore/metrics/drawdown.py` 第617-713行
- **问题**: `second_max_drawdown_days`、`second_max_drawdown_recovery_days`、`third_max_drawdown_days`、`third_max_drawdown_recovery_days` 各自调用 `get_all_drawdowns_detailed(returns)` 然后排序。如果用户同时调用多个函数，完整的回撤分析（累积收益、滚动最大值、回撤识别）被重复执行4次。
- **建议**: 提供 `nth_drawdown_stats(returns, n)` 统一接口，返回第n大回撤的所有统计信息（value、duration、recovery_duration）。或在 Empyrical 类中提供批量计算接口。

---

### 三、性能优化分析

#### 性能问题 1: `roll_alpha` 在每个窗口内重复执行序列对齐（严重度：高）

- **文件**: `fincore/metrics/rolling.py` 第66-90行
- **问题**: `roll_alpha` 在函数开头已通过 `aligned_series(returns, factor_returns)` 对齐了数据，但在 for 循环内部调用 `alpha(returns_aligned.iloc[i:i+window], factor_aligned.iloc[i:i+window])` 时，`alpha()` 内部又调用 `aligned_series()` 进行二次对齐。对于 5000 个数据点、window=252，这意味着约 4748 次冗余的对齐操作。
- **预估性能损失**: 每次冗余对齐约 30-50 微秒，总计约 0.15-0.24 秒的纯浪费。
- **建议修复**: 在循环中直接调用 `alpha_aligned()` 而非 `alpha()`，绕过对齐步骤：
  ```python
  from fincore.metrics.alpha_beta import alpha_aligned
  for i in range(n):
      out[i] = alpha_aligned(
          np.asanyarray(returns_aligned.iloc[i:i+window]),
          np.asanyarray(factor_aligned.iloc[i:i+window]),
          risk_free, period, annualization)
  ```
  `roll_alpha_beta` 同理应调用 `alpha_beta_aligned()`。

#### 性能问题 2: `gen_drawdown_table` 调用 `get_top_drawdowns` 的 O(n×top) 串行扫描（严重度：中）

- **文件**: `fincore/metrics/drawdown.py` 第314-366行
- **问题**: `gen_drawdown_table` → `get_top_drawdowns` → 循环调用 `get_max_drawdown_underwater` → 每次扫描完整 underwater 序列并删除已发现段。对于 top=10、n=5000 的序列，需要 10 次完整扫描。
- **建议**: 用 `get_all_drawdowns_detailed` 的单遍扫描逻辑替代，按严重度排序后取前 top 个，将时间复杂度从 O(n×top) 降到 O(n)。

#### 性能问题 3: `roll_sharpe_ratio` 仍使用 Python for 循环（从0012延续）（严重度：中）

- **文件**: `fincore/metrics/rolling.py` 第200-252行
- **问题**: `roll_sharpe_ratio` 使用 for 循环逐窗口计算，而同文件中 `rolling_sharpe`（第427-446行）使用向量化的 `pd.rolling` 实现。两者功能相同但性能差距约 100 倍。
- **建议**: `roll_sharpe_ratio` 应复用 `rolling_sharpe` 的向量化实现，仅在需要自定义 risk_free 或 annualization 时才使用额外逻辑。向量化版本：
  ```python
  ret_adj = returns - risk_free
  rolling_mean = ret_adj.rolling(window).mean()
  rolling_std = ret_adj.rolling(window).std(ddof=1)
  result = (rolling_mean / rolling_std) * sqrt_ann
  return result.iloc[window - 1:]
  ```

#### 性能问题 4: `consecutive.py` 中每个单独函数独立 resample（从0012延续）（严重度：低-中）

- **文件**: `fincore/metrics/consecutive.py`
- **问题**: `max_consecutive_up_weeks` 和 `max_consecutive_down_weeks` 各自独立 resample；月频同理。已有 `consecutive_stats()` 做了合并优化，但 Empyrical 类未暴露该批量方法。
- **建议**: 在 Empyrical 类中添加 `consecutive_stats()` 方法，在需要多个连续指标时推荐使用。

#### 性能问题 5: `perf_stats_bootstrap` 的 1000 次 bootstrap 循环可部分向量化（严重度：低-中）

- **文件**: `fincore/metrics/perf_stats.py` 第218-227行
- **问题**: `calc_bootstrap` 使用 Python for 循环执行 1000 次采样，每次调用完整的统计函数。对于简单统计量（如 `annual_return`、`annual_volatility`），可以批量计算。
- **建议**: 一次性生成 `(n_samples, len(returns))` 的随机索引矩阵，对可向量化的统计量（mean、std、max_drawdown）使用矩阵运算替代循环。对于复杂统计量，可用 `joblib.Parallel` 或 `concurrent.futures` 并行化。

#### 性能问题 6: `hurst_exponent` 外层 lag 循环可采样优化（严重度：低）

- **文件**: `fincore/metrics/stats.py` 第137-151行
- **问题**: `for lag in range(min_lag, max_lag + 1)` 逐 lag 计算 R/S 值。对于长序列（如 n=10000），`max_lag` 可达 3333，循环次数较多。
- **建议**: 使用对数等间距采样 lag 值（如 `np.unique(np.logspace(np.log10(min_lag), np.log10(max_lag), 50).astype(int))`），将循环次数控制在 50 次以内，对回归精度影响极小。

---

### 四、总结

| 类别 | 数量 | 优先级建议 |
|------|------|-----------|
| Bug | 10个 | Bug 1（DataFrame崩溃）和 Bug 8（aligned_series NaN）应优先修复 |
| 代码优化 | 7个 | 优化 1（间接调用开销）和优化 6（Pyfolio self传递）影响最大 |
| 性能优化 | 6个 | 性能问题 1（roll_alpha重复对齐）和性能问题 3（roll_sharpe向量化）收益最高 |

---

### 五、实际修复记录

> 修复日期: 2026-02-09
> 修复后测试: 1233个测试全部通过，102个warnings

#### 已修复项目

| # | 问题 | 修改文件 | 修改内容 |
|---|------|---------|---------|
| Bug 1 | `annual_return` DataFrame崩溃 | `fincore/metrics/yearly.py` | 对 `ending_value` 为 Series/ndarray 时使用元素级掩码处理 |
| Bug 2 | `calmar_ratio` 不处理 `ending_value <= 0` | `fincore/metrics/ratios.py` | 改为复用 `annual_return()` 而非重新计算年化收益率 |
| Bug 3 | `perf_attrib` 不使用实例数据 | `fincore/empyrical.py` | 添加 `positions`/`factor_returns`/`factor_loadings` 的实例回退 |
| Bug 4 | `r_cubed` 不处理 NaN | `fincore/metrics/stats.py` | 在 `np.corrcoef` 前显式过滤 NaN 值 |
| Bug 5 | `rolling_regression` alpha 不一致 | `fincore/metrics/rolling.py` | 在 docstring 中明确标注返回非年化 alpha |
| Bug 6 | `residual_risk` 硬编码年化因子 | `fincore/metrics/risk.py` | 添加 `period` 和 `annualization` 参数 |
| Bug 7 | `rolling_volatility`/`rolling_sharpe` 硬编码 | `fincore/metrics/rolling.py` | 添加 `period` 和 `annualization` 参数 |
| Bug 9 | `create_full_tear_sheet` 缺 `@customize` | `fincore/pyfolio.py` | 添加 `@customize` 装饰器 |
| Perf 1 | `roll_alpha` 循环内重复对齐 | `fincore/metrics/rolling.py` | 改用 `alpha_aligned`/`alpha_beta_aligned` + numpy 数组切片 |
| Perf 3 | `roll_sharpe_ratio` Python循环 | `fincore/metrics/rolling.py` | 向量化为 `pd.rolling().mean()`/`.std()` 实现 |
| Opt 1 | Empyrical 间接方法调用 | `fincore/empyrical.py` | `annual_active_risk`/`regression_annual_return`/`annualized_cumulative_return` 改为直接调用模块函数 |
| Opt 2 | `_dual_method` 闭包开销 | `fincore/empyrical.py` | 在实例 `__dict__` 上缓存绑定方法 |
| Opt 3 | `perf_stats` 重复计算 std | `fincore/metrics/perf_stats.py` | 预计算 `nanstd`/`nanmean`，内联 `annual_volatility` 和 `sharpe_ratio` |

#### 未修复项目（需更大范围重构）

| # | 问题 | 原因 |
|---|------|------|
| Bug 8 | `aligned_series` 对齐后保留 NaN | 改为 inner join 会破坏混合频率（周频vs日频）的对齐行为，`annual_active_return` 等函数依赖 outer join 语义。需在各依赖函数内部单独处理 NaN |
| Bug 10 | `get_max_drawdown_underwater` 全正收益 | 返回 `(NaT, NaT, NaT)` 会破坏 `get_top_drawdowns` 和 `gen_drawdown_table` 的下游逻辑，需同步更新所有下游代码和测试 |
