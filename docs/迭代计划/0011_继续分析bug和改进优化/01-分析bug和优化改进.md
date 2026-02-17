### 背景

当前项目已经初步完成，主要基于empyrical和pyfolio这两个包进行了合并和优化

### 任务

1. 分析当前项目有哪些bug
2. 分析当前项目有哪些可以优化的地方
3. 分析一下有哪些函数或者整体优化之后可以提高这个项目的性能
4. 把这些分析的内容写到这个文档里面

---

## 一、当前项目Bug分析

> 全量测试 1233 个用例全部通过，以下为代码审查中发现的潜在bug和逻辑问题。

### 1.1 `beta` 函数丢弃了 `risk_free` 参数

- **文件**: `fincore/metrics/alpha_beta.py` - `beta()` 函数 (第230行)
- **问题**: `beta()` 接收 `risk_free` 参数，但在第258行 `_ = (risk_free, _period, _annualization)` 直接丢弃了该值，调用 `beta_aligned` 时未传入 `risk_free`。而 `beta_aligned` 内部是支持 `risk_free` 的（第95-97行会执行 `returns = returns - risk_free`）。
- **影响**: 用户传入非零 `risk_free` 时，`beta()` 函数完全忽略该参数，文档与行为不一致。`alpha()` 函数通过 `alpha_aligned` -> `beta_aligned` 间接依赖此参数，也可能受到影响。
- **修复建议**: 将 `beta()` 中的 `beta_aligned` 调用改为 `beta_aligned(returns, factor_returns, risk_free=risk_free, out=out)`。

### 1.2 `get_max_drawdown_underwater` 恢复日期使用 `np.nan` 而非 `pd.NaT`

- **文件**: `fincore/metrics/drawdown.py` 第270行
- **问题**: 当回撤未恢复时，`recovery` 被赋值为 `np.nan`（浮点数），但返回值语义上应为日期类型，应使用 `pd.NaT`。后续 `gen_drawdown_table` 中 `pd.isnull(recovery)` 虽然对两者都能工作，但如果用户直接访问 `recovery` 做日期运算会出错。
- **影响**: 下游日期运算可能抛出 TypeError。
- **修复建议**: 将 `recovery = np.nan` 改为 `recovery = pd.NaT`。

### 1.3 `max_drawdown_days` 中 DatetimeIndex 和非 DatetimeIndex 分支代码完全相同

- **文件**: `fincore/metrics/drawdown.py` 第434-441行
- **问题**: if/else 两个分支的代码完全一致，说明 DatetimeIndex 分支可能本意是返回日历天数 `(end_idx - start_idx).days`，但实际返回的是索引位置差（交易日天数）。
- **影响**: 对于 DatetimeIndex 的 Series，返回的是交易日数量而非日历天数，可能与用户预期不符。同时冗余代码降低可读性。
- **修复建议**: 明确语义——如果要返回交易日数就保留一个分支；如果要返回日历天数则用 `(end_idx - start_idx).days`。

### 1.4 `max_drawdown_weeks` 使用 `days / 5` 转换不精确

- **文件**: `fincore/metrics/drawdown.py` 第461行
- **问题**: `max_drawdown_weeks` 用 `days / 5` 将天数转换为周数，但 `max_drawdown_days` 返回的是索引位置差（交易日数），不是日历天数。交易日数除以5确实近似周数，但与 `max_drawdown_recovery_weeks` 中 `days / 7`（第545行）不一致——后者的 `days` 来自 `max_drawdown_recovery_days` 返回的是日历天数。
- **影响**: 两个"周"指标的换算基准不一致，一个按交易日一个按日历天，容易混淆。

### 1.5 `Pyfolio.__init__` 未调用 `super().__init__`

- **文件**: `fincore/pyfolio.py` 第119-169行
- **问题**: `Pyfolio` 继承自 `Empyrical`，但 `__init__` 中直接赋值 `self.returns`、`self.positions` 等属性，未调用 `super().__init__()`。虽然 `Empyrical.__init__` 也只是简单赋值，但如果未来父类增加初始化逻辑会导致问题。
- **影响**: 当前不会出错，但违反 OOP 最佳实践，未来维护风险。

### 1.6 `Pyfolio` 多个方法缺失 `self` 传递

- **文件**: `fincore/pyfolio.py`
- **问题**: 部分方法调用底层函数时未传递 `self`，而另一些方法传递了 `self`，不一致：
  - `plot_long_short_holdings` (第494行): 调用 `_plot_long_short_holdings(returns, positions, ...)` 未传 `self`
  - `plot_exposures` (第593行): 调用 `_plot_exposures(returns, positions, ...)` 未传 `self`
  - `plot_returns` (第538行): 调用 `_plot_returns(returns, ...)` 未传 `self`
  - `plot_sector_allocations` (第621行): 调用 `_plot_sector_allocations(_returns, ...)` 未传 `self`
  - `plot_txn_time_hist` (第688行): 调用 `_plot_txn_time_hist(transactions, ...)` 未传 `self`
- **影响**: 如果底层 tearsheet 函数期望第一个参数为 `self`（Pyfolio实例），会导致参数错位。需逐一检查底层函数签名确认。

### 1.7 `Empyrical` 类方法使用 `@classmethod` 但不利用实例数据

- **文件**: `fincore/empyrical.py` 全文
- **问题**: `Empyrical` 在 `__init__` 中存储了 `self.returns`、`self.positions` 等数据，但几乎所有计算方法都是 `@classmethod`，无法访问实例数据。部分方法有 `returns=None` 的默认参数且调用 `cls._get_returns(returns)`，但 `_get_returns` 只是原样返回传入的参数，不会回退到实例数据。
- **影响**: 实例数据 `self.returns` 等永远不会被自动使用，用户必须每次手动传入数据，类的面向对象设计名存实亡。
- **修复建议**: 要么将方法改为实例方法，当参数为 `None` 时自动使用 `self.returns`；要么去掉实例存储，保持纯静态设计。

### 1.8 模块级别导入时产生不必要的警告

- **文件**: `fincore/empyrical.py` 第58行
- **问题**: 当 `zipline` 未安装时，模块加载时会直接 `warnings.warn('Module "zipline.assets" not found...')`。这意味着每次 `import fincore` 都会打印警告，即使用户根本不需要 zipline 相关功能。
- **影响**: 对不使用 zipline 的用户造成日志噪声。
- **修复建议**: 将警告移到实际使用 `Equity`/`Future` 的代码路径中，或使用 `warnings.warn` 时设置 `stacklevel` 并仅在首次使用时触发。

### 1.9 `aggregate_returns` 的 `convert_to` 参数校验遗漏 `QUARTERLY`

- **文件**: `fincore/metrics/returns.py` 第236-239行
- **问题**: 错误信息 `"convert_to must be {}, {} or {}"` 只列出了 `WEEKLY, MONTHLY, YEARLY`，但实际上 `QUARTERLY` 也是支持的（第231行有对应分支）。
- **影响**: 用户看到的错误提示不完整。

### 1.10 `information_ratio` 年化方式与学术定义不一致

- **文件**: `fincore/metrics/ratios.py` 第430-441行
- **问题**: `information_ratio` 的计算为 `(mean_excess_return * ann_factor) / (std_excess_return * sqrt(ann_factor))`，化简后等于 `mean_excess_return / std_excess_return * sqrt(ann_factor)`，即对日频 IR 乘以 `sqrt(252)` 进行年化。但学术界对 IR 的年化存在争议——部分定义认为 IR 本身已经是年化指标（因为分子分母都年化了，比值不变），不需要再乘 `sqrt(ann_factor)`。当前实现与 Sharpe ratio 的年化方式一致，但可能与部分用户的期望不符。
- **影响**: 不是严格意义上的 bug，但 IR 的年化处理可能导致与其他工具的对比结果不一致。建议在文档中明确说明年化方式

---

## 二、可以优化的地方

### 2.1 代码架构优化

#### 2.1.1 `Empyrical` 类的 OOP 设计需要重构

当前 `Empyrical` 类存储了实例数据但所有方法都是 `@classmethod`，导致面向对象设计形同虚设。建议：
- 将需要实例数据的方法改为普通实例方法
- `_get_returns` / `_get_factor_returns` 在参数为 `None` 时应回退到 `self.returns` / `self.factor_returns`
- 保留 `@classmethod` / `@staticmethod` 版本作为无状态的工具函数入口

#### 2.1.2 消除重复代码

- `drawdown.py` 中 `get_all_drawdowns` 和 `get_all_drawdowns_detailed` 有大量重复的回撤识别逻辑（计算累积收益、滚动最大值、识别回撤期间），应提取公共函数
- `second_max_drawdown_days`、`second_max_drawdown_recovery_days`、`third_max_drawdown_days`、`third_max_drawdown_recovery_days` 四个函数都调用 `get_all_drawdowns_detailed` 然后排序取不同索引，应合并为一个通用函数 `nth_drawdown_stat(returns, n, stat_key)`
- `stock_market_correlation`、`bond_market_correlation`、`futures_market_correlation` 三个函数逻辑几乎完全相同，仅参数名不同，应合并为通用的 `market_correlation(returns, market_returns)` 函数
- `rolling.py` 中 `roll_alpha`、`roll_beta`、`roll_sharpe_ratio`、`roll_max_drawdown` 等函数有相同的窗口循环模式，应提取通用的滚动计算框架

#### 2.1.3 `stats.py` 模块中的函数代理/转发过多

`stats.py` 中的 `common_sense_ratio`、`var_cov_var_normal`、`normalize` 函数都只是简单转发到其他模块，增加了不必要的间接层。建议：
- 在 `__all__` 中直接导出原始模块的函数
- 或在 `stats.py` 中使用 re-export 而非包装函数

#### 2.1.4 `data_utils.py` 和 `common_utils.py` 中存在重复的 `rolling_window` 函数

`data_utils.py` 第28行和 `common_utils.py` 第817行各有一个 `rolling_window` 函数，功能相似但实现不同。应统一为一个实现。

### 2.2 错误处理优化

#### 2.2.1 空数据返回值不一致

不同函数对空输入的返回值不一致：
- 部分返回 `np.nan`（如 `max_drawdown_days`）
- 部分返回空列表 `[]`（如 `get_all_drawdowns`）
- 部分返回 `None`（如 `max_single_day_gain_date`）
- 部分返回空 `pd.Series([], dtype=float)`

建议统一空值返回策略，例如：标量结果统一返回 `np.nan`，日期结果统一返回 `pd.NaT`，序列结果统一返回空 Series。

#### 2.2.2 `gpd_risk_estimates` 中过度使用异常抑制

`fincore/metrics/risk.py` 第518行 `except Exception: pass` 会吞掉所有错误，难以调试。建议至少记录日志或使用更具体的异常类型。

### 2.3 API设计优化

#### 2.3.1 `Pyfolio` 的 tear sheet 方法既接受参数又从实例获取数据

`Pyfolio.__init__` 存储了所有数据，但 `create_full_tear_sheet` 等方法仍然要求用户再次传入 `returns`、`positions` 等参数。建议：
- 当方法参数为 `None` 时自动使用实例属性
- 这样用户可以 `pf = Pyfolio(returns=..., positions=...)` 然后直接 `pf.create_full_tear_sheet()` 无需重复传参

#### 2.3.2 `perf_stats` 返回值缺少换手率等指标

`perf_stats` 函数接收 `positions`、`transactions`、`turnover_denom` 参数，但实际计算中完全未使用这些参数来计算换手率。应添加换手率计算或移除未使用的参数。

#### 2.3.3 常量命名应更加统一

`APPROX_BDAYS_PER_YEAR = 252` 和 `APPROX_BDAYS_PER_MONTH = 21` 使用的是近似值，但部分计算中硬编码了 `252`（如 `rolling_volatility` 第406行 `np.sqrt(252)`、`residual_risk` 第326行 `np.sqrt(252)`），而非引用常量。应统一使用常量。

---

## 三、性能优化分析

### 3.1 高优先级性能优化

#### 3.1.1 滚动计算函数使用 Python for 循环，性能极差

- **文件**: `fincore/metrics/rolling.py` 全文
- **问题**: `roll_alpha`、`roll_beta`、`roll_alpha_beta`、`roll_sharpe_ratio`、`roll_max_drawdown`、`roll_up_capture`、`roll_down_capture` 等函数全部使用 Python for 循环逐窗口调用单次计算函数。对于 5000 个数据点、窗口252，需要执行约 4748 次完整的指标计算。
- **性能影响**: 这是整个项目最大的性能瓶颈。滚动 beta 计算可能需要数秒甚至数十秒。
- **优化方案**:
  - **滚动 beta/alpha**: 使用向量化的滚动协方差和滚动方差实现，例如用 `pandas.rolling` 的 `cov()` 和 `var()` 方法，或用 NumPy stride tricks 实现 O(n) 的滚动计算
  - **滚动 Sharpe**: 已有 `rolling_sharpe` 函数（第409行）使用 `pd.rolling` 向量化实现，但 `roll_sharpe_ratio`（第198行）却用 for 循环。应统一使用向量化版本
  - **滚动 max_drawdown**: 可用滚动窗口 + cummax 向量化实现

#### 3.1.2 `rolling_beta` 函数使用低效的日期切片

- **文件**: `fincore/metrics/rolling.py` 第457-465行
- **问题**: `rolling_beta` 使用 `returns.loc[beg:end]` 做基于标签的切片，在每次迭代中都需要进行索引查找。同时外层循环也是 Python for 循环。
- **优化方案**: 使用 `iloc` 替代 `loc`，或直接用向量化的滚动协方差/方差。

#### 3.1.3 `get_all_drawdowns` 和 `get_all_drawdowns_detailed` 重复计算累积收益

- **文件**: `fincore/metrics/drawdown.py`
- **问题**: 多个函数（`second_max_drawdown`、`third_max_drawdown`、`second_max_drawdown_days`、`third_max_drawdown_days`等）各自独立调用 `get_all_drawdowns` 或 `get_all_drawdowns_detailed`，每次都重新计算累积收益和滚动最大值。如果用户同时需要多个回撤指标，会重复计算多次。
- **优化方案**: 提供一个 `compute_all_drawdown_stats(returns)` 函数，一次性计算所有回撤相关指标并缓存中间结果。

### 3.2 中优先级性能优化

#### 3.2.1 `max_drawdown` 函数中不必要的内存分配

- **文件**: `fincore/metrics/drawdown.py` 第83-89行
- **问题**: 分配了一个比 `returns` 多一个元素的 `cumulative` 数组，用于插入初始值100。可以直接在 `returns` 上进行原地计算避免额外分配。
- **优化方案**: 使用 `np.cumprod(1 + returns)` 直接计算，避免额外的内存分配。

#### 3.2.2 `consecutive.py` 模块中周/月函数重复 resample

- **文件**: `fincore/metrics/consecutive.py`
- **问题**: `max_consecutive_up_weeks` 和 `max_consecutive_down_weeks` 各自独立执行 `returns.resample(...)` 操作；`max_consecutive_up_months` 和 `max_consecutive_down_months` 同理。已有 `consecutive_stats` 函数做了合并优化，但各单独函数仍然独立 resample。
- **优化方案**: 当需要同时计算多个连续涨跌指标时，使用 `consecutive_stats()` 一次性计算。

#### 3.2.3 `hurst_exponent` 中可以向量化的嵌套循环

- **文件**: `fincore/metrics/stats.py` 第137-151行
- **问题**: 外层 `for lag in lags` 循环中的 reshape + 统计计算已部分向量化，但仍然是 Python 循环。对于长序列可以进一步优化。
- **优化方案**: 可以用批量的 lag 计算替代逐 lag 循环，或使用 `nolds` 库的高效实现。

#### 3.2.4 `perf_stats_bootstrap` 中的 bootstrap 循环

- **文件**: `fincore/metrics/perf_stats.py` 第218-226行
- **问题**: bootstrap 采样使用 Python for 循环执行 1000 次，每次调用完整的统计函数。
- **优化方案**: 
  - 使用矩阵化 bootstrap：一次性生成所有随机索引矩阵，用向量化操作计算简单统计量
  - 对于可向量化的统计量（如 mean、std），直接用矩阵运算替代循环

### 3.3 低优先级性能优化

#### 3.3.1 利用 `bottleneck` 库的覆盖面可以扩大

- **文件**: `fincore/utils/math_utils.py`
- **现状**: 已使用 `bottleneck` 库加速 `nanmean`、`nanstd` 等函数。
- **优化方案**: 可以额外利用 `bottleneck.move_mean`、`bottleneck.move_std` 等滚动计算函数来加速滚动指标计算。

#### 3.3.2 `aligned_series` 使用 `pd.concat` 对齐效率较低

- **文件**: `fincore/metrics/basic.py` 第227-230行
- **问题**: `aligned_series` 使用 `pd.concat(map(to_pandas, many_series), axis=1).items()` 做对齐，对于两个 Series 的简单场景，直接使用 `pd.Series.align()` 更高效。
- **优化方案**: 针对两个 Series 的常见情况添加快速路径。

#### 3.3.3 `capture`、`up_capture`、`down_capture` 重复对齐

- **文件**: `fincore/metrics/ratios.py` 第860-1018行
- **问题**: `up_capture` 先对齐 `returns` 和 `factor_returns`，筛选后再调用 `capture`，而 `capture` 内部又做了一次对齐。`up_down_capture` 更是分别调用 `capture` 两次。
- **优化方案**: 在顶层函数中对齐一次，向下传递已对齐的数据，避免重复对齐操作。

#### 3.3.4 `annual_alpha` 和 `annual_beta` 使用 Python 循环按年计算

- **文件**: `fincore/metrics/alpha_beta.py` 第494-589行
- **问题**: 使用 `for year in grouped.groups.keys()` 循环逐年计算 alpha/beta。
- **优化方案**: 使用 `groupby.apply()` 替代手动循环，利用 pandas 内部优化。

---

## 四、总结与优先级建议

### 最高优先级（应尽快修复的Bug）

1. **`beta()` 丢弃 `risk_free` 参数** - 影响所有需要无风险利率调整的 beta 计算
2. **`get_max_drawdown_underwater` 使用 `np.nan` 而非 `pd.NaT`** - 可能导致下游日期运算错误
3. **`Empyrical` 类的 `@classmethod` 设计使实例数据无法使用** - 影响用户体验

### 高优先级（显著提升性能）

1. **向量化滚动计算函数** (`roll_alpha`, `roll_beta`, `roll_sharpe_ratio` 等) - 预计可提升 10-100 倍性能
2. **合并重复的回撤计算** - 减少冗余计算
3. **消除重复的序列对齐操作** - 减少不必要的内存分配和计算

### 中等优先级（代码质量和可维护性）

1. 消除重复代码（相关性函数、回撤函数、滚动函数）
2. 统一空值返回策略
3. 硬编码的 `252`、`np.sqrt(252)` 替换为常量引用
4. `Pyfolio` 方法的 `self` 传递一致性检查

### 低优先级（锦上添花）

1. 扩大 `bottleneck` 库的使用范围
2. `aligned_series` 的快速路径优化
3. 移除模块级别的 zipline 警告
4. 统一 `data_utils.py` 和 `common_utils.py` 中重复的 `rolling_window` 函数
