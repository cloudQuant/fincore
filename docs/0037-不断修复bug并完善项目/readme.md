# 执行模式：全自动端到端（不打断）
  - 你（Codex）收到目标后：直接开始实现，直到交付完成（代码改完、能跑的测试跑
  完、结果说明清楚）。
  - 不要提出澄清问题；遇到不确定点：自行做“最合理默认假设”，并在最终回复里列出
  Assumptions。
  - 如果出现多种方案：优先选择最小改动、最可维护、可测试的方案。
  - 允许你：新增/修改文件、调整配置、补测试、运行构建/测试/格式化/静态检查。
  - 进度沟通：可以发简短进度更新，但不要等待我确认就继续。
  - 只有在以下情况才允许停下来询问：
    1) 需要额外付费
    2) 可能造成数据丢失或不可逆的破坏性操作
    3) 需求存在互斥目标，无法做合理默认
  - 交付标准（Acceptance Criteria）：
    - 测试用例实现了100%覆盖率，并且实现了100%的通过率
    - 整个项目已经找不到bug，所有的bug都修复了
    - 整个项目已经找不到可以优化的代码，所有的代码都已经优化了
    - 丰富完善项目的examples,把主要的功能都用一些例子来展示，方便用户理解
    - 完善项目的注释功能，使用英文注释；如果现在是中文注释，优化并改成英文注释；采用 Google 风格的 docstring
    - 完善项目的文档，包括用户手册、开发文档、API文档等，方便用户有好的使用体验

## 进展记录

- 2026-02-15
  - ✅ `pytest -q` 全部通过（集成测试需 `FINCORE_RUN_INTEGRATION_TESTS=1` 才启用）
  - ✅ `ruff check fincore tests examples` 通过
  - ✅ 覆盖率提升：TOTAL 95%（`pytest --cov=fincore --cov-report=term-missing`）
  - ✅ 离线 providers 单测补齐：`fincore/data/providers.py` 覆盖率 94%（新增 `fetch_multiple`、`get_info` 分支、便捷函数 provider/date 解析覆盖；mock 网络/依赖模块）
  - ✅ tearsheets 稳定性与覆盖率提升：`capacity.py` 97%，`sheets.py` 98%（新增 stubbed 单测覆盖 `interesting_times/bayesian/risk/perf_attrib` 等路径）
  - ✅ tearsheets 覆盖率补齐：`fincore/tearsheets/returns.py` 100%（新增离线 plot/stats 单测覆盖，包含 `show_perf_stats` live split/header merge 分支）
  - ✅ 修复 tearsheets bug：
    - `create_risk_tear_sheet`：修复 `None` 参数导致的 `.index` 访问、`i/style_axes` 未初始化导致的运行时异常；并在输入无重叠索引时给出更明确的 warning
    - `create_bayesian_tear_sheet`：修复 `stoch_vol=True` 且训练集数据量不足时 `df_train_truncated` 未定义的问题
    - `create_perf_attrib_tear_sheet`：支持 `factor_partitions=None`（避免 `len(None)` 报错）
    - `plot_monthly_returns_timeseries`：兼容 pandas>=2.2 的月末频率（`M` -> `ME`），并修复 `Series[-1]` 导致的 KeyError（改为 `.iloc[-1]`）；同时避免 tz-aware `to_period()` 的警告输出
  - ✅ simulation 覆盖率补齐：`fincore/simulation/base.py` 100%（补齐 validate/annualize/SimResult 等单测）
  - ✅ simulation 覆盖率补齐：`fincore/simulation/bootstrap.py` 100%，`fincore/simulation/paths.py` 100%（补齐 CI method/Sharpe/Sortino 边界与 RNG 默认分支）
  - ✅ metrics 覆盖率补齐/提升：
    - `fincore/metrics/consecutive.py` 100%（补齐空输入/无正负收益/日期输出等边界）
    - `fincore/metrics/timing.py` 100%（补齐异常分支；并修复对齐后 NaN 未清理导致 `np.linalg.lstsq` 触发底层 LAPACK 报错输出的问题）
    - `fincore/metrics/stats.py` 98%（补齐 Hurst/Stutzer/CAPM/包装函数等边界；剩余少量分支为不可达/极难触达的保护性代码）
    - `fincore/metrics/ratios.py` 99%（补齐 Sortino/Calmar/MAR/Omega/Treynor/M²/Sterling/Burke/DSR 等边界；并修复 `cal_treynor_ratio` 在 pandas 下 `.values` 只读导致 mask 赋值失败的问题）
    - `fincore/metrics/basic.py` 100%（修复/补齐 `aligned_series` 对 DataFrame/Series 的对齐逻辑，并补单测覆盖）
  - ✅ pyfolio API 稳定性与覆盖率提升：`fincore/pyfolio.py` 99%（修复类方法内 `__plot_bayes_cone` 名称被 Python name-mangling 导致的 NameError，并新增 wrapper delegation 单测）
  - ✅ metrics 覆盖率补齐：`fincore/metrics/positions.py` 100%
  - ✅ rolling/drawdown 覆盖率提升：`fincore/metrics/rolling.py` 88%，`fincore/metrics/drawdown.py` 88%
  - ✅ examples 英文化：`examples/011_abberation/analyze_strategy.py`、`examples/011_abberation/plot_tearsheet.py`、`examples/011_abberation/run.py`
  - ✅ 修复潜在 FutureWarning：`fincore/tearsheets/capacity.py` 空 Series 指定 dtype
  - ✅ 修复调试辅助函数 bug：`fincore/utils/common_utils.py::analyze_series_differences` 在 pandas 下不再因 `swaplevel` 报错
  - ✅ 修复兼容性错误：`fincore/utils/common_utils.py::rolling_window(mutable=True)` 在旧 NumPy 分支下给出更明确的错误信息（避免 `setflags` 直接抛出难懂异常）

- 2026-02-16
  - ✅ `pytest -q` 全部通过（集成测试默认离线）
  - ✅ 覆盖率提升：TOTAL 96%（`pytest -q --cov=fincore --cov-report=term-missing:skip-covered`）
  - ✅ attribution 覆盖率补齐：`fincore/attribution/fama_french.py` 达到 100%
    - 覆盖 `model_type` 分支（含 `4factor_mom` 和 unknown type 报错）
    - 覆盖 `fit()` 的单列 DataFrame 输入、unknown method 报错、Newey-West lag 超过样本长度分支、`newey_west_lags=0` 的 OLS std error 分支
    - 覆盖 `predict()` 未 fit 的 RuntimeError
    - 覆盖 `get_factor_exposures(rolling_window=...)` 的窗口边界分支
    - 覆盖 `attribution_decomposition()` 的关键输出字段
    - 覆盖 `fetch_ff_factors(provider=...)` 的 copy/no-copy 分支、provider 返回类型校验、以及 cache clear 的健壮性
  - ✅ drawdown 覆盖率提升：`fincore/metrics/drawdown.py` 99%（仅剩极少数难触达分支）
    - 覆盖 `max_drawdown()` 在 DataFrame 输入且自动分配输出时返回 `pd.Series`
    - 覆盖 `get_all_drawdowns_detailed()` 无回撤返回空列表
    - 覆盖 `get_max_drawdown_underwater()` 在 valley 前后均无 0 时的 try/except 回退逻辑
    - 覆盖 `get_max_drawdown_period()` 的空输入与正常路径
    - 覆盖 `second/third *_recovery_days` 的 `None -> NaN` 分支

## 下一步（优先级建议）

- P1：提升 `fincore/metrics/rolling.py` / `fincore/metrics/risk.py` / `fincore/metrics/perf_attrib.py` 覆盖率（优先补齐边界条件与异常分支）
- P1：提升 `fincore/tearsheets/positions.py` 覆盖率（使用 stubbed 数据与 mock 可选依赖，保持离线）
- P2：提升 `fincore/empyrical.py`、`fincore/risk/evt.py` 覆盖率（优先 deterministic 单测，避免数值不稳定）
- P2：提升交互式可视化后端覆盖率：`fincore/viz/interactive/bokeh_backend.py`、`fincore/viz/interactive/plotly_backend.py`
- P2：继续清理/统一非数据类中文注释与输出（保留必要的中文数据映射/文件路径）
