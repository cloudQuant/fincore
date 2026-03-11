# 代码质量改进记录

> 更新日期: 2026-03-11

本文档记录按行业最佳实践执行的代码质量优化。

## 一、已执行优化

### 1. Scripts 目录 Lint 修复

**问题**: CI 未检查 `scripts/`，存在 12 个 Ruff 违规。

**修复**:
- `scripts/simple_test.py`: 添加 `import fincore`，修复未定义 `fincore` (F821)
- `scripts/simple_test.py`, `simple_import_test.py`, `test_runner.py`: 使用 `Path(__file__).parent.parent / "tests"` 替代硬编码路径，提高可移植性
- `scripts/update_global_index_data.py`: 移除不必要的 `# -*- coding: utf-8 -*-` (UP009)，移除无占位符的 f-string (F541)
- `scripts/verify_environment.py`: 移除无占位符的 f-string
- 所有 scripts: 使用 `ruff check --fix` 自动修复 import 排序 (I001)

### 2. CI 扩展

**变更**: `.github/workflows/ci.yml`

- Ruff check: `fincore/ tests/` → `fincore/ tests/ scripts/`
- Ruff format: 同上

确保 scripts 与主代码库保持相同质量标准。

### 3. pyproject.toml 格式统一

**问题**: `[tool.ruff.lint]` 下 `ignore` 列表混用 tab 与空格缩进。

**修复**: 统一为 4 空格缩进，符合 TOML 惯例。

### 4. Pre-commit 配置

**新增**: `.pre-commit-config.yaml`

- Ruff lint + format（覆盖 fincore、tests、scripts）
- pre-commit-hooks: trailing-whitespace, end-of-file-fixer, check-yaml, check-toml, check-merge-conflict, detect-private-key

**使用**:
```bash
pip install pre-commit
pre-commit install
# 或手动运行
pre-commit run --all-files
```

### 5. 核心指标参数校验与边缘情况修复（2026-03-10）

**变更**:

- **value_at_risk / conditional_value_at_risk** (`fincore/metrics/risk.py`): 添加 `cutoff` 校验，当 `cutoff` 不在 `[0, 1]` 时返回 `np.nan`，避免非法输入产生未定义行为。
- **information_ratio** (`fincore/metrics/ratios.py`): 当追踪误差为零或结果为非有限值时显式返回 `NaN`，避免 `inf` 向下游传播。
- **excess_sharpe** (`fincore/metrics/ratios.py`): 使用 `_MIN_STD = 1e-15` 阈值替代 `np.nan_to_num`，避免将 NaN 转为 0 导致除零；当原始 tracking error 接近零时返回 `NaN`。
- **alpha_beta.py** (`fincore/metrics/alpha_beta.py`): 添加 `assert out is not None` 以帮助 mypy 类型推断，并补充 `type: ignore[union-attr]` 处理复杂控制流下的可选类型。

### 6. P2/P3 优化（2026-03-10）

**P2: alpha_ → volatility_power**

- `annual_volatility` (`fincore/metrics/risk.py`): 主参数重命名为 `volatility_power`，`alpha_` 保留为弃用别名并发出 DeprecationWarning。
- `scripts/verify_quality_fixes.py`: 更新为使用 `volatility_power`。

**P3: DURATION_STATS FIXME 处理**

- `fincore/constants/style.py`: 新增 `DURATION_STATS_GROUP`（Avg # round_trips per day/month），新增 `_duration_span_days` 辅助函数。
- `fincore/metrics/round_trips.py`: `agg_all_long_short` 支持 `group_aware_stats` 参数，在 duration 统计时传入 `(data, group)` 以计算日历跨度相关指标。

**P3: 类型注解**

- `agg_all_long_short`、`gen_round_trip_stats`: 补充参数与返回值类型注解。
- **consecutive.py** (2026-03-10): 为全部 20+ 公开函数补充参数与返回值类型（`pd.Series`、`float`、`pd.Timestamp | None` 等）。
- **positions.py** (2026-03-10): 为 12 个公开函数补充类型（`pd.DataFrame`、`pd.Series`、`tuple[...]` 等）。
- **alpha_beta.py** (2026-03-10): 为 `beta_aligned`、`beta`、`alpha`、`up_alpha_beta`、`down_alpha_beta`、`annual_alpha`、`annual_beta`、`alpha_percentile_rank`、`_conditional_alpha_beta` 补充完整类型。
- **transactions.py** (2026-03-10): 为 12 个公开函数补充类型（`pd.DataFrame`、`pd.Series`、`dict` 等）。

### 7. Mypy 与类型安全强化（2026-03-10）

**变更**:

- **fincore/constants/style.py**:
  - `_duration_span_days`: 修复 mypy `no-any-return`，显式 `float(span.total_seconds())` 确保返回类型明确。
  - 补充 `group: pd.DataFrame` 参数类型注解。
  - 添加 NaT 防护：当 `span` 为 `pd.NaT` 时返回 `0.0`，避免 `AttributeError`。
- **CI**: mypy 检查范围扩展至 `fincore/constants`。

### 8. Docstring 完善与 gpd_risk_estimates 重构（2026-03-10）

**变更**:

- **omega_ratio** (`fincore/metrics/ratios.py`): 在 Returns 中补充 `required_return <= -1` 时返回 NaN 的说明。
- **sterling_ratio / burke_ratio** (`fincore/metrics/ratios.py`): 明确说明无回撤时返回 `np.inf` 或 `NaN` 的语义，提醒下游显式处理 `np.inf`。
- **downside_risk** (`fincore/metrics/risk.py`): 在 Returns 中说明当全部收益高于阈值时返回 0。
- **gpd_risk_estimates** (`fincore/metrics/risk.py`): 将嵌套函数提取为模块级私有函数 `_gpd_loglikelihood_scale_and_shape`、`_gpd_loglikelihood_scale_only`、`_gpd_var_calculator`、`_gpd_es_calculator`、`_gpd_loglikelihood`，提高可读性与可测试性。

### 9. Ruff PIE790 与抽象方法优化（2026-03-10）

**变更**:

- **pyproject.toml**: 新增 `PIE790`（unnecessary-placeholder）规则，消除冗余 `pass` 语句。
- **fincore/data/providers.py**: 抽象方法 `fetch`、`fetch_multiple`、`get_info` 的 `pass` 替换为 `...`（抽象方法惯用写法）。
- **fincore/exceptions.py**: `FincoreError` 基类移除冗余 `pass`。
- **tests/**: `test_var.py`、`test_win_loss_rate.py`、`test_alpha_beta_missing_coverage.py` 中基类与占位测试的 `pass` 移除。

### 10. 行业最佳实践补充（2026-03-10）

**变更**:

- **VaR/CVaR docstring 对齐** (`fincore/metrics/risk.py`): 将 `cutoff` 参数说明从 "(0, 1)" 更新为 "[0, 1]"，与实现一致；补充 "Values outside [0, 1] return NaN" 说明。
- **Ruff C4 规则** (`pyproject.toml`): 新增 flake8-comprehensions（C401–C420），提升推导式与字面量写法质量；C408 暂忽略（plotly `dict()` 调用）。
- **Bandit 安全扫描**:
  - `pyproject.toml`: 新增 `bandit[toml]>=1.7` 至 dev 依赖；新增 `[tool.bandit]` 配置，跳过 B101（assert 用于类型收窄）。
  - `.github/workflows/ci.yml`: 新增 `security` job，对 `fincore/` 运行 `bandit -r fincore/ -c pyproject.toml`。

## 二、建议后续优化（参考 CODE_QUALITY_REVIEW.md）

| 优先级 | 项 | 说明 |
|--------|-----|------|
| ~~P1~~ | ~~value_at_risk / cvar cutoff 校验~~ | ✅ 已完成 |
| ~~P1~~ | ~~information_ratio 零追踪误差~~ | ✅ 已完成 |
| ~~P2~~ | ~~excess_sharpe 波动率阈值~~ | ✅ 已完成 |
| ~~P2~~ | ~~alpha_ 参数重命名~~ | ✅ 已完成（volatility_power，alpha_ 保留为弃用别名） |
| ~~P3~~ | ~~类型注解~~ | ✅ 已完成（consecutive、positions、alpha_beta、transactions 等核心 metrics 模块） |
| ~~P3~~ | ~~FIXME 处理~~ | ✅ 已完成（DURATION_STATS 补充 Avg # round_trips per day/month） |
| P4 | gpd_risk_estimates 重构 | ✅ 已完成（嵌套函数提取到模块级） |
| P4 | docstring 完善 | ✅ 已完成（omega/sterling/burke/downside_risk） |
| P5 | Ruff C4 + Bandit | ✅ 已完成（C4 规则，C408 暂忽略；Bandit CI） |
| P5 | VaR/CVaR docstring 对齐 | ✅ 已完成 |
| P6 | examples/ 纳入 lint 范围 | ✅ 已完成（CI + pre-commit；修复 PIE790/F541/B905/E722） |
| P6 | benchmarks/ 纳入 lint 范围 | ✅ 已完成（CI + pre-commit） |
| P6 | rolling/timing/perf_stats/perf_attrib 类型注解 | ✅ 已完成 |
| P6 | perf_attrib docstring 完善 | ✅ 已完成 |
| P6 | 串行测试标记（serial） | ✅ 已完成（mock 敏感测试使用 -n 0） |
| ~~P7~~ | ~~Ruff PERF/PTH 规则~~ | ✅ 已完成（见下方 12） |
| P8 | examples/ Path 迁移 | ✅ 已完成（2026-03-11，见下方 16） |
| P9 | 行业最佳实践优化（2026-03-11） | ✅ 已完成（见下方 29） |

### 12. Ruff PTH/PERF 规则启用（2026-03-10）

**变更**:

- **pyproject.toml**: Ruff `select` 新增 `PTH`、`PERF` 规则组。
- **fincore/**: `render_pdf.py`、`viz/html_backend.py` 中 `open()` 改为 `Path.open()` 或 `Path.write_text()`。
- **tests/**: `test_perf_attrib.py`、`test_pyfolio/*`、`perf_attrib/conftest.py` 等使用 `Path(__file__).resolve().parent` 替代 `os.path.dirname`/`os.path.join`；`test_viz`、`test_report`、`test_plotly_coverage` 中 `open()` 改为 `Path.read_text()`/`Path.open()`；`test_providers_*` 中 `for-append` 改为 `list.extend`（PERF401）。
- **scripts/**: `update_global_index_data.py` 全面使用 `Path` 替代 `os.path`。
- **benchmarks/**: `bench_metrics.py` 中 `open()` 改为 `Path.write_text()`。
- **examples/**: 通过 `per-file-ignores` 暂不启用 PTH/PERF，后续可增量修复。

**已知问题（已解决）**: `tests/test_metrics/test_alpha_beta_line_543_596.py` 与 `tests/test_empyrical/test_empyrical_line_718_coverage.py` 中基于 mock 的测试在 `pytest -n 4` 并行执行时偶发失败。已添加 `@pytest.mark.serial`，CI 中串行运行 (`-n 0`)。

### 13. Mypy 类型收窄与 perf_attrib 修复（2026-03-10）

**变更**:

- **fincore/metrics/rolling.py**:
  - 使用 `isinstance(returns_aligned, pd.Series)` 替代 `if is_series:` 在提前返回分支，帮助 mypy 正确收窄 `pd.Series | np.ndarray` 联合类型。
  - 在 `if not is_series:` 转换块后添加 `assert isinstance(returns_aligned, pd.Series)` 断言，消除 28 个 `union-attr` 错误。
  - `roll_max_drawdown`: 移除未使用的 `is_series` 变量，统一使用 `isinstance(returns, pd.Series)`。
- **fincore/metrics/perf_attrib.py**:
  - `missing_dates_displayed` 在 `<= 5` 分支由 `list(...)` 改为 `str(list(...))`，与 `> 5` 分支类型一致，修复 mypy `assignment` 错误。

### 14. 行业最佳实践补充（2026-03-10 续）

**变更**:

- **Pathlib 替代 os.path** (`fincore/core/context.py`, `fincore/report/render_html.py`, `fincore/report/render_pdf.py`): 使用 `Path(path).write_text()`、`Path(output).write_text()` 替代 `open(path, "w")`；使用 `Path(output).resolve().parent` 替代 `os.path.dirname(os.path.abspath(output))`；使用 `Path.unlink(missing_ok=True)` 替代 `os.remove()`。
- **PERF401 列表推导式** (`fincore/utils/common_utils.py`): `get_colormap_colors` 中 for 循环改为 list comprehension，提升可读性与性能。
- **异常处理最佳实践** (`examples/16_custom_optimization.py`): 将 `except Exception: pass` 改为 `except (ValueError, RuntimeError, OptimizationError): continue`，显式捕获预期异常类型，避免静默吞掉所有错误。
- **OptimizationError 导出** (`fincore/optimization/__init__.py`): 将 `OptimizationError` 加入公开 API，便于用户捕获优化失败异常。

### 15. 异常可观测性与示例 Path 优化（2026-03-10）

**变更**:

- **gpd_risk_estimates 异常日志** (`fincore/metrics/risk.py`): GPD 优化循环中的 `except (ValueError, RuntimeError, FloatingPointError): pass` 改为记录 `logger.debug`，便于调试时追踪优化失败原因。
- **examples/17_visualization_backends.py**: `open()` 改为 `Path.write_text()`，符合 PTH100/103 最佳实践。
- **examples/12_data_provider_usage.py**: 文档内示例代码中的 pickle 缓存由 `open()` 改为 `Path.read_bytes()`/`Path.write_bytes()`，展示 Pathlib 用法。

### 16. examples/ 全面 Path 迁移与 PTH/PERF 规则启用（2026-03-11）

**变更**:

- **examples/011_abberation/plot_tearsheet.py**: `os.path`、`os.listdir`、`open()` 全部改为 `pathlib.Path`；`json.load(f)` 改为 `json.loads(Path.read_text())`；`os.makedirs` 改为 `Path.mkdir()`。
- **examples/011_abberation/run.py**: `log_dir` 使用 `BASE_DIR / "logs"`；`load_config` 使用 `Path.read_text()` 替代 `open()`。
- **examples/011_abberation/analyze_strategy.py**: 全面迁移至 Path。
- **examples/analyze_with_fincore.py**: 全面迁移至 Path。
- **examples/generate_report.py**: `sys.path.insert` 使用 `Path(__file__).resolve().parent.parent`；`create_strategy_report` 的 `output` 参数传 `str(Path)` 以兼容 API。
- **examples/quick_start.py**: `html_path` 使用 `Path(tmpdir) / "report.html"`；`os.path.getsize` 改为 `Path.stat().st_size`。
- **pyproject.toml**: 移除 `examples/**` 的 PTH/PERF per-file-ignores，examples 现已符合 PTH/PERF 规则。

### 17. Ruff RUF 规则与代码质量强化（2026-03-11）

**变更**:

- **pyproject.toml**: Ruff `select` 新增 `RUF` 规则组；`ignore` 新增 RUF001/RUF002/RUF003（中文文档与数学符号中的 Unicode  intentionally）、RUF012（测试类可变属性）、RUF043（pytest match 正则）。
- **自动修复**（`ruff check --fix --unsafe-fixes`）:
  - RUF022: 全库 `__all__` 列表按字母排序（30+ 模块）。
  - RUF005: 使用 `[*a, *b]` 替代 `a + b` 拼接（fincore/attribution, metrics/drawdown, hooks/events 等）。
- **手动修复**:
  - RUF059: `tests/test_metrics/test_bayesian_pymc_stub.py` 中未使用的 `model` 解包变量改为 `_model`。

### 18. ERA001 注释代码清理与 DOC 规则启用（2026-03-11）

**ERA001 变更**:

- **删除死代码**: `fincore/pyfolio.py` 移除 ~40 行已废弃的 `plot_symbol_round_trips` 实现；`fincore/utils/common_utils.py` 移除调试用 `print` 注释。
- **重构文档式注释**: `_registry.py` 中 `# (method_name, ...)` 改为 `# Each entry: ...`；`brinson.py` 中 BHB 公式注释合并为单行；`ratios.py` 函数分组说明并入模块 docstring；`pyfolio.py` 模块结构注释改为正式 docstring；`tearsheets/__init__.py`、`garch.py` 中分组标签改为非代码形式。
- **per-file-ignores**: `examples/**`、`tests/**`、`benchmarks/**` 忽略 ERA001（保留配置片段与说明性注释）。

**DOC 规则变更**:

- **pyproject.toml**: Ruff 新增 `D100`（公开模块需 docstring）、`D104`（公开包需 docstring）。
- **新增模块 docstring**: `fincore/__init__.py`、`fincore/pyfolio.py`、`fincore/utils/common_utils.py`、`fincore/utils/deprecate.py`。
- **per-file-ignores**: `tests/**`、`scripts/**` 忽略 D100/D104（测试与脚本暂不强制模块 docstring）。

### 19. HTTP 超时、异常可观测性与 Ruff S 规则（2026-03-11）

**P0 安全与健壮性**:

- **HTTP 超时** (`fincore/data/providers.py`): 新增模块级常量 `HTTP_TIMEOUT = 30`；`AlphaVantageProvider` 的 `fetch` 与 `get_info` 中两处 `_session.get()` 调用均添加 `timeout=HTTP_TIMEOUT`，避免请求无限挂起（符合 S113 / request-without-timeout 最佳实践）。
- **静默异常修复** (`fincore/report/format.py`): `css_cls()` 中的 `except (TypeError, ValueError): pass` 改为 `contextlib.suppress(TypeError, ValueError)`，并补充注释说明 `np.isnan` 对部分数值类型的边缘情况会抛出异常，语义不变、可观测性更好。

**P1 Ruff S 规则**:

- **pyproject.toml**: Ruff `select` 新增 `S110`（try-except-pass 检测）、`S113`（request-without-timeout）。
- **tests/test_risk/evt/test_gpd_fit.py**: `test_gpd_mle_beta_le_zero_branch` 中的 `try-except-pass` 改为 `with (patch.object(...), contextlib.suppress(Exception)):` 组合 `with`，满足 S110 与 SIM117。

### 20. B904 raise-from 与 report/format 类型注解（2026-03-11）

**B904 启用与修复**:

- **pyproject.toml**: 从 `ignore` 列表移除 `B904`，启用「except 内 raise 需使用 from」规则。
- **修复 8 处违规**：`fincore/data/providers.py`（YahooFinance、AlphaVantage、Tushare、AkShare 四类 Provider 的 `except ImportError`）、`fincore/metrics/basic.py`（`except KeyError`）、`fincore/report/render_pdf.py`、`fincore/viz/interactive/bokeh_backend.py`、`fincore/viz/interactive/plotly_backend.py`，均改为 `raise ... from e` 以保留异常链。

**report/format.py 类型注解**:

- 新增 `typing.Any`、`collections.abc.Iterable`/`Mapping`、`pandas` 导入。
- `fmt`、`css_cls`、`html_table`、`html_df`、`html_cards`、`safe_list`、`date_list` 补充完整参数与返回值类型；`fmt` 内部使用 `int(v)`/`float(v)` 以消除 mypy 对 numpy 标量的 operator 告警。

### 21. metrics/bayesian 与 tearsheets 类型注解（2026-03-11）

**metrics/bayesian.py**:

- 为 12 个公开函数补充完整类型注解：`model_returns_t_alpha_beta`、`model_returns_normal`、`model_returns_t`、`model_best`、`model_stoch_vol`、`compute_bayes_cone`、`compute_consistency_score`、`run_model`、`simulate_paths`、`summarize_paths`、`forecast_cone_bootstrap`。
- 返回值使用 `tuple[Any, Any]`（PyMC model/trace）、`dict[int, list[float]]`、`list[float]`、`pd.DataFrame`、`np.ndarray`。
- `compute_consistency_score` 使用 `np.asarray(returns_test_cum)` 以兼容 Series/ndarray；`summarize_paths` 中 `cone_std` 使用局部变量 `stds: list[float]` 消除 mypy 告警。

**tearsheets**:

- **perf_attrib.py**: `plot_perf_attrib_returns`、`plot_alpha_returns`、`plot_factor_contribution_to_perf`、`plot_risk_exposures`、`show_perf_attrib_stats` 补充参数与返回值类型。
- **capacity.py**: `plot_capacity_sweep`、`plot_cones` 补充类型。
- **risk.py**: `plot_style_factor_exposures`、`plot_sector_exposures_*`、`plot_cap_exposures_*`、`plot_volume_exposures_*` 共 9 个函数补充类型。

### 22. Ruff 格式统一（2026-03-11）

**变更**:

- 对 4 个未按 Ruff 格式化的文件执行 `ruff format`:
  - `fincore/data/providers.py`
  - `fincore/metrics/basic.py`
  - `fincore/viz/interactive/bokeh_backend.py`
  - `fincore/viz/interactive/plotly_backend.py`
- 确保全库 427 个文件符合 Ruff 格式规范。

### 23. Ruff RET 规则与返回语句优化（2026-03-11）

**变更**:

- **pyproject.toml**: Ruff `select` 新增 `RET` 规则组（行业最佳实践：简化返回逻辑、消除冗余分支）。
- **自动修复**（`ruff check --fix --unsafe-fixes --select RET`）:
  - RET505: 移除 `else` 后多余的 `return`（superfluous-else-return），约 25 处。
  - RET502: 修复隐式返回 `None`（implicit-return-value），约 3 处。
  - RET506: 移除 `else` 后多余的 `raise`（superfluous-else-raise），约 2 处。
  - RET504: 移除不必要的中间变量赋值（unnecessary-assign），直接返回表达式，约 54 处。
- 共修复 94 处 RET 违规，提升代码简洁性与可读性。
- 格式：`ruff format` 对 9 个受影响文件重新格式化。

### 24. Ruff SIM 与 TCH 规则启用（2026-03-11）

**SIM 规则**:

- **pyproject.toml**: 从 `ignore` 移除 SIM102、SIM105、SIM108、SIM118、SIM201（启用 collapsible-if、contextlib.suppress、三元运算符、key-in-dict、!= 替代 not ==）。
- **自动修复**（`ruff check --fix --unsafe-fixes --select SIM`）:
  - SIM108: if-else 改为三元表达式（约 19 处）。
  - SIM118: `key in d.keys()` 改为 `key in d`（约 6 处）。
  - SIM201: `not x == 0` 改为 `x != 0`（约 2 处）。
- **手动修复**:
  - SIM102: `context.py`、`exceptions.py`、`validation.py` 嵌套 if 合并为单条件。
  - SIM108: `ratios.py`、`evt.py` 三元表达式。
  - SIM105: `test_final_coverage_edges.py`、`test_coverage_final_edges/test_integration.py` 中 try-except-pass 改为 `contextlib.suppress()`。
- `providers.py`: 冗余 `pd.to_datetime(x) if isinstance(x, str) else pd.to_datetime(x)` 简化为 `pd.to_datetime(x)`。

**TCH 规则**:

- **pyproject.toml**: Ruff `select` 新增 `TCH` 规则组。
- **自动修复**（`ruff check --fix --unsafe-fixes --select TCH`）:
  - TC002/TC003: 将仅用于类型注解的导入移至 `TYPE_CHECKING` 块（如 `pandas`、`datetime`、`pathlib.Path`、`collections.abc.Callable` 等），共 22 处。
  - TC005: 删除空 `TYPE_CHECKING` 块。
  - TC006: 为 `typing.cast()` 添加引号。
- 涉及文件：`engine.py`、`hooks/events.py`、`alpha_beta.py`、`bayesian.py`、`consecutive.py`、`report/format.py`、`risk/garch.py`、`simulation/bootstrap.py`、`monte_carlo.py`、`paths.py`、`tearsheets/capacity.py`、`tearsheets/risk.py`、`viz/base.py`、`data/providers.py`、`optimization/_utils.py`、`plugin/registry.py`、`report/__init__.py` 及部分测试文件。
- I001: 对新增 `from __future__ import annotations` 与 TYPE_CHECKING 导入块执行 import 排序修复。

### 25. BLE001 规则与异常处理精准化（2026-03-11）

**变更**:

- **pyproject.toml**: Ruff `select` 新增 `BLE` 规则组（blind-except），禁止 `except Exception` 等过于宽泛的异常捕获；`scripts/**` 保留 `BLE001` 的 per-file-ignore（维护脚本需兼容网络/环境波动）。
- **examples/ 异常类型显式化**:
  - `07_market_timing_analysis.py`: EVT 改为 `(ImportError, ValueError, RuntimeError, FloatingPointError)`；GARCH 改为 `(ImportError, ValueError, RuntimeError)`。
  - `08_portfolio_optimization_deep_dive.py`: 有效前沿/风险平价改为 `(ImportError, ValueError, RuntimeError, np.linalg.LinAlgError)`。
  - `10_market_timing_analysis.py`: Cornell/R-squared 改为 `(ValueError, TypeError, KeyError, np.linalg.LinAlgError, FloatingPointError)` 及 `(ValueError, TypeError, KeyError)`。
  - `11_performance_attribution.py`: 滚动 Alpha 改为 `(TypeError, ValueError, KeyError)`。
  - `12_data_provider_usage.py`: 四类数据源改为 `(ConnectionError, TimeoutError, ValueError, KeyError, OSError, RuntimeError)`。
  - `15_report_generation.py`: PDF 改为 `(ImportError, OSError, ValueError, RuntimeError)`。
  - `17_visualization_backends.py`: 七处可视化后端改为 `(ValueError, TypeError, AttributeError, RuntimeError)`。
  - `20_complete_workflow.py`: 优化改为 `(ImportError, ValueError, RuntimeError, np.linalg.LinAlgError)`。
- **tests/**:
  - `test_providers_integration.py`: AkShare 集成测试改为 `(ConnectionError, TimeoutError, OSError, RuntimeError, ValueError, KeyError, AttributeError)`。
  - `test_common_utils_legend_coverage.py`: 图例排序 fallback 改为 `(RuntimeError, IndexError, TypeError, ValueError)`。
- 符合 PEP 8 与行业最佳实践：仅捕获预期异常，便于调试与异常链追踪。

### 26. Ruff LOG 与 G（logging）规则启用（2026-03-11）

**变更**:

- **pyproject.toml**: Ruff `select` 新增 `LOG`、`G` 规则组。
- **LOG（flake8-logging）**: LOG001 禁止直接实例化 `Logger`、LOG002 校验 `getLogger` 参数、LOG004/LOG007/LOG009/LOG014/LOG015 规范 exception/exc_info/root-logger 用法。
- **G（flake8-logging-format）**: G001–G004 禁止在日志首参使用 `str.format`/`%`/`+`/f-string；G010 要求使用 `warning` 替代 `warn`；G101 检测 `extra` 与 LogRecord 字段冲突；G201/G202 规范 `exc_info` 与 `exception` 用法。
- **验证**: 全库已符合上述规则，无新增违规（日志均采用 `logger.xxx("msg %s", arg)` 传递参数，符合惰性求值最佳实践）。

### 27. PGH 规则、魔数常量与 pre-commit 扩展（2026-03-11）

**PGH（pygrep-hooks）规则**:

- **pyproject.toml**: Ruff `select` 新增 `PGH` 规则组。
- **PGH003 修复**: `tests/test_attribution/coverage/test_style_result_summary_advanced.py` 中 3 处 `# type: ignore` 改为 `# type: ignore[arg-type]`，符合「类型忽略需指定具体规则码」的行业最佳实践。

**魔数常量提取**:

- **fincore/attribution/fama_french.py**: Newey-West 自相关计算中的 `1e-15` 提取为模块级常量 `_MIN_STD`，与 `ratios.py` 等模块保持一致，避免魔数散落。

**pre-commit 扩展**:

- **.pre-commit-config.yaml**: 新增 Bandit 安全扫描 hook，在每次 commit 前对 `fincore/` 运行 `bandit -c pyproject.toml`，与 CI security job 保持一致的检查策略。

### 28. Future Annotations 全面启用（2026-03-11）

**变更**:

- **行业最佳实践**: 为尚未使用 `from __future__ import annotations` 的模块补充导入，实现全库统一的延迟求值注解。
- **受益**: 支持 `X | Y` 联合类型语法、避免前向引用问题、提升 mypy 兼容性。
- **涉及文件**:
  - `fincore/__init__.py`（主包入口）
  - `fincore/empyrical.py`（核心 Empyrical 类）
  - `fincore/pyfolio.py`（Pyfolio 类）
  - `fincore/_registry.py`（注册表）
  - `fincore/constants/__init__.py`、`periods.py`、`interesting_periods.py`、`color.py`

**备注**: Ruff N（命名）规则暂未启用。量化金融代码中常见的数学符号（X、T、N、S0、Z 等）在领域内约定俗成，保留可读性更佳。

### 29. 行业最佳实践优化（2026-03-11）

**变更**（基于 BMAD 边缘情况猎人 + 代码质量分析）:

- **魔数常量化**:
  - `fincore/optimization/risk_parity.py`: 协方差年化因子 `252` 改为 `APPROX_BDAYS_PER_YEAR`。
  - `fincore/viz/html_backend.py`: `plot_rolling_sharpe` 默认 `window=252` 改为 `window=APPROX_BDAYS_PER_YEAR`。
- **子进程超时（S113 最佳实践）**:
  - `scripts/test_runner.py`: `subprocess.run` 添加 `timeout=60`。
  - `scripts/verify_environment.py`: `subprocess.run` 添加 `timeout=30`；`except Exception` 收紧为 `except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError)`。
  - `tests/benchmarks/test_performance.py`: 三处 `subprocess.run` 添加 `timeout=120`。
- **validation.py 注释**: `validate_input` 中宽泛 `except` 补充说明注释，解释为何需捕获多种异常（第三方 validator 可能抛出 KeyError、ValueError 等）。
- **tearsheets/__init__.py**: `__getattr__` 补充 `name: str`、`-> Any` 类型注解及 docstring，符合类型与文档规范。

### 30. 后续优化 1–4 执行（2026-03-11）

**1. Pre-commit 增加 mypy**:
- `.pre-commit-config.yaml`: 新增 `local` repo 的 mypy hook，与 CI 相同的模块列表（fincore/core, constants, metrics, plugin, data, optimization, attribution, report, risk, simulation, utils, viz, empyrical, tearsheets, pyfolio），仅对 `^fincore/` 变更触发。

**2. 依赖版本升级**:
- `pyproject.toml`: `numpy>=1.17.0` → `numpy>=1.24.0`，`pandas>=0.25.0` → `pandas>=1.5.0`；`parameterized` 指定为 `parameterized>=0.7`。
- `requirements-test.txt`: 与 pyproject.toml 对齐，numpy、pandas 使用相同下限。

**3. tearsheets 魔数常量化**:
- `fincore/constants/style.py`: 新增 `LIQUIDATION_DAILY_VOL_LIMIT`(0.2)、`TRADE_DAILY_VOL_LIMIT`(0.05)、`CAPACITY_CAPITAL_BASE`(1e6)、`CAPACITY_SWEEP_MIN_PV`(100k)、`CAPACITY_SWEEP_MAX_PV`(300M)、`CAPACITY_SWEEP_STEP`(1M)，并导出至 `constants/__init__.py`。
- `fincore/tearsheets/sheets.py`: `create_capacity_tear_sheet` 使用上述常量；`plot_capacity_sweep` 调用传入命名常量。
- `fincore/tearsheets/capacity.py`、`fincore/pyfolio.py`: `plot_capacity_sweep` 默认参数改用常量。

**4. CI 覆盖率阈值**:
- `.github/workflows/ci.yml`: coverage 任务添加 `--cov-fail-under=60`，确保整体覆盖率不低于 60% 时 CI 通过。

### 31. 边缘情况参数校验（2026-03-11）

**变更**（基于 BMAD review-edge-case-hunter + 行业最佳实践）:

- **SimResult.var / cvar** (`fincore/simulation/base.py`): 为 `alpha` 添加 `(0, 1)` 校验，非法时返回 `float("nan")`，与 VaR/CVaR 行为一致。
- **bootstrap_ci** (`fincore/simulation/bootstrap.py`): 为 `alpha` 添加 `(0, 1)` 校验，非法时 `raise ValueError`，避免 `np.percentile` 边界未定义行为。
- **_calculate_size_exposure** (`fincore/attribution/style.py`): 为 `quantiles` 添加校验，要求 `len(quantiles) >= 2` 且 `0 <= q <= 1`，非法时 `raise ValueError`。
- **新增测试**:
  - `test_sim_result_var_cvar_invalid_alpha_returns_nan`: 验证 var/cvar 对 alpha=0/1/1.5 返回 NaN。
  - `test_bootstrap_ci_invalid_alpha_raises`: 验证 bootstrap_ci 对 alpha=0/1/1.5 抛出 ValueError。
  - `test_style_analysis_invalid_size_quantiles_raises`: 验证 style_analysis 对非法 size_quantiles 抛出 ValueError。

### 32. Dependabot 依赖更新自动化（2026-03-11）

**新增**: `.github/dependabot.yml`

- 每周检查 pip 与 GitHub Actions 依赖更新。
- 符合行业最佳实践：减少安全漏洞暴露窗口、自动跟踪上游修复。

### 33. 代码质量优化（2026-03-11）

**基于 BMAD review-edge-case-hunter + 行业最佳实践**:

- **Ruff 格式**: 对 `fincore/attribution/style.py`、`tests/test_attribution/test_style_analysis.py` 执行 `ruff format` 统一格式。
- **Flaky 测试标记**: `test_returns_plots_cover_ax_none_and_live_start_date_str`、`test_annual_alpha_mock_to_hit_line_543` 添加 `@pytest.mark.serial`，避免并行执行下 matplotlib/mock 相关偶发失败。
- **conditional_sharpe_ratio** (`fincore/metrics/ratios.py`): 在 `std_ret == 0` 检查后追加 `np.isfinite(mean_ret)` 与 `np.isfinite(std_ret)` 校验，避免 inf 传播。
- **value_at_risk / conditional_value_at_risk** (`fincore/metrics/risk.py`): 对 `returns` 添加 `np.isfinite` 校验，非有限值时返回 NaN；VaR 返回值显式 `float()` 以消除 mypy return-value 告警。
- **downside_risk** (`fincore/metrics/risk.py`): 仅对**标量** `required_return` 增加 nan/inf 校验（数组型 required_return 允许 per-period NaN，保持向后兼容）。

## 三、验证命令

```bash
# Lint
ruff check fincore/ tests/ scripts/ examples/ benchmarks/

# Format
ruff format fincore/ tests/ scripts/ examples/ benchmarks/

# Security scan
bandit -r fincore/ -c pyproject.toml

# Type check
mypy fincore/core fincore/constants fincore/metrics fincore/plugin fincore/data ...

# Tests
pytest tests/ -n 4 -m "not slow and not integration" --ignore=tests/benchmarks/
```
