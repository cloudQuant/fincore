# 0036 全面质量提升与 Bug 修复计划

## 问题发现日期

2026-02-14

## 进展快照（更新于 2026-02-14）

### 当前实时状态（本地实测）

1. `pytest tests -q`：**1685 passed**, **14 skipped**, **0 failed**。
2. `pytest --collect-only`：收集用例 **1699**。
3. `ruff check fincore tests`：**0 issues**。
4. `mypy`（`core/metrics/plugin/data/optimization/attribution/report/risk/simulation/utils/viz/empyrical/tearsheets/pyfolio`）通过（`ignore_errors` 已清零）。
5. 已新增 `RuntimeWarning` 防回归修复与测试（`stats/style/fama_french/report.compute`）。
6. 已完成优化模块输入鲁棒性加强（`frontier/objectives/risk_parity` 参数校验与异常路径测试）。
7. 已统一 `pytest` 配置到 `pyproject.toml`，并移除 `pytest.ini`，消除双源漂移。
8. 已实现 `fetch_ff_factors` provider 注入与进程内缓存（可测试、可替换、默认返回拷贝以防止外部误改缓存对象）。
9. CI 增强：加入 `compileall` 语法检查、`ruff format --check` 格式一致性检查，并扩大 `mypy` 覆盖面。

### 已完成（相对原计划）

- Phase A：style 归因 API 端到端可调用与契约兼容（`StyleResult` 契约、独立调用路径）。
- Phase B：关键 P1 缺陷修复（含归因/统计边界条件与优化模块异常路径）。
- Phase D：补充归因、报表、优化、统计边界测试（覆盖退化输入和不合理参数）。
- Phase E（部分）：ruff 从历史 230 告警收敛至当前 0 告警。

---

## 一、历史基线状态（首次审计）

> 以下数据为 2026-02-14 首次审计快照，已不代表最新状态；最新状态以上方“进展快照”为准。

### 已达成

1. `pytest tests -n 8`：**1569 passed**，0 failed，14 skipped（均为集成测试）。
2. `mypy fincore --ignore-missing-imports`：**0 errors**，85 个源文件全通过。
3. `python -m compileall -q fincore`：通过。
4. 已完成迭代 0028–0035 的主要目标：插件系统修复、归因模块修复、优化模块 MVP、报告模块化、Bokeh 兼容修复等。

### 仍存在的问题汇总

通过全面代码审计（ruff 静态分析、grep 扫描、运行时验证、测试覆盖分析）发现以下问题：

- **P0 Bug**：4 个（style 归因 API 运行时错误）
- **P1 Bug**：8 个（未使用变量引发的潜在逻辑错误、名称遮蔽、异常吞没）
- **P2 代码质量**：230 个 ruff 警告、45+ 模块缺少专属测试
- **P3 工程改进**：大文件拆分、文档漂移、配置统一

---

## 二、Bug 清单（按优先级）

### P0：style 归因 API 运行时错误（4 个）

这些函数已在 `fincore/attribution/__init__.py` 对外导出，但调用即报错或返回无效结果。

| # | 文件 | 函数 | 问题描述 | 实测结果 |
|---|------|------|----------|----------|
| 1 | `attribution/style.py` | `calculate_style_tilts()` | 返回空 DataFrame `(0, 0)`，未执行任何 tilt 计算 | 调用不报错但结果为空 |
| 2 | `attribution/style.py` | `calculate_regression_attribution()` | 需要 `style_returns` 和 `style_exposures` 两个位置参数，但对外导出未说明依赖 `style_analysis` 的前置输出 | `TypeError: missing 2 required positional arguments` |
| 3 | `attribution/style.py` | `analyze_performance_by_style()` | 需要 `style_exposures` 位置参数，同上问题 | `TypeError: missing 1 required positional argument` |
| 4 | `attribution/style.py` | `style_analysis()` | 返回 `StyleResult` 对象而非 dict，但部分下游代码（如 0035 报告）用 `in` 运算符检查键 | 语义不一致，`'style_summary' in result` 抛 `TypeError` |

**根因分析**：style 模块的 API 设计存在断裂——`style_analysis` 产生中间数据，但下游函数要求这些中间数据作为独立参数传入，缺少端到端的 pipeline 设计。`calculate_style_tilts` 内部逻辑在无 style factor 数据时直接返回空 DataFrame。

### P1：代码缺陷与名称遮蔽（8 个）

| # | 文件:行 | 类型 | 问题描述 |
|---|---------|------|----------|
| 5 | `metrics/ratios.py:1048` | F811 名称遮蔽 | `tail_ratio` 函数重定义遮蔽了 `:25` 行的导入 |
| 6 | `core/context.py:128` | F811 名称遮蔽 | `annual_volatility` 属性遮蔽了 `:40` 行的导入 |
| 7 | `metrics/consecutive.py:86-87` | E731 lambda 赋值 | `up = lambda s: s > 0` / `down = lambda s: s < 0` 应改为 `def` |
| 8 | `report/render_pdf.py:150` | F841 未使用变量 | `total_height` 赋值后未使用（书签页码计算缺失） |
| 9 | `risk/evt.py:312,421,430` | F841 未使用变量 | `n`、`m`、`mu` 被赋值但未参与后续计算——**可能为计算逻辑遗漏** |
| 10 | `risk/garch.py:65,71,212` | F841 未使用变量 | `gamma`、`long_run_var`、`T` 被赋值但未使用——**`long_run_var` 未参与预测公式可能导致 GARCH 长期方差回归不正确** |
| 11 | `viz/interactive/plotly_backend.py:236` | F841 未使用变量 | `colors` 赋值后未使用（可能图表着色逻辑缺失） |
| 12 | `attribution/brinson.py:353` | F841 未使用变量 | `portfolio_returns` 计算后未使用 |

**重点关注 #10**：`garch.py` 的 `forecast()` 方法中 `long_run_var` 被计算但未用于预测公式，当前预测使用 `omega + (alpha + beta) * forecasts[h-1]`，这在 `alpha + beta < 1` 时缺少向长期方差均值回归的修正项。

### P1：异常吞没（19 处）

以下位置使用 `except Exception` 后静默 `pass` 或仅 `print`，导致错误不可观测：

| 模块 | 处数 | 影响 |
|------|------|------|
| `data/providers.py` | 5 | 批量抓取部分失败时上层无感知 |
| `metrics/timing.py` | 4 | 市场择时计算失败静默返回 NaN |
| `report/compute.py` | 3 | 报告关键指标可能无声缺失 |
| `metrics/round_trips.py` | 2 | 往返交易统计出错静默填 NaN |
| `metrics/stats.py` | 2 | Hurst 指数等计算失败静默返回 NaN |
| `empyrical.py` | 1 | perf_stats 内部异常被吞 |
| `pyfolio.py` | 1 | matplotlib backend 切换失败被吞 |
| `utils/common_utils.py` | 1 | 文件保存出错仅 print |

---

## 三、改进项清单

### P1：ruff 静态分析问题（230 个）

| 规则 | 数量 | 说明 |
|------|------|------|
| F401 未使用导入 | 136 | 大量模块有未使用的 import（部分可能为动态使用需保留） |
| E402 非顶部导入 | 68 | 部分为延迟导入（有意为之），部分为代码组织问题 |
| F841 未使用变量 | 18 | 见 P1 Bug 清单 |
| E501 行过长 | 4 | `perf_attrib.py`、`format.py`、`render_html.py` |
| E731 lambda 赋值 | 2 | `consecutive.py` |
| F811 名称重定义 | 2 | `ratios.py`、`context.py` |

### P1：测试覆盖严重不足（45+ 模块无专属测试）

以下核心模块没有专属测试文件，仅依赖 smoke import 或间接覆盖：

**归因模块**（3 个）：
- `attribution/brinson.py`、`attribution/fama_french.py`、`attribution/style.py`（`test_attribution.py` 存在但覆盖薄弱）

**指标模块**（10 个）：
- `metrics/basic.py`、`metrics/bayesian.py`、`metrics/consecutive.py`、`metrics/drawdown.py`
- `metrics/perf_stats.py`、`metrics/positions.py`、`metrics/returns.py`、`metrics/rolling.py`
- `metrics/transactions.py`、`metrics/yearly.py`

**报告模块**（4 个）：
- `report/compute.py`、`report/format.py`、`report/render_html.py`、`report/render_pdf.py`

**风险模块**（2 个）：
- `risk/evt.py`、`risk/garch.py`（有 `test_risk_models.py` 但可能覆盖不足）

**可视化模块**（4 个）：
- `viz/base.py`、`viz/html_backend.py`、`viz/matplotlib_backend.py`、`viz/interactive/plotly_backend.py`

**工具模块**（5 个）：
- `utils/common_utils.py`、`utils/data_utils.py`、`utils/date_utils.py`、`utils/deprecate.py`、`utils/math_utils.py`

**其他**（7+ 个）：tearsheets 全系列、`empyrical.py`、`pyfolio.py`、`simulation/base.py` 等

### P2：大文件拆分

| 文件 | 行数 | 建议 |
|------|------|------|
| `metrics/ratios.py` | 1331 | 按功能域拆分：基础比率、风险调整比率、下行比率 |
| `pyfolio.py` | 1244 | 已有 `empyrical.py` 分离，可进一步提取工具函数 |
| `tearsheets/sheets.py` | 1070 | 按报告类型拆分：全量、简要、有趣时期 |
| `data/providers.py` | 921 | 每个 provider 独立文件 |

### P2：工程治理

1. **pytest 配置双源**：`pyproject.toml` 和 `pytest.ini` 并存，存在漂移风险。
2. **文档漂移**：README 中测试数量（1299）与实际（1569+）不一致。
3. **mypy ignore_errors 范围过宽**：`viz`、`empyrical`、`pyfolio`、`utils`、`report` 整体忽略。
4. **优化模块求解器鲁棒性**：`frontier.py`、`objectives.py`、`risk_parity.py` 均未检查 `res.success`。

---

## 四、0036 目标

1. 修复全部 P0 Bug，使 style 归因 API 端到端可用。
2. 修复 P1 关键 Bug（GARCH 预测公式、名称遮蔽、未使用变量）。
3. 补齐核心模块测试覆盖，消除"假绿"风险。
4. 收敛异常处理策略，建立可观测性基线。
5. 清理 ruff 静态分析问题，目标警告数降至 50 以下。

---

## 五、任务分解

### Phase A（P0）修复 style 归因 API

1. **重构 `calculate_style_tilts`**：修复空 DataFrame 返回问题，实现完整的 tilt 计算逻辑。
2. **重构 `calculate_regression_attribution` / `analyze_performance_by_style`**：
   - 方案一：改为接受 `StyleResult` 对象而非裸数据；
   - 方案二：提供端到端 pipeline 函数，内部调用 `style_analysis` 获取前置数据。
3. **统一 `StyleResult` 契约**：实现 `__contains__`（支持 `in` 运算符）和 `__getitem__`（支持 `result['key']` 语法），或改为返回 dict。
4. **补充 style 端到端契约测试**：键白名单、维度校验、贡献求和守恒。

### Phase B（P1）修复代码缺陷

1. **修复 GARCH `forecast()` 长期方差回归**（`risk/garch.py:65-81`）：将 `long_run_var` 纳入预测公式。
2. **修复 EVT 未使用变量**（`risk/evt.py`）：确认 `n`、`m`、`mu` 是否应参与计算。
3. **修复名称遮蔽**：
   - `metrics/ratios.py:1048` → 重命名局部导入或调整导入结构。
   - `core/context.py:128` → 重命名局部属性或导入别名。
4. **清理 lambda 赋值**（`consecutive.py:86-87`）→ 改为 `def`。
5. **修复 `report/render_pdf.py:150`**：`total_height` 要么用于书签计算，要么移除。

### Phase C（P1）异常处理治理

1. 将 `metrics/timing.py`、`metrics/stats.py`、`metrics/round_trips.py` 中的 `except Exception: return np.nan` 改为记录 `logger.debug` 后再返回 NaN。
2. 将 `report/compute.py` 中 `except Exception: pass` 改为 `logger.warning`（已部分完成，确认遗留）。
3. 为 `data/providers.py` 的 `fetch_multiple` 增加 `strict: bool = False` 参数：
   - `strict=False`（默认）：记录 warning，继续执行。
   - `strict=True`：收集错误后抛出聚合异常。
4. 清理 `empyrical.py:193` 和 `pyfolio.py:46` 的 `except Exception: pass`。

### Phase D（P1）核心模块测试补齐

按优先级补齐以下测试（目标：每个核心模块至少 3 个用例）：

1. **`tests/test_report/test_compute.py`**：compute_sections 的空输入、全量输入、benchmark 分支、异常输入。
2. **`tests/test_report/test_format.py`**：fmt/css_cls/html_table/safe_list 边界值。
3. **`tests/test_metrics/test_drawdown.py`**：max_drawdown、gen_drawdown_table、drawdown_periods 数值正确性。
4. **`tests/test_metrics/test_rolling.py`**：rolling_sharpe、rolling_volatility、rolling_beta 窗口边界。
5. **`tests/test_risk/test_garch_detailed.py`**：GARCH forecast 长期回归、异常输入。
6. **`tests/test_attribution/test_style_contract.py`**：style 端到端契约（键集合、维度、守恒）。

### Phase E（P2）ruff 清理与工程治理

1. **清理 F401 未使用导入**：区分"动态使用需保留"（加 `# noqa: F401`）和"真正无用"（删除），目标将 136 降至 20 以下。
2. **清理 F841 未使用变量**：逐一确认是否为 Bug 或可安全标注。
3. **统一 pytest 配置**：保留 `pyproject.toml` 为唯一来源，删除或最小化 `pytest.ini`。
4. **更新 README**：测试数量、能力列表与当前代码对齐。
5. **缩小 mypy `ignore_errors`**：优先启用 `attribution`、`report`。

### Phase F（P2）优化模块鲁棒性

1. 为 `frontier.py`、`objectives.py`、`risk_parity.py` 增加 `res.success` 检查。
2. 对 NaN/inf 权重做结果守卫。
3. 增加异常场景测试：协方差近奇异、不可行约束。

---

## 六、验收标准（量化）

| 维度 | 指标 | 目标值 |
|------|------|--------|
| Bug 修复 | P0 style API 可端到端调用 | 4/4 修复 |
| Bug 修复 | P1 代码缺陷修复 | 8/8 修复 |
| 异常可观测 | `except Exception: pass` 残留 | ≤ 3 处（仅三方库兼容） |
| 测试覆盖 | 新增测试用例 | ≥ 40 个 |
| 静态分析 | ruff 警告数 | ≤ 50（当前 230） |
| 类型检查 | mypy ignore_errors 模块数 | ≤ 3（当前 5） |
| 文档 | README 与代码一致 | 测试数、能力列表对齐 |

---

## 七、建议执行节奏（2 周）

1. **第 1-3 天**：Phase A（style 归因 API 修复 + 契约测试）。
2. **第 4-5 天**：Phase B（GARCH 预测修复、名称遮蔽、未使用变量）。
3. **第 6-7 天**：Phase C（异常处理治理）。
4. **第 8-10 天**：Phase D（核心模块测试补齐）。
5. **第 11-12 天**：Phase E（ruff 清理 + 工程治理）。
6. **第 13-14 天**：Phase F（优化模块鲁棒性 + 最终回归）。

---

## 八、风险与控制

1. **风险**：style API 修复可能改变返回类型，影响下游调用方。
   - **控制**：提供 `to_dict()` 兼容方法，新增 `__contains__` / `__getitem__` 保持向后兼容。
2. **风险**：GARCH 预测公式修正后数值结果变化。
   - **控制**：先固化现有行为的回归测试，再引入修正并更新期望值。
3. **风险**：清理 F401 可能误删动态使用的导入。
   - **控制**：仅删除通过 `--unsafe-fixes` 确认安全的项，其余标注 `noqa`。
4. **风险**：异常处理变严格可能暴露历史静默容错路径。
   - **控制**：默认保持兼容（warning），通过 `strict` 参数可选严格模式。

---

## 附录 A：完整 ruff 扫描结果

```
136  F401  unused-import
 68  E402  module-import-not-at-top-of-file
 18  F841  unused-variable
  4  E501  line-too-long
  2  E731  lambda-assignment
  2  F811  redefined-while-unused
─────────────────────────────────
230  total
```

## 附录 B：无专属测试的核心模块清单

```
attribution/brinson.py          metrics/bayesian.py
attribution/fama_french.py      metrics/consecutive.py
attribution/style.py            metrics/drawdown.py
metrics/basic.py                metrics/perf_stats.py
metrics/positions.py            metrics/returns.py
metrics/rolling.py              metrics/transactions.py
metrics/yearly.py               report/compute.py
report/format.py                report/render_html.py
report/render_pdf.py            risk/evt.py
risk/garch.py                   utils/common_utils.py
utils/data_utils.py             utils/date_utils.py
utils/deprecate.py              utils/math_utils.py
viz/base.py                     viz/html_backend.py
viz/matplotlib_backend.py       viz/interactive/plotly_backend.py
simulation/base.py              empyrical.py
pyfolio.py                      tearsheets/sheets.py
tearsheets/bayesian.py          tearsheets/positions.py
tearsheets/returns.py           tearsheets/transactions.py
tearsheets/utils.py
```

## 附录 C：异常吞没位置明细

```
data/providers.py:294       except Exception as e  (fetch_multiple, 5处)
metrics/timing.py:72        except Exception        (treynor_mazuy_timing)
metrics/timing.py:112       except Exception        (henriksson_merton_timing)
metrics/timing.py:214       except Exception        (cornell_timing)
metrics/timing.py:240       except Exception        (extract_interesting_date_ranges)
metrics/round_trips.py:75   except Exception        (单笔统计)
metrics/round_trips.py:92   except Exception        (汇总统计)
metrics/stats.py:188        except Exception        (hurst_exponent)
metrics/stats.py:259        except Exception        (d_ratio)
report/compute.py:74        except Exception as e   (turnover 计算)
report/compute.py:87        except Exception as e   (gross leverage)
report/compute.py:221       except Exception as e   (turnover from txn)
empyrical.py:193            except Exception        (perf_stats 内部)
pyfolio.py:46               except Exception        (matplotlib backend)
utils/common_utils.py:385   except Exception as e   (文件保存)
```
