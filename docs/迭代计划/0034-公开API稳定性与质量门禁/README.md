# 0034 公开 API 稳定性与质量门禁迭代计划

## 问题发现日期

2026-02-13

---

## 一、当前基线（实测）

1. `pytest -q` 全量通过，但存在 14 个跳过（均为 integration）与 81 条 warning。
2. `python -m compileall -q fincore` 通过。
3. `mypy fincore` 通过（85 个源文件无错误）。
4. `ruff check fincore tests` 未通过：`tests/test_data/test_providers_integration.py:13` 存在 import 排序问题。

---

## 二、当前主要问题（按优先级）

## P0：公开 API 可用性风险（已复现）

1. `fincore.attribution.style` 多个公开函数存在运行时错误：
   - `style_analysis` 在 `fincore/attribution/style.py:121` 触发 `TypeError`（`where` 参数使用错误）。
   - `calculate_style_tilts` 在 `fincore/attribution/style.py:378` 触发 `KeyError: 'large'`。
   - `calculate_regression_attribution` 在 `fincore/attribution/style.py:437` 触发 `TypeError`（`np.corrcoef` 返回矩阵被直接转 float）。
   - `analyze_performance_by_style` 在 `fincore/attribution/style.py:487` 触发 `AttributeError`（Series 当作 DataFrame 使用）。
2. 以上 API 已在 `fincore/attribution/__init__.py` 对外导出，但测试未覆盖：`tests/` 中无上述函数的直接用例。

## P0：错误被静默吞掉，定位成本高

1. `fincore/data/providers.py` 多处 `except Exception` 后仅 `print` 并返回空表（如 `:290`, `:428`, `:574`, `:704`）。
2. `fincore/report/compute.py` 多处 `except Exception: pass`（`:70`, `:79`, `:209`），可能导致报告指标缺失但无可观测信号。

## P1：质量门禁存在盲区

1. CI lint 仅检查 `fincore/`，未覆盖 `tests/`（`.github/workflows/ci.yml`），导致测试代码风格问题无法在 CI 中被拦截。
2. `pytest` 配置在 `pyproject.toml` 与 `pytest.ini` 双份维护，存在漂移风险。
3. `report` 关键路径缺少直接回归测试：`tests/` 中无 `compute_sections` / `generate_pdf` 的直接测试。

## P2：可维护性与前向兼容风险

1. 超大文件偏多（如 `fincore/metrics/ratios.py` 1288 行，`fincore/pyfolio.py` 1219 行，`fincore/tearsheets/sheets.py` 1019 行），重构成本与回归半径大。
2. 测试存在依赖库弃用预警（seaborn/matplotlib 相关 warning），需提前治理以降低后续升级风险。

---

## 三、0034 迭代目标

1. 修复所有已导出的 style attribution API 运行时错误，达到“可调用、可测试、可解释”。
2. 建立“失败可观测”机制，避免静默降级。
3. 补齐 CI 质量门禁盲区（尤其是 tests 代码质量）。
4. 为报告与归因关键路径补齐回归测试，提升迭代安全性。

---

## 四、任务分解

## Phase A（P0）修复公开 API 运行时错误

1. 修复 `fincore/attribution/style.py` 的数据结构与索引逻辑错误（`style_analysis`、`calculate_style_tilts`、`calculate_regression_attribution`、`analyze_performance_by_style`）。
2. 明确输入输出契约（列名约定、返回 DataFrame/Series 形状、异常边界）。
3. 新增归因 style 模块单测，覆盖正常路径与异常路径。

## Phase B（P0）消除静默失败

1. 将 `providers.fetch_multiple` 中“吞异常 + 空表”改为可配置策略：
   - 默认记录结构化 warning（含 symbol/provider/错误类型）；
   - 提供 strict 模式可直接抛错。
2. 对 `report/compute.py` 中 `except Exception: pass` 改为显式告警或错误汇总输出，避免关键指标无声缺失。
3. 为失败路径补充单测，验证日志/异常行为符合预期。

## Phase C（P1）完善质量门禁

1. CI lint 扩展为 `ruff check fincore tests`。
2. 清理当前 lint 问题（先修复 `tests/test_data/test_providers_integration.py:13`）。
3. 统一 pytest 配置来源（保留一个主配置，另一个仅最小兼容或删除）。

## Phase D（P1）补齐关键路径测试

1. 为 `report.compute.compute_sections` 增加直接测试（基准为空、含 positions/transactions、异常输入）。
2. 为 `report.render_pdf.generate_pdf` 增加依赖缺失路径测试（无 Playwright 时错误提示可诊断）。
3. 增加 attribution-style 的 smoke + 数值合理性测试（非 NaN、维度一致、键完整）。

## Phase E（P2）可维护性改进

1. 制定大文件拆分计划（先从 `metrics/ratios.py` 与 `pyfolio.py` 开始，按能力域切分）。
2. 统一版本来源（`pyproject.toml` / `setup.py` / `fincore/__init__.py` 避免多点维护）。
3. 对 warning 做分级治理，优先处理可由本仓代码消除的 warning。

---

## 五、验收标准（量化）

1. 公开 API 稳定性：
   - 上述 4 个 style 函数新增回归测试并通过；
   - 不再出现当前复现的 4 类运行时异常。
2. 门禁质量：
   - `ruff check fincore tests` 0 错误；
   - CI 中新增 tests lint 检查并稳定通过。
3. 可观测性：
   - 数据抓取与报告计算失败路径有明确日志或异常，不再无声 `pass`。
4. 测试覆盖：
   - 新增 report/style 关键路径测试并纳入默认回归。

---

## 六、建议执行节奏（1 周）

1. 第 1-2 天：Phase A（style API 修复 + 单测）。
2. 第 3 天：Phase B（错误处理策略与可观测性）。
3. 第 4 天：Phase C（CI 门禁扩展 + 配置收敛）。
4. 第 5-6 天：Phase D（报告与归因关键路径测试补齐）。
5. 第 7 天：Phase E（拆分方案评审与 warning 治理清单）。

---

## 七、风险与控制

1. 风险：style 模块修复会改变已有输出形状。
   - 控制：先固化契约测试，再做实现替换，必要时提供兼容层。
2. 风险：严格错误策略可能影响现有“容错”调用方。
   - 控制：提供 `strict=False` 兼容默认，并记录 deprecation 路线。
3. 风险：CI 门禁收紧导致短期 PR 失败率上升。
   - 控制：分阶段启用，先修复现存问题再强制阻断。
