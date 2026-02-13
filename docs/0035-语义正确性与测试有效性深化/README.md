# 0035 语义正确性与测试有效性深化计划

## 问题发现日期

2026-02-13

---

## 一、0034 后基线状态（实测）

1. `pytest -q` 全量通过；集成测试默认跳过 14 条。
2. `ruff check fincore tests` 全通过。
3. `mypy fincore` 全通过；`python -m compileall -q fincore` 通过。
4. 迭代34修复已提交：`985ba24`。

---

## 二、当前仍存在的问题（按优先级）

## P0：style 归因“语义正确性”仍有缺陷（虽不报错但结果不可靠）

1. `style_analysis` 会把资产代码混入 style 维度。
   - 根因：`fincore/attribution/style.py:123` 使用 `size_exposure.to_dict(orient='index')` 写入 `exposures_data`，键变成资产名。
   - 影响：`fincore/attribution/style.py:181` 遍历 `all_exposures.columns` 计算 style 收益时，会把资产名也当 style 计算。
   - 实测：`style_summary` 出现 `A/B/C/D` 等资产键（应只包含 style 键）。
2. `calculate_regression_attribution` 的计算口径不一致。
   - `fincore/attribution/style.py:432` 取 `style_returns[style].iloc[0]`，未做时间对齐与聚合口径说明；
   - 变量 `style_return` 被定义后未参与贡献计算，公式语义与文档描述不一致。
3. style 单测覆盖了“能跑通”，但未校验关键语义约束（键集合、权重守恒、时间对齐），存在“假绿”风险。

## P1：报告与可视化链路仍有回归盲区

1. `report` 关键函数仍缺直接测试：`compute_sections`、`generate_pdf` 只有导入或间接覆盖，缺少输出契约测试。
2. 当前测试仍有 81 条 warning，主要来自 tearsheet 绘图链路：
   - seaborn `PendingDeprecationWarning`（`vert` 参数相关）；
   - matplotlib datetime timezone 警告。

## P1：异常治理仍未完全闭环

1. `fincore/empyrical.py:193` 和 `fincore/pyfolio.py:46` 仍有 `except Exception: pass`，异常不可观测。
2. `providers.fetch_multiple` 已改 warning，但尚无统一的 strict/fail-fast 策略，批量抓取时仍默认“部分失败继续”，上层难以感知严重失败比例。

## P2：工程可维护性问题持续存在

1. 超大文件仍偏多：`metrics/ratios.py`(1330行)、`pyfolio.py`(1244行)、`tearsheets/sheets.py`(1070行)、`data/providers.py`(921行)。
2. 类型检查虽通过，但 `pyproject.toml` 仍对多个核心模块 `ignore_errors = true`，类型门禁约束不足。
3. pytest 配置双源（`pyproject.toml` + `pytest.ini`）并存，仍有漂移空间。

---

## 三、0035 目标

1. 从“运行不报错”提升到“结果语义正确且可验证”。
2. 用契约测试替代弱断言，避免关键链路出现“假绿”。
3. 进一步收敛异常处理与门禁策略，提升发布可控性。

---

## 四、任务分解

## Phase A（P0）修复 style 归因语义正确性

1. 重构 `style_analysis` 的 exposure 组织方式，明确维度：`index=asset`、`columns=style`。
2. 保证 `returns_by_style` 仅包含 style 键，不得混入资产代码。
3. 明确并实现 `calculate_regression_attribution` 的口径：
   - 时间对齐（portfolio/style 同索引）；
   - style 收益聚合方式（均值或累计）；
   - 贡献公式一致且可解释。

## Phase B（P0）补强 style 契约测试

1. 在 `tests/test_attribution/test_style.py` 增加强断言：
   - style 键白名单；
   - 暴露矩阵形状与取值域（0/1或权重和）；
   - 归因贡献和残差一致性（`sum(contrib)+residual≈portfolio`）。
2. 增加最小数值回归样例（固定随机种子 + 期望值区间）。

## Phase C（P1）补齐报告关键路径测试

1. 新增 `tests/test_report/test_compute.py`：
   - 空/缺字段输入；
   - benchmark/positions/transactions 分支；
   - summary_text 字段完整性。
2. 新增 `tests/test_report/test_render_pdf.py`：
   - 无 Playwright 依赖时错误信息断言；
   - 临时文件清理行为断言（mock）。

## Phase D（P1）异常与 warning 治理

1. 将 `empyrical.py` / `pyfolio.py` 的裸 `pass` 改为可观测 warning/debug 日志。
2. 在 provider 层设计 `strict` 参数（默认兼容，支持 fail-fast）。
3. 消减绘图 warning（seaborn 参数适配、timezone 处理），目标将默认回归 warning 显著降低。

## Phase E（P2）门禁与维护性收敛

1. 缩小 mypy `ignore_errors` 范围，优先收紧 `attribution`、`report`、`tearsheets`。
2. 统一 pytest 配置入口，避免双份配置漂移。
3. 输出大文件拆分方案（先拆 `pyfolio.py`、`ratios.py`）。

---

## 五、验收标准（量化）

1. 语义正确性：
   - `style_summary` 不包含资产代码键；
   - style 归因的贡献与残差校验通过。
2. 测试有效性：
   - 新增 style/report 契约测试并纳入默认 CI；
   - 关键路径不再仅依赖 smoke 测试。
3. 质量信号：
   - 默认 `pytest -q` warning 数显著下降（目标 < 30）；
   - 异常处理路径具备可观测日志或可配置抛错。
4. 工程治理：
   - mypy `ignore_errors` 范围较 0034 进一步收敛；
   - pytest 配置单一事实来源落地。

---

## 六、建议节奏（1 周）

1. 第 1-2 天：Phase A（style 语义修复）。
2. 第 3 天：Phase B（style 契约测试）。
3. 第 4-5 天：Phase C（report 关键测试补齐）。
4. 第 6 天：Phase D（异常与 warning 治理）。
5. 第 7 天：Phase E（门禁收敛与拆分方案评审）。

---

## 七、风险与控制

1. 风险：style 语义修复可能引发下游结果变化。
   - 控制：先固化契约测试，再引入兼容开关并记录迁移说明。
2. 风险：warning 治理涉及三方库版本差异。
   - 控制：采用最小改动策略并在 CI 三平台回归验证。
3. 风险：mypy 收紧会短期增加修复成本。
   - 控制：分模块逐步启用，按优先级拆批推进。
