# 0042 Unified Analytics Platform — 实施与验收记录

日期：2026-08-21  
验收对象：`codex/fincore-iteration0042-completion`；初始数值修复与验收基线提交为 `93fbf54d9a03b9f14b37bfb25ce5c3b821ef5710`，后续经验证的局部切片列于下表。
结论：**BLOCKED — 不可宣称计划完成、不可作为 1.0 或发布候选。**

本记录区分“本轮已验证的代码切片”与“0042 计划的全部完成定义”。局部通过的测试、候选 wheel 或性能数字不替代法律、供应链、治理、完整类型和所有金融工作流的验收。

## 本轮已落地并验证的切片

- 数值安全：修正 Brinson 多期 Carino linking 的累计绝对主动收益对账，并增加近损失边界和独立 Decimal oracle；XIRR/MWR 明确日历日、标签对齐及非传统现金流 fail-closed 策略；walk-forward historical VaR 使用有限样本 Weibull 分位数并复核回测诊断。
- 风险验证报告：增强 `walk_forward_var` 返回的结果可经 `build_risk_validation_report` 生成 `RiskValidationReport` JSON 台账，逐条保留 out-of-sample forecast、realized return、exception、refit 参数、输入 digest 与 backtest digest；能力清单明确标注为 experimental，Basel traffic-light 字段只作参考，不宣称监管批准。
- 现金流绩效：增强层已新增现金流调整 TWR，明确时区、期初/期末、逐笔交易事件台账、净/毛费用、报告币种和全索引 FX；终值为零的总损失被保留，歧义、溢出和不可表示的回报会 fail closed。该 API 已注册到 `OperationCatalog` 并写入 0.4.0.dev0 公共 API 快照，未改变 strict Empyrical/Pyfolio 行为。
- 绩效报告披露：`create_strategy_report`、HTML 和 PDF 渲染现在传播同一份结构化 `DisclosureContext`；普通周期收益默认明确为“调用者提供、未做现金流调整”，不伪称 TWR。报告显示计算口径、return type、单位、频率、样本期、数据质量、费用、现金流、benchmark、risk-free 和 annualization；可选 manifest 记录经脱敏的已解析披露而不复制原始输入数值。HTML 对调用方的披露文本进行转义。
- 运行与性能：NumPy drawdown 后端保持 canonical 语义；新增多尺度、语义 digest 驱动的 workload profile；有效前沿对不可行约束、协方差和残差 fail closed。
- 打包与发布前检查：候选 wheel/sdist 通过 metadata、内容、发布一致性和 5 种隔离 consumer profile；最低/最新依赖矩阵实际安装候选 wheel 并记录版本。
- 工程门禁：工作流结构检查、公开 API snapshot、Ruff、Mypy、严格 MkDocs、性能预算与全部基准套件均已执行。工作流不会在本记录中被视为远端分支保护或 PyPI 环境审批的替代品。

## 已通过的本地验收证据

| 检查 | 结果 |
| --- | --- |
| `pytest tests -q --ignore=tests/benchmarks` | 5,241 passed, 22 skipped, 99 warnings，844.92s |
| `pytest tests/benchmarks -q -n 0` | 89 passed, 2 warnings，53.20s |
| Ruff check / format | 通过；670 个文件已格式化 |
| `mypy fincore --ignore-missing-imports` | 150 个源文件无 error；有 4 个未注解函数体提示 |
| `mkdocs build --strict` | 通过（Material 另给出 MkDocs 迁移警告） |
| workflow / public API / performance gates | 通过；dispatch p95 300.0 us，DAG p95 4.5 us |
| 当前构建的 wheel/sdist | `twine check`、release consistency、candidate consistency 通过 |
| 隔离 wheel consumers | `core`、`factor-analysis`、`alphalens`、`alphalens-pyfolio`、`all`：5/5 通过 |
| 候选依赖矩阵 | minimum / latest 两条隔离安装均通过，实际版本日志已由 `check_dependency_matrix.py` 输出 |
| 现金流绩效切片 | `tests/numerical/test_cashflow_performance.py --cov=fincore.performance.cashflows --cov-branch`：14 passed、1 warning；`cashflows.py` branch coverage 77% |
| Task 8 报告披露补充（2026-08-22） | 报告/API/capability/候选源码入口 focused 集合：115 passed、1 warning；含 Task 8 数值/属性/strict 兼容/报告/候选源码入口的完整集合：839 passed、5 warnings；Ruff、Mypy、`mkdocs build --strict` 通过（MkDocs/Material 给出上游迁移警告） |
| Task 9 walk-forward 风险报告补充（2026-08-22） | 风险数值/属性/`tests/test_risk`、新报告契约、能力/API snapshot 与文档示例：222 passed、1 warning；覆盖 event/refit/backtest digest、可变输入防御重验、DST 与 dateutil 时区、索引元数据重放；Ruff、Mypy、`mkdocs build --strict`、capability/API snapshot check 通过（MkDocs/Material 给出上游迁移警告） |
| Task 0 EVT Hill 补充（2026-08-22） | 独立 NumPy threshold-Hill oracle、上下尾反射和阈值边界，加上现有 EVT/GARCH 风险回归：153 passed、1 warning；Ruff、Mypy、`mkdocs build --strict` 通过（MkDocs/Material 给出上游迁移警告）。这只关闭 Hill 公式切片，不替代剩余 EVT/GARCH 的 out-of-sample 验收。 |
| Task 0 GARCH/EGARCH 补充（2026-08-22） | 独立条件标准化 EGARCH recursion oracle、溢出候选有限 penalty、EGARCH 的零/非有限/溢出方差输入 fail-closed、GARCH/GJR/EGARCH 平稳性 fail-closed 与增强适配器失败状态：161 passed、1 warning；完整 `tests/test_risk`：138 passed、1 warning；Ruff、Mypy、`mkdocs build --strict` 通过（MkDocs/Material 给出上游迁移警告）。这不替代 GARCH/EVT 的 out-of-sample calibration、残差或参数不确定性验收。 |
| Task 0 EVT GPD/GEV 补充（2026-08-22） | 独立 GPD sample-L-moment/PWM oracle、GEV PDF quadrature ES oracle（含 bounded/Gumbel/heavy-tail 三种 `xi`）、POT 概率域与最小浮点 `alpha` 的 Gumbel 稳定性反例：风险数值/属性/验证/`tests/test_risk` 集合 229 passed、1 warning；Ruff、Mypy、`mkdocs build --strict` 通过（MkDocs/Material 给出上游迁移警告）。这只关闭公式、数值稳定性和阈值域切片，不替代 EVT out-of-sample calibration 或阈值不确定性验收。 |
| Task 10 Fama–MacBeth 对齐补充（2026-08-22） | 单行静态暴露会按 return dates 广播，时变暴露按资产标签而非输入列位置对齐；新增独立 statsmodels OLS cross-sectional oracle、静态截面和乱序资产 adversarial fixtures。`tests/numerical/test_factor_inference.py tests/test_factor_analysis tests/compat/alphalens`：1,076 passed、61 warnings；Ruff、Mypy 通过。该修复只收敛既有 Fama–MacBeth 例程，不构成 PIT、FDR、HAC/cluster SE、成本或容量工作流的完成证据。 |
| Task 10 Benjamini–Hochberg FDR 补充（2026-08-22） | 新增可审计的独立推断函数：保留唯一 factor label、返回原始 p 值、BH adjusted p-value 与 step-up 决策；无效概率、重复标签及无效 `alpha` fail closed，空输入返回显式空结果。它与 `statsmodels.stats.multitest.multipletests(method="fdr_bh")` 的乱序/tie fixture 一致。`tests/numerical/test_factor_inference.py tests/test_factor_analysis tests/compat/alphalens`：1,085 passed、61 warnings；Ruff、Mypy 通过。该函数尚未接入所有 enhanced factor 报告或研究者 trials 记录。 |
| Task 10 IC inference 边界补充（2026-08-22） | IC mean/t-stat/interval 现在丢弃 `NaN`、拒绝 infinite observation，零均值常数样本返回 `0.0`（不再伪造 `-inf`）；`z` 必须为有限正数。普通样本 t-stat 与 `scipy.stats.ttest_1samp` 一致。`tests/numerical/test_factor_inference.py tests/test_factor_analysis tests/compat/alphalens`：1,087 passed、61 warnings；Ruff、Mypy 通过。该 i.i.d. helper 不构成 HAC/cluster inference 或 factor workflow 的完成证据。 |
| Task 10 PIT 因果物化补充（2026-08-22） | enhanced 层新增 `materialize_pit_factor` 与 `prepare_pit_factor_data`：账本强制 `as_of <= known_at <= effective_from`、统一时区、有限值及显式 universe；每个评估日只选择当时已知且生效的最新修订，`in_universe=False` 会撤出资产，空可用集明确 fail closed；包装入口拒绝全样本 `filter_zscore`，strict Alphalens 未改。手写 timeline oracle 和未来数据扰动 fixture 覆盖“不影响此前因子值”；machine-readable capability inventory 将它标记为 experimental。`tests/numerical/test_factor_inference.py tests/numerical/test_factor_pit_materialization.py tests/numerical/test_optimization_feasibility.py tests/test_factor_analysis tests/test_attribution tests/test_optimization tests/compat/alphalens`：1,398 passed、61 warnings，306.91s；capability/renderer focused：7 passed、1 warning；public API snapshot、Ruff、Mypy、`mkdocs build --strict` 通过（MkDocs/Material 给出上游迁移警告）。这是增强数据准备路径，不替代 corporate-action/calendar provenance、研究 trial 记录、成本、borrow、容量或所有 factor workflow 的完成证据。 |

这些命令均在 `/Users/yunjinqi/opt/anaconda3` 的 `base` 环境中执行；候选构建目录为临时目录，未上传、未发布。

## 阻断项与失败门禁

1. **当前质量快照仍未通过覆盖率门禁。** 在 `f8174ae` 修复每轮复用同一个 disposable copy 的交叉污染后，重新收集的 [current-baseline.json](current-baseline.json) 显示 trusted、serial、single-process、xdist 与 branch-coverage 五轮均以 0 退出，副本完整性均为真，且非串行计数一致；此前 branch-coverage 轮的 17 个打包测试错误已消失。真实 branch-coverage 仍为 **45.0%**，低于 60% 下限。随后新增的提交使该快照的 `source.commit` 也不再匹配当前 HEAD；在新的完整收集达到门槛前，`check_quality_snapshot.py` 必须继续 fail closed。旧的“97%”快照不再可作为当前代码证据。
2. **发布级许可证门禁失败。** `check_notices.py --require-approved` 对 `empyrical`、`pyfolio`、`alphalens` 三项均 fail closed；这需要具名人工/法律审批，不能由测试代替。
3. **计划中的核心工作流仍未完成验收。** 现金流语义已接入增强报告与 manifest，但这不把它升级为完整的多币种报告工作流或 GIPS 认证；`fincore/risk/report.py` 已为现有 one-step walk-forward VaR 提供可复现的 reference 报告，却不覆盖 GARCH/EVT 的完整 out-of-sample calibration、convergence/residual/parameter-uncertainty 诊断或监管认证。PIT 因果物化已接入 enhanced 数据准备入口，但 corporate-action/calendar provenance、研究 trial 记录及完整 factor workflow 仍未验收；FDR 也尚未接入所有 enhanced 报告或 trials。`fincore/factor_analysis/costs.py` 与 `fincore/factor_analysis/capacity.py` 仍不存在，因此成本容量和其余 T9–T10 验收不能据此视为完成。
4. **公开类型与发布证明未完成。** 仓库无 `.pyi`，且 `scripts/check_public_typing.py`、`scripts/verify_attestation.py` 不存在；没有 pyright/stubtest installed-wheel 证明、SBOM/provenance/attestation 的本地验收。
5. **远端治理不能由本地 checkout 验收。** 受保护分支、required checks、PyPI environment reviewer、Actions SHA pin 和实际发布 provenance 需要仓库管理员在远端完成并提供当前证据。

## 后续收口顺序

1. 已修复质量基线收集中的打包测试错误；下一步为新增/低覆盖模块补测试。不得降低 60% branch-coverage 门槛或用旧快照、`--skip-commit-check` 充当发布证据。
2. 完成三方许可的人工审阅并让 release profile 继续 fail closed。
3. 扩展并独立验收剩余风险模型（GARCH/EVT 的 out-of-sample calibration、convergence/residual/parameter-uncertainty）、PIT 的 corporate-action/calendar provenance 与完整 enhanced workflow 整合、FDR 的报告/trials 记录、交易成本、容量和优化 KKT/残差等 T9–T10 项；每项需 oracle、property 与 adversarial fixture。绩效报告后续若要支持完整多币种台账，必须以带估值和 FX provenance 的端到端工作流验收，不能把 `DisclosureContext` 当作计算证据。
4. 交付 `.pyi` / pyright / stubtest consumer gate、SBOM、provenance/attestation 检查，再由管理员完成远端分支与发布环境治理。
5. 只有全部自动与人工 gate 为绿，才生成 1.0 readiness seal；本轮明确没有 merge、push、tag 或 publish。

## 可复跑命令

```bash
/Users/yunjinqi/opt/anaconda3/bin/conda run -n base python -m pytest -o addopts='' tests -q --tb=short --maxfail=0 --ignore=tests/benchmarks
/Users/yunjinqi/opt/anaconda3/bin/conda run -n base python -m pytest -o addopts='' tests/benchmarks -q --tb=short --maxfail=0 -n 0
/Users/yunjinqi/opt/anaconda3/bin/conda run -n base python scripts/check_notices.py --require-approved
/Users/yunjinqi/opt/anaconda3/bin/conda run -n base python scripts/check_quality_snapshot.py --snapshot docs/quality/current-baseline.json
/Users/yunjinqi/opt/anaconda3/bin/conda run -n base python -m pytest -o addopts='' tests/numerical/test_performance_semantics.py tests/numerical/test_cashflow_performance.py tests/oracles/performance tests/property tests/compat/empyrical tests/compat/pyfolio tests/contracts/test_operation_catalog.py tests/contracts/test_public_api_snapshot.py tests/contracts/test_unified_invocation.py tests/docs/test_examples.py -q --tb=short --maxfail=0
/Users/yunjinqi/opt/anaconda3/bin/conda run -n base python -m pytest -o addopts='' tests/test_report tests/contracts/test_disclosure_context_contract.py tests/contracts/test_public_api_snapshot.py tests/contracts/test_capabilities.py tests/quality/test_render_capability_inventory.py tests/quality/test_snapshot_public_api_source.py tests/quality/test_check_performance_source.py -q --tb=short --maxfail=0
/Users/yunjinqi/opt/anaconda3/bin/conda run -n base python -m pytest -o addopts='' tests/numerical/test_cashflow_performance.py --cov=fincore.performance.cashflows --cov-branch --cov-report=term-missing -q --tb=short --maxfail=0
/Users/yunjinqi/opt/anaconda3/bin/conda run -n base python -m pytest -o addopts='' tests/numerical/test_risk_model_validation.py tests/numerical/test_risk_validation_report.py tests/property/test_risk_model_properties.py tests/test_risk tests/contracts/test_capabilities.py tests/contracts/test_public_api_snapshot.py tests/docs/test_examples.py -q --tb=short --maxfail=0
/Users/yunjinqi/opt/anaconda3/bin/conda run -n base python -m pytest -o addopts='' tests/numerical/test_factor_inference.py tests/numerical/test_factor_pit_materialization.py tests/numerical/test_optimization_feasibility.py tests/test_factor_analysis tests/test_attribution tests/test_optimization tests/compat/alphalens -q --tb=short --maxfail=0
/Users/yunjinqi/opt/anaconda3/bin/conda run -n base python scripts/render_capability_inventory.py --check
/Users/yunjinqi/opt/anaconda3/bin/conda run -n base python scripts/snapshot_public_api.py --check
```
