# 0042 Unified Analytics Platform — 实施与验收记录

日期：2026-08-21  
验收对象：`codex/fincore-iteration0042-completion`，数值修复与验收提交至 `93fbf54d9a03b9f14b37bfb25ce5c3b821ef5710`。  
结论：**BLOCKED — 不可宣称计划完成、不可作为 1.0 或发布候选。**

本记录区分“本轮已验证的代码切片”与“0042 计划的全部完成定义”。局部通过的测试、候选 wheel 或性能数字不替代法律、供应链、治理、完整类型和所有金融工作流的验收。

## 本轮已落地并验证的切片

- 数值安全：修正 Brinson 多期 Carino linking 的累计绝对主动收益对账，并增加近损失边界和独立 Decimal oracle；XIRR/MWR 明确日历日、标签对齐及非传统现金流 fail-closed 策略；walk-forward historical VaR 使用有限样本 Weibull 分位数并复核回测诊断。
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

这些命令均在 `/Users/yunjinqi/opt/anaconda3` 的 `base` 环境中执行；候选构建目录为临时目录，未上传、未发布。

## 阻断项与失败门禁

1. **当前质量快照失败。** 重新收集的 [current-baseline.json](current-baseline.json) 绑定干净源码提交 `93fbf54`，但 branch-coverage 为 **45.0%**，低于 60% 下限；branch-coverage 轮还出现 17 个打包测试错误。其余 trusted、serial、single-process 和 xdist 运行均通过且计数一致。旧的“97%”快照不再可作为当前代码证据。
2. **发布级许可证门禁失败。** `check_notices.py --require-approved` 对 `empyrical`、`pyfolio`、`alphalens` 三项均 fail closed；这需要具名人工/法律审批，不能由测试代替。
3. **计划中的核心工作流仍缺失或未完成验收。** `fincore/performance/cashflows.py`、`fincore/risk/report.py`、`fincore/factor_analysis/costs.py` 与 `fincore/factor_analysis/capacity.py` 不存在；PIT/FDR、成本容量、现金流多币种报告、监管风险报告等 T8–T10 验收不能据此视为完成。
4. **公开类型与发布证明未完成。** 仓库无 `.pyi`，且 `scripts/check_public_typing.py`、`scripts/verify_attestation.py` 不存在；没有 pyright/stubtest installed-wheel 证明、SBOM/provenance/attestation 的本地验收。
5. **远端治理不能由本地 checkout 验收。** 受保护分支、required checks、PyPI environment reviewer、Actions SHA pin 和实际发布 provenance 需要仓库管理员在远端完成并提供当前证据。

## 后续收口顺序

1. 先修复质量基线收集中的打包测试错误，并为新增/低覆盖模块补测试；不得降低 60% branch-coverage 门槛或用旧快照、`--skip-commit-check` 充当发布证据。
2. 完成三方许可的人工审阅并让 release profile 继续 fail closed。
3. 实施并独立验收现金流、风险报告、PIT/FDR、交易成本、容量和优化 KKT/残差等 T8–T10 项；每项需 oracle、property 与 adversarial fixture。
4. 交付 `.pyi` / pyright / stubtest consumer gate、SBOM、provenance/attestation 检查，再由管理员完成远端分支与发布环境治理。
5. 只有全部自动与人工 gate 为绿，才生成 1.0 readiness seal；本轮明确没有 merge、push、tag 或 publish。

## 可复跑命令

```bash
/Users/yunjinqi/opt/anaconda3/bin/conda run -n base python -m pytest -o addopts='' tests -q --tb=short --maxfail=0 --ignore=tests/benchmarks
/Users/yunjinqi/opt/anaconda3/bin/conda run -n base python -m pytest -o addopts='' tests/benchmarks -q --tb=short --maxfail=0 -n 0
/Users/yunjinqi/opt/anaconda3/bin/conda run -n base python scripts/check_notices.py --require-approved
/Users/yunjinqi/opt/anaconda3/bin/conda run -n base python scripts/check_quality_snapshot.py --snapshot docs/quality/current-baseline.json
```
