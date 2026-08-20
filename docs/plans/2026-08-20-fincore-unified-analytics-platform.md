# Fincore Unified Analytics Platform Implementation Plan

> **For Codex:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 在保持 Empyrical、Pyfolio、Alphalens 严格兼容面的前提下，把 fincore 从“多套优秀功能的集合”收敛为具有统一金融语义、统一操作目录、统一调用协议、可独立验证数值正确性、可扩展执行后端和不可绕过发布证据链的一流金融分析平台。

**Architecture:** 保留“冻结兼容 façade + 增强 canonical kernel”的分层，只统一增强层的操作元数据、输入语义、验证、执行编排、结果与 provenance；通过不可变 `OperationCatalog` 生成现有 registry 的只读投影，以 `AnalysisSnapshot + DAG` 共享中间计算，并以 profile adapter 隔离 strict 与 enhanced 行为。

**Tech Stack:** Python 3.11+（支持窗口由本迭代重新决策）、NumPy、pandas、SciPy、statsmodels（测试 oracle / 可选能力）、pytest、Hypothesis、Ruff、mypy、pyright/stubtest、MkDocs、GitHub Actions、PyPA Trusted Publishing、SPDX/CycloneDX/SLSA provenance。

---

## 0. 文档状态与范围

| 项目 | 内容 |
| --- | --- |
| 迭代编号 | 0042（建议） |
| 审计日期 | 2026-08-20 |
| 审计基线 | `dev@41edb33` |
| 前置计划 | `docs/plans/2026-08-17-fincore-platform-excellence.md` |
| 当前阶段 | Proposal / 待批准执行 |
| 目标版本 | 先由 Task 1 决定；建议进入 `0.4.0.dev0`，而不是继续复用已发布的 `0.3.0` |

本计划不是上一轮“能力补齐、覆盖率补齐、基准基建”任务的重复。它针对当前实现已经暴露出的三个新问题：

1. 多套 registry、validation、result model 和执行入口仍各自演进；
2. 部分风险、模拟和归因方法虽然测试为绿，但存在会改变金融结论的数值错误；
3. CI、质量快照、许可证审批与发布构件没有形成同一候选、不可绕过的证据闭环。

本迭代不授权自动合并、打 tag、发布或推送。每个任务的 commit 文案仅用于实施者拆分提交。

## 1. 审计结论

### 1.1 当前值得保留的基础

- strict façade 与 enhanced kernel 已有清晰边界意识，Empyrical、Pyfolio、Alphalens 都有冻结签名或兼容 manifest。
- `METRIC_REGISTRY` 已覆盖 237 个 surface entry；`AnalysisContext` 有快照与缓存；`RollingEngine` 已共享 rolling moments。
- factor analysis 已形成 prepare → analyze → render、compute-once model 的正确方向；report 已形成 compute-once/render-many 与 audit manifest。
- 插件 registry 已有 `RLock`、scope、duplicate policy 和 isolation rollback；可在此基础上扩展，不应推倒重写。
- 当前审计运行的 415 个 contracts/plugin/context/risk/quality/package 聚焦测试全部通过；Ruff、mypy、compileall 通过。
- 跨平台矩阵、wheel consumer、细粒度 extras、MkDocs strict、README 示例测试、OIDC Trusted Publishing 和构件内容检查均已存在。

### 1.2 当前不能支持“世界一流”声明的阻断项

| 优先级 | 发现 | 可复现证据 | 风险 |
| --- | --- | --- | --- |
| P0 | Kupiec LR-POF 符号与边界错误 | `kupiec_lr(100, 5, 0.99) == -8.2582`；LR 理论上必须非负 | 错误 VaR 可能得到 `p=1`，模型校准被误判通过 |
| P0 | GARCH ES 实际等于 VaR | 固定 seed 下 VaR 与 ES 同为 `-0.0442422740`；`horizon` 未进入数值计算 | 风险资本、限额与压力结论错误 |
| P0 | GARCH/EGARCH/GJR 名称与实现不一致 | 接受任意 `(p,q)`，实现固定 1,1；EGARCH/GJR 共用普通 GARCH forecast | 用户以为使用了未实际实现的模型 |
| P0 | EVT tail/shape/Hill 语义存在偏差 | upper/lower tail 可得到相同结果；SciPy GEV shape 与标准 `xi` 符号未转换 | 尾部风险方向和量级错误 |
| P0 | GBM 参数二次缩放，Monte Carlo 参数被忽略 | 20% 年波动率的一年终值 log 波动仅约 1.26%；同 seed 下不同 drift/volatility 输出相同 | 模拟分布和置信区间错误 |
| P0 | Fama-French HAC/WLS 错误 | 四个 HAC 标准误完全相同；与 statsmodels oracle 最大相对误差约 94.16%；WLS 与 OLS 同路径 | 显著性、alpha 与暴露结论错误 |
| P0 | 多期 Brinson linking 与 style 指标错误 | 效果算术求和而组合收益几何复利；momentum 可恒为 0，beta 使用相关系数而非斜率 | 归因无法 reconciliation |
| P0 | 正式质量证据已过期 | `check_quality_snapshot.py` 对当前 HEAD 报 `source.commit` 不匹配，正式 branch coverage 仍为 55%，低于 60% 门槛 | 新增测试尚未成为当前提交的发布证据 |
| P0 | CI/发布图可被绕过 | `.github/workflows/ci.yml` 两次定义 `integration-offline`；release build 未依赖全部质量 job；publish 重新构建 | 绿色 badge 不等于发布候选通过全部门禁 |
| P0 | 许可证阻断只有文档声明 | 三个 adapted component 均 pending；checker 接受 pending，publish 不运行 approval gate | 发布合规结论不可证明 |
| P0 | 版本身份漂移 | PyPI 已发布 0.3.0，而源码、README、CHANGELOG 仍以 0.3.0 Beta/unreleased RC 演进 | 同一版本号无法唯一映射源码与构件 |
| P1 | 多套能力目录并存 | 237 metric entry、11 Pyfolio workflow、61 Alphalens function、7 Alphalens workflow、19 capability 各自维护 | API、docs、stability、types 和依赖易漂移 |
| P1 | 增强验证不统一 | `fincore/validation.py`、`contracts/validation.py`、risk/simulation 私有 validator 并存；plugin 绕过 dispatch | 相同错误在不同入口产生不同结果/异常 |
| P1 | 结果与资源生命周期不统一 | scalar/dict/dataclass/model 混用；ReportArtifacts 有 `close()`，factor artifacts 依赖外部关闭函数 | 序列化、审计、资源释放和下游组合困难 |
| P1 | `py.typed` 与真实类型体验不匹配 | 动态 `__getattr__ -> Any`、动态方法、无 `.pyi`，部分模块关闭 untyped body 检查 | IDE、mypy/pyright 消费者得不到可靠契约 |
| P1 | “minimum/latest”矩阵不是隔离安装证明 | 两个 constraints 都使用 `>=`；脚本只在当前解释器 import | 声明最低依赖实际可能不可安装或不可运行 |

结论：fincore 已经不是简单的历史项目拼接，但当前首先是“金融正确性与证据链”问题，其次才是 API 统一和速度问题。P0 数值错误全部修复、独立 oracle 通过之前，不应提高相关 capability 的稳定性，也不应发布 1.0。

## 2. 对标标准与目标边界

### 2.1 本迭代采用的外部基准

- [GIPS Standards](https://www.gipsstandards.org/standards/)：用于定义收益计算、记录、披露和可复核性目标。fincore 只能声明“提供 GIPS-aware 计算与披露支持”，不能由软件自行宣称用户或报告符合 GIPS。
- [Basel Committee market-risk standard](https://www.bis.org/bcbs/publ/d457.htm)：用于 VaR/ES、回测区间、异常计数和模型验证的参考实现；不宣称监管认证。
- [Scientific Python SPEC 0](https://scientific-python.org/specs/spec-0000/)：对 Python 和核心依赖支持窗口做显式决策；若为机构用户保留更老版本，必须记录偏离理由并真实测试。
- [Scientific Python SPEC 1](https://scientific-python.org/specs/spec-0001/)：规范 lazy import、显式导出、eager-import 测试与类型信息。
- [Scientific Python SPEC 7](https://scientific-python.org/specs/spec-0007/)：统一随机数参数为 `rng` 语义，并为旧 `seed/random_state` 提供迁移期。
- [Scientific Python SPEC 8](https://scientific-python.org/specs/spec-0008/)：固定 Actions 完整 SHA、最小权限、受保护发布环境、可信发布和 provenance。
- [Python Array API standard](https://data-apis.org/array-api/latest/API_specification/index.html)：仅用于适合 dense array 的纯数值 kernel；带标签、时区和交易日历的语义仍由 pandas reference backend 负责。
- [PyPA attestations](https://packaging.python.org/en/latest/specifications/index-hosted-attestations/) 与 [wheel SBOM 规范](https://packaging.python.org/en/latest/specifications/binary-distribution-format/)：构建同一候选、生成 SBOM/provenance 并允许消费者验证。
- [OpenSSF OSPS Baseline](https://baseline.openssf.org/versions/2026-02-19.html) 与 [SLSA 1.2](https://slsa.dev/spec/v1.2/)：作为仓库治理和供应链验收清单。

### 2.2 同类项目借鉴点

| 项目 | 借鉴点 | fincore 的差异化目标 |
| --- | --- | --- |
| [empyrical-reloaded](https://github.com/stefan-jansen/empyrical-reloaded) | 轻量、熟悉的指标函数 API | 保留兼容，同时提供版本化增强语义与审计元数据 |
| [alphalens-reloaded](https://github.com/stefan-jansen/alphalens-reloaded) | IC、returns、turnover、group analysis | 增加 causal/PIT、统计推断、多重检验、成本与容量 |
| [QuantStats](https://github.com/ranaroussi/quantstats) | 易用报告、stats/plots/reports 分层 | 以统一计算模型、严格验证和可复核报告形成更强可信度 |
| [Riskfolio-Lib](https://github.com/dcajasn/Riskfolio-Lib) | 丰富优化与风险度量 | 补齐 solver diagnostics、约束 residual、风险模型验证 |
| [vectorbt](https://github.com/polakowo/vectorbt) | 大规模向量化与批量计算 | 用 DAG、共享中间量与可选 backend 获得速度，同时保留标签语义 |
| [PerformanceAnalytics](https://github.com/cran/PerformanceAnalytics) | 完整绩效/风险口径与成熟 oracle | 建立跨语言 fixture 和统计定义对照，而非只与自身旧实现比较 |

## 3. 目标架构

```text
Public API
├── strict compatibility: fincore.empyrical / pyfolio / alphalens
├── enhanced domains: metrics / risk / factor_analysis / attribution / report
└── extension API: plugins and explicit entry-point discovery
                         │
                         ▼
              Immutable OperationCatalog
  OperationDefinition: canonical semantics · contracts · kernel · provenance
  PublicBinding: public path · profile · signature · adapter · projection
                         │
                         ▼
              Unified Invocation Pipeline
  bind → normalize/validate → align → execute → project → diagnose/provenance
             │                           │
             │                           └── strict/enhanced profile adapters
             ▼
        AnalysisSnapshot + computation DAG
  shared alignment · cumulative returns · drawdowns · moments · factor panels
             │
             ▼
          Pure domain kernels / optional numeric backends
             │
             ▼
   typed values + ResultMetadata + ArtifactBundle + versioned serialization
```

### 必须保持的架构不变量

1. strict Empyrical/Pyfolio/Alphalens 的签名、返回形状、异常、NaN、stdout、show/close 等可观察语义继续由冻结 oracle 决定。
2. strict profile 不经过 enhanced validation；共享的是 raw kernel 和编排协议，不是行为策略。
3. `fincore` 不安装或伪装为顶层 `empyrical`、`pyfolio`、`alphalens` 包。
4. 增强 API 的 canonical name、输入单位、频率、符号和结果 schema 必须版本化。
5. 每个 stable 金融算法至少有一个独立 oracle；自身旧实现不能作为唯一正确性证据。
6. 性能优化必须先通过语义 digest/tolerance，再比较 wall time 和 RSS；不得以降低精度、删除边界检查换速度。
7. 发布只允许使用通过所有门禁的同一候选构件；release workflow 不重新构建另一份 wheel。
8. strict façade 的实现路径不得构造 enhanced stateful class，也不得进入 enhanced validator；共享 kernel 必须位于二者之下。

## 4. 任务依赖与团队拆分

```text
T0 数值安全闸 ──> T2 语义 ADR ──> T3 Catalog ─┬─> T4 输入/异常协议
       │                                       └─> T5 结果/资源协议
       └──────────────────────────────────────────> T4/T5 domain handoff
                                                        │
                                              T6A strict adapter isolation
                                                        │
                                              T6B invocation/plugin overlay
                                                        │
                                              T7 Snapshot/DAG/API codegen
                                             ┌──────────┼──────────┐
                                             T8         T9         T10
                                           绩效标准化  风险验证  因子/归因/优化
                                             └──────────┼──────────┘
                                                        T11 性能后端

T1 发布与证据真相 ───────────────────────────────────────┐
T0–T11 全部 gates ──────────────────────────────────────┴─> T12 1.0 seal
```

建议所有权：

- Quant correctness A：T0 的 risk/EVT/GARCH；Quant correctness B：simulation/attribution/optimization。
- Platform architecture：T2–T7，拥有 catalog、contracts、dispatch、results、DAG。
- Performance engineering：T11，只有在 T0 与相应 domain oracle 通过后才改 kernel。
- Release engineering：T1、T12 的 CI、packaging、supply chain、typing consumer。
- Documentation/community：T2 ADR、T8–T10 方法文档、T12 adoption/governance。

任何并行开发者都必须知道自己不是仓库唯一修改者，不得回滚其他任务的提交；共享文件（`pyproject.toml`、`fincore/__init__.py`、CI）由 release/platform owner 串行落地。T0 先拥有 risk/simulation/attribution 的数值 kernel 并提交 oracle+修复；T4/T5 对相同 domain 文件的 contract/result tranche 只能在该 T0 commit 之后开始，且 T0 commit 必须是其祖先。任何可并行任务不得同时拥有同一文件。

## 5. 实施任务

### Task 0：建立 P0 数值安全闸并修复会改变金融结论的错误

**Owner:** Quant correctness A + B  
**Depends on:** 无  
**Blocks:** T3 之后的稳定性提升、T8–T12、任何新版本发布

**Files:**

- Modify: `fincore/risk/backtesting.py`
- Modify: `fincore/risk/models.py`
- Modify: `fincore/risk/garch.py`
- Modify: `fincore/risk/evt.py`
- Modify: `fincore/simulation/base.py`
- Modify: `fincore/simulation/paths.py`
- Modify: `fincore/simulation/monte_carlo.py`
- Modify: `fincore/attribution/fama_french.py`
- Modify: `fincore/attribution/style.py`
- Modify: `fincore/attribution/brinson.py`
- Modify: `fincore/metrics/ratios.py`
- Modify: `fincore/capabilities.py`
- Create: `tests/oracles/risk/`
- Create: `tests/oracles/simulation/`
- Create: `tests/oracles/attribution/`
- Create: `tests/numerical/test_risk_reference_oracles.py`
- Create: `tests/numerical/test_simulation_reference_oracles.py`
- Create: `tests/numerical/test_attribution_reference_oracles.py`
- Create: `docs/quality/numerical-oracle-register.md`

**Steps:**

1. 先写失败测试，冻结本次审计复现的错误，不先改实现。
2. 用稳定 log-likelihood/`xlogy` 公式实现 Kupiec LR-POF；`x=0`、`x=n` 使用连续极限，LR 永远非负。Christoffersen 保留现有能力，但加入独立 oracle。
3. 将 VaR/ES 变为一致的 forecast pair：明确 distribution、sign convention、horizon aggregation/recursive forecast。若某模型没有正确 ES，在 T2/T5 的判别式结果契约落地前只能降级 capability 或抛出已批准的现有异常；不得提前改变 direct scalar 返回形状，也禁止把 VaR 冒充 ES。
4. GARCH 系列只暴露真实支持的阶数和模型；检查优化收敛、参数约束与平稳性；EGARCH/GJR 使用各自递归 forecast。完整实现前将 capability 降为 experimental。
5. 修复 EVT 上下尾选择、SciPy GEV shape 转换和 Hill estimator；增加 tail reflection、threshold stability、单调性和参数恢复测试。
6. 统一 simulation 参数单位，移除二次年化/反年化；`drift`、`volatility` 必须影响输出；antithetic 使用同一随机流的 `Z/-Z` 配对。
7. Fama-French 使用真正 OLS/WLS 与 Newey-West sandwich covariance；按时间标签对齐数据；公开返回类型与实际值一致。style beta 使用 `cov/var`，momentum 使用不同回看区间。Brinson 多期使用明确的 Carino/Menchero linking。
8. 修正 DSR 的普通/超额 kurtosis 和 trial Sharpe variance 定义，并保留旧行为的显式 legacy profile（如兼容面需要）。
9. 每个修复记录公式、来源、单位、边界、oracle 生成方式与 tolerance；fixture 必须包含 provenance，不能只保存神秘数字。

**Acceptance:**

- Kupiec `LR >= 0`，与手算及独立 R/Python oracle 误差 `<= 1e-12`；仅 Kupiec 一项即可拒绝严重失准的 VaR。
- 正常/Student-t/GARCH 的 ES 在 losses-negative 口径下满足 `ES <= VaR`，解析 oracle `rtol <= 1e-8`。
- horizon 1/5/10/20 的数值与 forecast shape 具有明确、可测试且不同的语义。
- GARCH/EGARCH/GJR 与 `arch` 生成的固定仿真 fixture 比较参数、log-likelihood、条件波动率和 forecast；未收敛不得返回 `status=ok`。
- GBM 一年终值的均值/方差进入解析公式 99% Monte Carlo CI；20% 年波动率估计相对误差 `<= 1%`。
- HAC/WLS 与 statsmodels oracle `rtol <= 1e-10`；多期 Brinson reconciliation residual `<= 1e-12`。
- 所有 P0 domain function 有至少一个 independent oracle、一个 property test 和一个错误模型反例。

**Commands:**

```bash
/Users/yunjinqi/opt/anaconda3/bin/conda run -n base python -m pytest -o addopts='' \
  tests/numerical tests/oracles tests/test_risk tests/test_simulation tests/test_attribution \
  tests/property/test_risk_invariants.py -q --tb=short --maxfail=0
/Users/yunjinqi/opt/anaconda3/bin/conda run -n base python -m ruff check fincore tests
/Users/yunjinqi/opt/anaconda3/bin/conda run -n base python -m mypy fincore --ignore-missing-imports
```

**Suggested commits:**

- `test: add independent financial correctness oracles`
- `fix: correct risk forecast and backtesting semantics`
- `fix: correct simulation and attribution numerical contracts`

### Task 1：恢复版本、质量快照、CI 与发布候选的事实一致性

**Owner:** Release engineering  
**Depends on:** 无；正式发布 seal 依赖 T0  
**Blocks:** 所有发布声明

**Files:**

- Modify: `pyproject.toml`
- Modify: `README.md`
- Modify: `CHANGELOG.md`
- Modify: `.github/workflows/ci.yml`
- Modify: `.github/workflows/publish.yml`
- Modify: `.github/workflows/test-priority.yml`
- Modify: `scripts/check_quality_snapshot.py`
- Modify: `scripts/check_dependency_matrix.py`
- Modify: `scripts/check_notices.py`
- Create: `scripts/check_workflow_integrity.py`
- Create: `scripts/check_release_candidate.py`
- Create: `tests/quality/test_workflow_integrity.py`
- Create: `tests/packaging/test_release_candidate.py`
- Regenerate only after a clean full run: `docs/quality/current-baseline.json`, `docs/quality/current-baseline.md`

**Steps:**

1. 记录 PyPI/tag/source 事实，选择下一个唯一开发版本；禁止已发布 `0.3.0` 继续承载新源码。
2. 删除重复 `integration-offline` key，统一 `dev/master` 分支政策；加入 YAML duplicate-key、schema 和 actionlint 门禁。
3. 新增 `release-gate` 汇总 job，明确依赖 test、compat、offline integration、property、coverage、perf、factor benchmark、extras、docs、security、license、wheel consumer、quality freshness。
4. CI 对一个 clean commit 只构建一次 wheel/sdist，保存 SHA256 与 release manifest；publish 只能下载这个 candidate，验证 digest 后发布，不得重新构建。
5. `check_notices.py` 增加 `--require-approved`；普通 PR 可检查 inventory 结构，release 必须所有 adapted component 为 approved、带 reviewer/date/decision。
6. `minimum.txt` 使用精确可安装的 oldest-resolvable 组合；在隔离环境真实安装 wheel、`pip check`、核对版本、运行 smoke。另建 newest 与 prerelease lane。
7. T0 与全量 gates 通过后，在 clean commit 上重建正式 quality snapshot；不允许 `--skip-commit-check` 作为 release 证据，不降低 60% 门槛。若新测试确实达到 90%+ branch coverage，再把绝对 floor 提到 90%，并继续保留 changed-lines 95%。

**Acceptance:**

- 所有 workflow mapping key 唯一；任一 release dependency 失败时 candidate job 与 publish job 都不可运行。
- 发布 artifact digest 与 CI candidate 完全一致，tag/commit/version/changelog/runtime/wheel metadata 一致。
- 当前 quality checker 对当前 clean HEAD 通过，branch coverage 有可重建证据；不再引用旧 `6cb26ab` 快照说明新 HEAD。
- 任一 pending/unresolved license record 在 release profile 下 fail closed。
- oldest/newest 测试日志包含实际 installed versions；修改 constraint 的负向测试会改变环境并触发验证。

**Commands:**

```bash
/Users/yunjinqi/opt/anaconda3/bin/conda run -n base python scripts/check_workflow_integrity.py
/Users/yunjinqi/opt/anaconda3/bin/conda run -n base python scripts/check_dependency_matrix.py
/Users/yunjinqi/opt/anaconda3/bin/conda run -n base python scripts/check_notices.py --require-approved
/Users/yunjinqi/opt/anaconda3/bin/conda run -n base python scripts/check_quality_snapshot.py \
  --snapshot docs/quality/current-baseline.json
```

**Suggested commits:**

- `fix: make release gate graph fail closed`
- `build: promote one verified release candidate artifact`
- `test: refresh current quality evidence on clean head`

### Task 2：批准统一金融语义 ADR，并冻结公共 API characterization

**Owner:** Platform architecture + Quant leads  
**Depends on:** T0 的问题清单可用  
**Blocks:** T3–T10

**Files:**

- Create: `docs/architecture/adr/0042-unified-operation-model.md`
- Create: `docs/architecture/financial-semantics.md`
- Create: `docs/architecture/public-surface-policy.md`
- Create: `docs/architecture/public-api-map.md`
- Create: `scripts/snapshot_public_api.py`
- Create: `tests/contracts/test_public_api_snapshot.py`
- Create: `tests/contracts/test_public_api_behavior_probes.py`
- Create: `tests/contracts/fixtures/public-api-0.3.x.json`
- Create: `tests/contracts/fixtures/public-api-probes-0.3.x.json`

**Steps:**

1. 将 characterization 拆为两层：静态扫描自动枚举 path、surface、profile、signature、optional dependency、stability 和 provenance；带版本的 probe registry 为每个 public callable 提供最小输入 fixture，实际记录 success shape/dtype、异常/NaN，以及适用的 stdout、show/close 行为。不得靠无输入反射猜测返回形状。
2. 为 enhanced 语义定义：simple/log return、price/return 边界、frequency/calendar、timezone、currency、benchmark、risk-free 单位、cashflow、fees、NaN/inf、duplicate/order、alignment、ddof、sign convention、weight timestamp、as-of/known-at。
3. 定义 profile：`strict_empyrical_0_6_0`、`strict_pyfolio_0_9_6`、`strict_alphalens_cloudquant_0_4_0`、`enhanced_v1`、`plugin_v1`。
4. 定义 pre-1.0 与 1.0 API policy、稳定级别、弃用窗口和 breaking-change 审批流程。
5. 在 ADR 中冻结 `Success / Unsupported / Failed` 判别式高层结果状态、wire schema 与 direct scalar projection；实现延后到 T5，T0 不得自行发明临时返回类型。
6. 先用 characterization tests 锁定现状；ADR 获批准前不得进行大规模 registry/dispatch 重构。

**Acceptance:**

- 每个 public export 恰好属于一个 operation/profile/stability；重复和未知归属均使测试失败。
- strict façade 的静态 snapshot 与行为 probes 覆盖完整 C0–C4：签名、成功路径、异常/NaN，以及适用的 stdout/show/close 均与冻结 manifest/oracle 一致。
- direct scalar 返回形状保持不变；`Success / Unsupported / Failed` 只通过批准的高层 envelope 暴露，并可按版本化 schema round trip。
- 中英文文档使用同一套 API surface 名称；不再出现标题写 Three、表格实际 Four 的漂移。
- ADR 明确哪些口径是兼容事实，哪些是增强层设计选择。

**Commands:**

```bash
/Users/yunjinqi/opt/anaconda3/bin/conda run -n base python scripts/snapshot_public_api.py --check
/Users/yunjinqi/opt/anaconda3/bin/conda run -n base python -m pytest -o addopts='' \
  tests/contracts/test_public_api_snapshot.py tests/contracts/test_public_api_behavior_probes.py \
  tests/compat -q --tb=short --maxfail=0
```

**Suggested commit:** `docs: approve unified financial semantics and public surface ADR`

### Task 3：建立唯一不可变 OperationCatalog

**Owner:** Platform architecture  
**Depends on:** T2  
**Blocks:** T4–T7

**Files:**

- Create: `fincore/api/__init__.py`
- Create: `fincore/api/specs.py`
- Create: `fincore/api/catalog.py`
- Create: `fincore/api/builtins.py`
- Modify: `fincore/_registry.py`
- Modify: `fincore/contracts/workflows.py`
- Modify: `fincore/contracts/factor_analysis.py`
- Modify: `fincore/contracts/factor_workflows.py`
- Modify: `fincore/capabilities.py`
- Modify: `scripts/render_capability_inventory.py`
- Create: `tests/contracts/test_operation_catalog.py`

**Catalog entities and minimum fields:**

```python
OperationDefinition(
    operation_id, semantic_profile, domain, canonical_name, aliases, stability,
    input_contract, output_contract, kernel_ref, optional_extra, deterministic,
    rng_policy, provenance, semantic_version,
)
PublicBinding(
    binding_id, operation_id, semantic_profile, public_path, surface, signature,
    adapter_ref, result_projection, typing_contract_ref, overloads,
    introduced_in, deprecated_in, remove_in,
)
```

**Steps:**

1. 实现 immutable catalog、唯一 key 和 lazy reference resolver，不导入可选重依赖；语义定义与公共绑定是两个实体，不能在每个 surface 复制 contract/provenance。
2. 将现有 237 metric surface entry 映射为 `PublicBinding`，并与 11 Pyfolio workflow、61 Alphalens function、7 Alphalens workflow 一起归并到较少的 `OperationDefinition`。
3. 现有 `METRIC_REGISTRY`/workflow specs 暂时保留为 catalog 生成的只读 compatibility view，避免一次性重写。
4. capability inventory、API map、docs table 和 deprecation map 全部从 catalog 生成；删除手工重复事实源。
5. canonical name 与 alias 分离，例如 `gross_leverage`、`turnover`、`sharpe_ratio`；旧名称在兼容窗口内继续工作。

**Acceptance:**

- 所有当前 public entry 100% 进入 catalog；每个 canonical `operation_id + semantic_profile` 恰有一个 `OperationDefinition`，每个 public path 恰有一个 `PublicBinding`。
- 同一逻辑操作在 strict/class/metrics 等 surface 之间只共享 definition，不重复 contract、kernel provenance 或稳定性事实。
- `result_contract_key` 不再只是无人消费的字符串：必须解析到真实 schema/projection 或删除。
- registry、`__all__`、capability inventory、docs 与 API snapshot drift gate 全绿。
- `import fincore` 不增加重依赖导入，strict C0–C4 零变化。

**Commands:**

```bash
/Users/yunjinqi/opt/anaconda3/bin/conda run -n base python -m pytest -o addopts='' \
  tests/contracts/test_operation_catalog.py tests/contracts/test_metric_surface_profiles.py \
  tests/compat -q --tb=short --maxfail=0
/Users/yunjinqi/opt/anaconda3/bin/conda run -n base python scripts/render_capability_inventory.py --check
```

**Suggested commit:** `refactor: make operation catalog the public semantic authority`

### Task 4：统一 enhanced 输入、时间序列、验证与异常协议

**Owner:** Platform architecture  
**Depends on:** T0 的相关 domain kernel/oracle commit、T3  
**Blocks:** T6–T10

**Files:**

- Create: `fincore/contracts/analysis.py`
- Create: `fincore/contracts/profiles.py`
- Modify: `fincore/contracts/validation.py`
- Modify: `fincore/contracts/time_series.py`
- Modify: `fincore/validation.py`
- Modify: `fincore/exceptions.py`
- Modify: `fincore/core/context.py`
- Modify: `fincore/risk/models.py`
- Modify: `fincore/simulation/base.py`
- Modify: `fincore/factor_analysis/exceptions.py`
- Modify: `fincore/factor_analysis/data.py`
- Modify: `fincore/factor_analysis/analysis.py`
- Create: `tests/contracts/test_enhanced_analysis_contract.py`
- Create: `tests/property/test_enhanced_contract_invariants.py`

**Steps:**

1. 定义 `AnalysisInput`/`SeriesSemantics`/`PortfolioSemantics`，数据与 metadata 分离；不强迫 strict API 使用新对象。
2. 统一 timezone 策略：naive 不再静默等同 UTC；必须由 profile 或调用者指定 localize/convert/reject。
3. 统一 alignment、missing、duplicate、finite、copy/mutation 与 empty/small-sample 策略，并在 diagnostics 中记录丢弃行与原因。
4. `fincore/validation.py` 逐步变为兼容 shim；risk、simulation、factor enhanced direct API 使用同一 contract pipeline，并迁移 factor data/analysis 当前裸 `TypeError`/`ValueError` 路径。
5. 建立 `FincoreError` 子类：`InputContractError`、`AlignmentError`、`NumericalConvergenceError`、`ResultContractError`、`ResourceLifecycleError`；错误携带 `operation_id/parameter/path/profile`。
6. strict profile 保留上游原生错误优先级和类型，不被 enhanced exception 包装污染。

**Acceptance:**

- 除 Python 自身 signature binding `TypeError` 外，所有 enhanced 公共输入错误可被 `except FincoreError` 捕获。
- 相同 builtin enhanced operation 通过 flat/module/class/context 调用时使用相同输入政策；plugin 入口矩阵在 T6B 完成后验收。
- caller input 不被修改；行丢弃、timezone conversion、calendar inference 均可审计。
- property tests 覆盖乱序、重复、DST、naive/aware、空交集、NaN/inf、simple/log、不同频率与不同标签类型。
- strict compatibility suites 全绿。

**Commands:**

```bash
/Users/yunjinqi/opt/anaconda3/bin/conda run -n base python -m pytest -o addopts='' \
  tests/contracts tests/property/test_time_series_contracts.py \
  tests/property/test_enhanced_contract_invariants.py tests/compat -q --tb=short --maxfail=0
```

**Suggested commit:** `refactor: unify enhanced input and error contracts`

### Task 5：统一结果 metadata、序列化和 ArtifactBundle 生命周期

**Owner:** Platform architecture  
**Depends on:** T0 的相关 domain kernel/oracle commit、T3  
**Blocks:** T6–T12

**Files:**

- Create: `fincore/results/__init__.py`
- Create: `fincore/results/base.py`
- Create: `fincore/results/artifacts.py`
- Create: `fincore/results/serialization.py`
- Modify: `fincore/risk/models.py`
- Modify: `fincore/risk/backtesting.py`
- Modify: `fincore/report/model.py`
- Modify: `fincore/report/artifacts.py`
- Modify: `fincore/factor_analysis/models.py`
- Modify: `fincore/factor_analysis/tears.py`
- Modify: `fincore/optimization/_utils.py`
- Create: `tests/contracts/test_result_protocol.py`
- Create: `tests/contracts/test_artifact_lifecycle.py`

**Steps:**

1. 按 T2 已批准的 `Success / Unsupported / Failed` 判别式契约定义可组合的 `ResultMetadata`：operation/profile/schema version、status、units、frequency、sign、input/config digest、software/dependency provenance、warnings、diagnostics、uncertainty。
2. 定义 `AnalysisResult[T]` 和 `ArtifactBundle` protocol；所有 renderer artifact 支持幂等 `close()` 与 context manager。
3. direct scalar API 在 0.3.x/0.4.x 保持原返回形状；统一 envelope 先用于 `execute()`、context/report/risk/factor/optimization 等高层 API，避免隐式 breaking change。
4. RiskEstimate、RiskBacktestResult、FactorAnalysisModel、ReportModel、OptimizationResult 组合公共 metadata，而不是各自重新发明字段。
5. 定义版本化 JSON schema 和向前兼容读取；NaN/inf、timezone、dtype、Index、货币与 calendar 必须有明确 wire policy。
6. 模型采用 copy-on-ingest + 显式 `copy(deep=True)`；移除“名义 frozen 但暴露可变 DataFrame”和“每次读取都 pickle 深复制”两个极端。

**Acceptance:**

- 所有 enhanced session/workflow result 可回答：算了什么、用什么语义、输入/配置 digest、何时算、状态和诊断是什么。
- direct scalar 形状保持冻结；错误 kernel 输出触发 `ResultContractError`，判别式状态与 wire schema 可 round trip。
- 所有 artifact 可用 `with ... as artifacts:`，close 幂等且异常安全；资源泄漏测试全绿。
- JSON round trip 保留 schema、timezone、dtype、index、metadata 与 result digest。
- 100k/1m 行 factor model 的构造、字段读取和复制有 wall/RSS benchmark；读取不再触发隐藏 pickle round-trip。

**Commands:**

```bash
/Users/yunjinqi/opt/anaconda3/bin/conda run -n base python -m pytest -o addopts='' \
  tests/contracts/test_result_protocol.py tests/contracts/test_artifact_lifecycle.py \
  tests/test_factor_analysis/test_models.py tests/test_report -q --tb=short --maxfail=0
```

**Suggested commit:** `feat: add versioned result metadata and artifact lifecycle`

### Task 6：隔离 strict adapter、统一调用 pipeline，并把插件变成原子 catalog overlay

**Owner:** Platform architecture  
**Depends on:** T3、T4、T5  
**Blocks:** T7–T12

**Files:**

- Create: `fincore/api/invoke.py`
- Create: `fincore/api/adapters.py`
- Create: `fincore/_compat/__init__.py`
- Create: `fincore/_compat/empyrical_adapter.py`
- Create: `fincore/_compat/pyfolio_adapter.py`
- Create: `fincore/_compat/alphalens_adapter.py`
- Modify: `fincore/_dispatch.py`
- Modify: `fincore/empyrical.py`
- Modify: `fincore/_pyfolio_impl.py`
- Modify: `fincore/pyfolio.py`
- Modify: `fincore/alphalens/__init__.py`
- Modify: `fincore/alphalens/performance.py`
- Modify: `fincore/alphalens/utils.py`
- Modify: `fincore/alphalens/plotting.py`
- Modify: `fincore/alphalens/tears.py`
- Modify: `fincore/contracts/workflows.py`
- Modify: `fincore/core/context.py`
- Modify: `fincore/core/engine.py`
- Modify: `fincore/plugin/specs.py`
- Modify: `fincore/plugin/registry.py`
- Create: `fincore/plugin/discovery.py`
- Modify: `pyproject.toml` only for documented sample entry-point groups
- Create: `tests/contracts/test_unified_invocation.py`
- Create: `tests/contracts/test_compat_adapter_isolation.py`
- Create: `tests/contracts/test_plugin_overlay_snapshot.py`
- Create: `tests/test_plugin/test_entrypoint_discovery.py`
- Create: `tests/fixtures/sample_plugin/`

**Steps:**

1. **T6A 先落地 strict adapter isolation：** 用 T2 的 C0–C4 probes 保护现状，再把 Empyrical、Pyfolio 和 Alphalens strict 执行移入 `_compat` adapter；strict Pyfolio 不再实例化 enhanced `Pyfolio(Empyrical)`，strict Alphalens tears 不再直接进入 enhanced `analyze_factor()`。
2. strict adapters 明确 bypass enhanced validation，并保留现有 raw-kernel recursion guard；通过 trace test 证明不构造 enhanced stateful class、不进入 enhanced validator，再逐步减少隐式 module mutation。
3. **T6B 再落地 enhanced pipeline：** bind → profile contract → normalize/align → raw kernel → result contract → projection → metadata/hooks；`AnalysisContext.compute()`、RollingEngine plugin、report extension 与 direct enhanced execution 全部进入该 pipeline。
4. 扩展 `PluginSpec`：operation id、API range、distribution/version、contract、deterministic、rng policy、requires、result projection、load diagnostics。
5. plugin overlay 使用不可变 snapshot、单调 generation 和原子替换；builtin catalog 默认不可被 `OVERWRITE` 静默替换，覆盖必须显式 `override_builtin=True`，strict operation 永远不可 shadow。
6. 增加显式 `discover_plugins()` 和 `fincore.metrics/providers/renderers/exporters` entry-point groups；`import fincore` 时不得自动执行第三方代码。

**Acceptance:**

- 同一 enhanced operation 经所有入口调用时，输入、异常、结果 value 和 metadata 一致。
- trace test 证明 strict 调用不构造 enhanced stateful class、不进入 enhanced validator；完整 C0–C4 全绿。
- 插件注册时验证 signature/result contract；错误包含 distribution 身份和 compatibility range。
- 安装独立 sample-plugin wheel 后可显式发现、调用、禁用；坏 import 不破坏核心包。
- 注册、覆盖、禁用会生成新的原子 overlay generation；并发读取看不到半更新，overlay snapshot 身份可进入 T7 cache key。
- strict Empyrical/Pyfolio/Alphalens 不可被插件 shadow，所有 compatibility oracle 全绿。
- dispatch overhead 有独立 benchmark，不能因统一 pipeline 无界增长。

**Commands:**

```bash
/Users/yunjinqi/opt/anaconda3/bin/conda run -n base python -m pytest -o addopts='' \
  tests/contracts/test_unified_invocation.py tests/contracts/test_compat_adapter_isolation.py \
  tests/contracts/test_plugin_overlay_snapshot.py tests/test_plugin tests/compat \
  -q --tb=short --maxfail=0
```

**Suggested commits:**

- `refactor: isolate strict compatibility adapters from enhanced state`
- `refactor: route enhanced operations through one invocation pipeline`
- `feat: make plugin overlays immutable and discoverable`

### Task 7：建立 AnalysisSnapshot、计算 DAG、显式 façade 与类型生成

**Owner:** Platform architecture  
**Depends on:** T6  
**Blocks:** T8–T12

**Files:**

- Create: `fincore/core/snapshot.py`
- Create: `fincore/core/planner.py`
- Create: `fincore/core/execution.py`
- Modify: `fincore/core/context.py`
- Modify: `fincore/core/engine.py`
- Modify: `fincore/report/compute.py`
- Modify: `pyproject.toml`
- Create: `scripts/generate_public_api.py`
- Create: `scripts/check_public_typing.py`
- Create: `fincore/__init__.pyi`
- Create: `fincore/empyrical.pyi`
- Create: `fincore/pyfolio.pyi`
- Create: `fincore/metrics/__init__.pyi`
- Create: `fincore/alphalens/__init__.pyi`
- Create: `fincore/factor_analysis/__init__.pyi`
- Create: `fincore/risk/__init__.pyi`
- Create: `fincore/simulation/__init__.pyi`
- Create: `fincore/attribution/__init__.pyi`
- Create: `fincore/optimization/__init__.pyi`
- Create: `fincore/report/__init__.pyi`
- Create: `tests/test_core/test_computation_dag.py`
- Create: `tests/typing/`

**Steps:**

1. `AnalysisSnapshot` 持有一次验证后的 returns/benchmark/positions/transactions/factors/semantics/config digest；对调用方输入做 copy-on-ingest。
2. DAG node 声明依赖和 cache key，共享 alignment、cum returns、drawdown、rolling moments、alpha/beta、factor panels 等中间量。
3. report 不再读取 `context._returns` 等私有属性，也不混用 Empyrical façade；context、report、rolling 共享 planner 和 snapshot。
4. cache key 包含数据内容、semantic profile、operation version、config、backend，以及 plugin overlay generation/kernel/distribution digest；默认只做进程内缓存，不缓存 credential/raw provider response。
5. 从 catalog 的 binding/typing contract 生成显式 wrapper、`__all__`、`__dir__`、所有 stable package `.pyi` 和 API docs table；逐步减少 runtime `exec`/ModuleType 魔法，但不一次删除兼容机制。
6. 在隔离目录构建 wheel，再对已安装 wheel 同时运行 mypy、`pyright --verifytypes`、stubtest 和独立 consumer 正/负例；不得只验证源码树。为 ndarray/Series/DataFrame、`out`、optional extra 添加 overload/Protocol。

**Acceptance:**

- report、context 与 direct enhanced API 对同一 snapshot 数值一致；共享 kernel 在同一 DAG 中只执行一次。
- data/config/semantic version/backend/plugin overlay 任一变化都会改变 cache key；等价 snapshot 与同一原子 overlay 产生稳定 digest。
- 所有 stable export 都有非 `Any`、非 Unknown 的 public typing；registry ↔ wrapper ↔ stub ↔ docs drift gate 通过。
- strict API signature snapshot、reload、monkeypatch、lazy import 和 eager import test 全绿。
- report 计算路径不再依赖 enhanced Empyrical/Pyfolio 类继承链。

**Commands:**

```bash
/Users/yunjinqi/opt/anaconda3/bin/conda run -n base python scripts/generate_public_api.py --check
/Users/yunjinqi/opt/anaconda3/bin/conda run -n base python -m pytest -o addopts='' \
  tests/test_core/test_computation_dag.py tests/typing tests/test_core tests/test_report tests/compat \
  -q --tb=short --maxfail=0
/Users/yunjinqi/opt/anaconda3/bin/conda run -n base python -m mypy fincore --ignore-missing-imports
/Users/yunjinqi/opt/anaconda3/bin/conda run -n base python -m pyright --verifytypes fincore
/Users/yunjinqi/opt/anaconda3/bin/conda run -n base python -m mypy.stubtest fincore
/Users/yunjinqi/opt/anaconda3/bin/conda run -n base python -m build --wheel --outdir dist/
/Users/yunjinqi/opt/anaconda3/bin/conda run -n base python scripts/check_public_typing.py --dist dist/
```

**Suggested commits:**

- `feat: share analysis intermediates through a deterministic dag`
- `build: generate explicit public api and type stubs`

### Task 8：建立机构级绩效分析语义与不确定性报告

**Owner:** Quant performance  
**Depends on:** T0、T7  
**Blocks:** T12 stable performance seal

**Files:**

- Create: `fincore/performance/returns.py`
- Create: `fincore/performance/cashflows.py`
- Create: `fincore/performance/inference.py`
- Create: `fincore/performance/disclosures.py`
- Modify: `fincore/metrics/returns.py`
- Modify: `fincore/metrics/ratios.py`
- Modify: `fincore/report/compute.py`
- Modify: `fincore/report/provenance.py`
- Create: `tests/oracles/performance/`
- Create: `tests/numerical/test_performance_semantics.py`
- Create: `mkdocs_docs/concepts/performance-semantics.md`

**Steps:**

1. 明确 simple/log、arithmetic/geometric、TWR/MWR/XIRR、gross/net-of-fees、cashflow timing、valuation timing、currency 与 benchmark/risk-free 单位。
2. 对 irregular frequency 与交易日历使用显式 annualization policy，不再依赖隐式 252；所有输出携带 units/frequency。
3. 为 Sharpe/Sortino/alpha/IC 等增加 standard error、置信区间或 small-sample diagnostic；修正并文档化 PSR/DSR。
4. 报告显示计算口径、数据区间、缺失/丢弃、费用、现金流、benchmark 和是否 annualized；禁止只显示无语境的单个数字。
5. 建立 PerformanceAnalytics/手算表格等独立 fixture，覆盖规则/不规则日期、极端回撤、现金流、费用和多币种边界。
6. 提供 GIPS-aware 计算与 disclosure 模板，但在文档与结果中明确“不是 GIPS compliance certification”。

**Acceptance:**

- TWR/MWR、现金流、费用、annualization 与独立 oracle 误差 `<= 1e-12`（迭代求解类指标按已批准 tolerance）。
- 同一数据在不同 return/frequency/currency profile 下要么得到已解释的不同结果，要么被明确拒绝。
- 每个 performance report 都包含语义、单位、样本期、数据质量和 provenance。
- strict Empyrical/Pyfolio 结果保持冻结；新语义只进入 enhanced profile。

**Commands:**

```bash
/Users/yunjinqi/opt/anaconda3/bin/conda run -n base python -m pytest -o addopts='' \
  tests/numerical/test_performance_semantics.py tests/oracles/performance \
  tests/property tests/compat/empyrical tests/compat/pyfolio -q --tb=short --maxfail=0
```

**Suggested commit:** `feat: add institution-grade performance semantics and disclosures`

### Task 9：完善风险模型验证、校准与监管参考报告

**Owner:** Quant risk  
**Depends on:** T0、T7  
**Blocks:** T12 stable risk seal

**Files:**

- Create: `fincore/risk/specs.py`
- Create: `fincore/risk/calibration.py`
- Create: `fincore/risk/diagnostics.py`
- Create: `fincore/risk/report.py`
- Modify: `fincore/risk/models.py`
- Modify: `fincore/risk/backtesting.py`
- Modify: `fincore/risk/garch.py`
- Modify: `fincore/risk/evt.py`
- Create: `tests/numerical/test_risk_model_validation.py`
- Create: `tests/property/test_risk_model_properties.py`
- Create: `mkdocs_docs/concepts/risk-model-validation.md`

**Steps:**

1. 用 `RiskModelSpec` 定义 forecast target、confidence、horizon、distribution、tail、sign、window、refit cadence 和 model version。
2. 统一输出 VaR/ES pair、forecast path、parameter estimates、convergence、residual diagnostics、data sufficiency 与 provenance；`risk/calibration.py` 只负责模型校准/统计检验，不建立第三套公共输入 validator。
3. 加入 Kupiec、Christoffersen independence/conditional coverage、Basel traffic-light reference、ES calibration 的明确方法与 experimental/stable 标签。
4. 实现真正 walk-forward/out-of-sample 回测；禁止用评估窗口本身估计同一时点 forecast。
5. GARCH/EVT 提供阈值/平稳性/残差/参数不确定性诊断；失败或样本不足使用结构化状态，不产生看似正常的数字。
6. 风险报告可重建每个 exception、forecast timestamp、model refit 和输入 digest；明确其为 Basel reference，不宣称监管批准。

**Acceptance:**

- 理论分布仿真下 VaR/ES coverage 与校准统计进入预先登记的置信区间；错误模型能被各独立测试拒绝。
- 250 observation traffic-light fixture 与 Basel reference 区间一致。
- walk-forward test 证明每个 forecast 只使用当时已知数据。
- risk capability 只有在 oracle、calibration、convergence 和 docs gates 全部通过后才可从 experimental 升 stable。

**Commands:**

```bash
/Users/yunjinqi/opt/anaconda3/bin/conda run -n base python -m pytest -o addopts='' \
  tests/numerical/test_risk_model_validation.py tests/property/test_risk_model_properties.py \
  tests/test_risk -q --tb=short --maxfail=0
```

**Suggested commit:** `feat: add auditable out-of-sample risk model validation`

### Task 10：将增强因子分析、归因与优化提升到研究级可信度

**Owner:** Quant factors/portfolio  
**Depends on:** T0、T7  
**Blocks:** T12 stable factor/attribution/optimization seal

**Files:**

- Create: `fincore/factor_analysis/pit.py`
- Create: `fincore/factor_analysis/inference.py`
- Create: `fincore/factor_analysis/costs.py`
- Modify: `fincore/factor_analysis/data.py`
- Modify: `fincore/factor_analysis/analysis.py`
- Modify: `fincore/factor_analysis/models.py`
- Modify: `fincore/factor_analysis/calendar.py`
- Modify: `fincore/attribution/fama_french.py`
- Modify: `fincore/attribution/brinson.py`
- Modify: `fincore/optimization/_utils.py`
- Modify: `fincore/optimization/frontier.py`
- Modify: `fincore/optimization/risk_parity.py`
- Create: `tests/oracles/factor_analysis/`
- Create: `tests/numerical/test_factor_inference.py`
- Create: `tests/numerical/test_optimization_feasibility.py`
- Create: `mkdocs_docs/concepts/factor-research-protocol.md`

**Steps:**

1. enhanced factor input 增加 `as_of/known_at/effective_from/universe`、corporate-action 与 calendar version；默认 causal/PIT，禁止全样本 z-score 过滤。
2. multi-horizon 数据按 horizon 保留可用样本，不用一次全列 `dropna()` 删除短周期有效记录；data-loss report 分原因、分 horizon。
3. 增加 Fama-MacBeth/cross-sectional regression、HAC 或双向 cluster SE、IC confidence、multiple-testing/FDR；记录研究者自由度和 trials。
4. 把 neutralization、turnover、交易成本、slippage、capacity 和 borrow availability 纳入 factor portfolio 结果。
5. strict Alphalens 完全保留 source-shaped 行为；所有 causal/PIT 改进只在 enhanced factor profile 中默认启用。
6. Brinson Hood 未实现的公开选项在 1.0 前必须二选一：实现并通过 oracle，或移出 stable public surface；不得长期公开抛 `NotImplementedError`。
7. optimization 增加 frequency contract、PSD/shrinkage/conditioning、可行性预检、constraint residual/KKT、solver provenance；不可行 frontier 不得静默填 NaN。

**Acceptance:**

- 构造未来数据扰动测试：改变 `as_of` 之后的数据不得改变之前的 enhanced factor 结果。
- null factor 的 IC/alpha 假阳性率进入预注册区间；FDR 与透明 oracle 一致。
- 成本前/成本后 factor returns、turnover/capacity 可 reconciliation。
- OLS/WLS/HAC/Fama-MacBeth 与 statsmodels/R fixture 在批准 tolerance 内一致。
- 两资产解析解/CVXPY oracle、权重和、边界、目标收益 residual `<= 1e-8`；不可行约束在求解前返回确定性错误。

**Commands:**

```bash
/Users/yunjinqi/opt/anaconda3/bin/conda run -n base python -m pytest -o addopts='' \
  tests/numerical/test_factor_inference.py tests/numerical/test_optimization_feasibility.py \
  tests/test_factor_analysis tests/test_attribution tests/test_optimization \
  tests/compat/alphalens -q --tb=short --maxfail=0
```

**Suggested commits:**

- `feat: add causal factor inference and research diagnostics`
- `fix: make attribution and optimization results reconcilable`

### Task 11：在语义 oracle 之后建立可扩展、高性能执行后端

**Owner:** Performance engineering  
**Depends on:** T0、T7；每个 domain 优化还依赖对应 T8/T9/T10 oracle  
**Blocks:** T12 performance seal

**Files:**

- Modify: `benchmarks/workloads.py`
- Modify: `benchmarks/bench_metrics.py`
- Modify: `benchmarks/bench_factor_analysis.py`
- Create: `benchmarks/bench_dispatch.py`
- Create: `benchmarks/bench_dag.py`
- Create: `benchmarks/bench_model_copy.py`
- Create: `fincore/backends/__init__.py`
- Create: `fincore/backends/numpy_backend.py`
- Optional after approval: `fincore/backends/array_api_backend.py`
- Create: `scripts/profile_workloads.py`
- Create: `scripts/check_performance.py`
- Create: `docs/quality/performance-budget.md`

**Steps:**

1. workload digest 必须 hash 实际输入内容、shape、dtype、calendar、seed 和 semantic profile，不只 hash 名称/行数。
2. 统一 small/medium/large：单序列、multi-asset、rolling、transactions、factor、risk forecast、report、DAG、serialization；warmup ≥2、repeat ≥7，记录 median/p95/MAD/RSS。
3. profile 后只优化排名前三的 CPU/RSS hotspot；先利用 DAG/shared intermediate、减少复制和 vectorization，再评估 Numba/Rust。
4. pandas/NumPy 为 reference backend；Array API 只覆盖不依赖 label/timezone/calendar 的 dense kernels，并要求 dtype/device/promotion 测试。
5. optional compiled/backend 必须显式选择、可回退、结果携带 backend/version；strict profile 默认继续使用 reference kernel。
6. 对 10k/100k/1m observations 建立 scaling curve，不只比较一个小样本的均值。

**Acceptance:**

- 每个 benchmark 先过 output digest/tolerance，再比较性能；语义不一致时不得生成“性能通过”。
- 相对批准 baseline 的 median/p95/RSS 回归分别不超过 10%/15%/10%，平台噪声策略写入 budget。
- 前三热点每项达到已批准的收益目标（建议至少 1.5× wall-time 或 30% RSS 改善），否则记录“不值得复杂化”的决策。
- dispatch/DAG 带来的固定开销有明确预算；单标量 metric 不因平台层增加数量级开销。
- baseline 必须 clean、同平台、批准、repeat 完整；candidate-only artifact 不能作为 release gate。

**Commands:**

```bash
/Users/yunjinqi/opt/anaconda3/bin/conda run -n base python scripts/profile_workloads.py --sizes small medium large
/Users/yunjinqi/opt/anaconda3/bin/conda run -n base python -m pytest -o addopts='' \
  tests/benchmarks -q --tb=short --maxfail=0 -n 0
/Users/yunjinqi/opt/anaconda3/bin/conda run -n base python scripts/check_performance.py
```

**Suggested commit:** `perf: add semantics-gated multi-scale execution benchmarks`

### Task 12：完成 1.0 级发布、供应链、治理、文档与最终 seal

**Owner:** Release engineering + Docs/community + Maintainers  
**Depends on:** T0–T11  
**Blocks:** 1.0 candidate

**Files:**

- Modify: `.github/workflows/ci.yml`
- Modify: `.github/workflows/publish.yml`
- Modify: `.github/dependabot.yml`
- Create: `.github/CODEOWNERS`
- Create: `.github/pull_request_template.md`
- Create: `SECURITY.md`
- Create: `SUPPORT.md`
- Create: `GOVERNANCE.md`
- Create: `MAINTAINERS.md`
- Create: `CODE_OF_CONDUCT.md`
- Create: `CITATION.cff`
- Modify: `pyproject.toml`
- Modify: `THIRD_PARTY_NOTICES.md`
- Modify: `MANIFEST.in`
- Modify: `CONTRIBUTING.md`
- Modify: `README.md`
- Modify: `mkdocs.yml`
- Modify: `mkdocs_docs/`
- Archive/index only: `docs/`
- Create: `scripts/build_release_candidate.py`
- Create: `scripts/verify_attestation.py`
- Create: `scripts/check_api_diff.py`
- Create: `docs/quality/1.0-readiness.json`
- Create: `docs/quality/1.0-readiness.md`

**Steps:**

1. 采用 PEP 639 license metadata/`license-files`；由人工/法律负责人完成 adapted code 文件级审批，机器只验证审批证据。
2. 固定所有 GitHub Actions 完整 SHA、workflow 最小权限、受保护 master、required checks、禁止 force push；PyPI environment 至少一名 reviewer。
3. 加入 dependency review、pip-audit/OSV、CodeQL/SAST、secret scan、SBOM、threat model；高危未豁免漏洞阻断 release。
4. 对同一 source candidate 生成 wheel/sdist、SPDX/CycloneDX SBOM、SLSA provenance、attestation、SHA256 release manifest；提供公开验证命令。
5. 明确 `mkdocs_docs` 为公共文档唯一源，`docs` 只保留计划、证据与历史；从 catalog 生成 API/capability/stability 页面。
6. 修复 CONTRIBUTING、project URLs、issue/PR 模板；补 migration、semantic cookbook、methodology、benchmark、citation、release/EOL policy。
7. 执行 API diff；breaking change 必须有批准 ADR、正确版本升级和弃用期。1.0 前完成所有 not_implemented stable path 的实现或移除。
8. 生成 readiness seal，但任何 human approval、当前 quality snapshot、oracle、candidate artifact 或 remote protection 缺失时状态必须是 blocked，而不是“接近完成”。

**Final acceptance matrix:**

| Gate | Exit condition |
| --- | --- |
| Numerical | 所有 stable domain 有独立 oracle + property + adversarial fixture；T0 所有错误关闭 |
| Compatibility | Empyrical/Pyfolio/Alphalens C0–C4 按各自 profile 全绿 |
| Semantics | catalog 覆盖 public exports 100%；enhanced contract/result/API drift 为 0 |
| Quality | clean current-commit full snapshot；branch floor ≥90%（若经 Task 1 批准），changed lines ≥95% |
| Types | mypy + pyright + stubtest installed-wheel 通过；stable public symbol 不退化为 Any |
| Performance | semantic digest 先通过；批准平台 baseline 无超预算回归 |
| Packaging | oldest/newest/optional extras 独立 wheel consumer 全绿；两次隔离构建可解释/可复核 |
| Supply chain | 同一 candidate digest；SBOM/provenance/attestation 可公开验证；Actions pin SHA |
| Legal | 所有 adapted component approved，reviewer/date/decision 与 artifact notice 一致 |
| Governance | protected branch、required checks、security/support/governance/maintainers/CoC 生效 |
| Documentation | strict MkDocs、links、代码块、版本、PyPI、README、中英文事实一致 |

**Final commands:**

```bash
/Users/yunjinqi/opt/anaconda3/bin/conda run -n base python -m ruff check fincore tests scripts examples benchmarks
/Users/yunjinqi/opt/anaconda3/bin/conda run -n base python -m ruff format --check fincore tests scripts examples benchmarks
/Users/yunjinqi/opt/anaconda3/bin/conda run -n base python -m mypy fincore --ignore-missing-imports
/Users/yunjinqi/opt/anaconda3/bin/conda run -n base python -m compileall -q fincore
/Users/yunjinqi/opt/anaconda3/bin/conda run -n base python -m pytest -o addopts='' \
  tests -q --tb=short --maxfail=0 --ignore=tests/benchmarks
/Users/yunjinqi/opt/anaconda3/bin/conda run -n base python -m pytest -o addopts='' \
  tests/benchmarks -q --tb=short --maxfail=0 -n 0
/Users/yunjinqi/opt/anaconda3/bin/conda run -n base python -m mkdocs build --strict
/Users/yunjinqi/opt/anaconda3/bin/conda run -n base python -m build --outdir dist/
/Users/yunjinqi/opt/anaconda3/bin/conda run -n base python scripts/test_installed_wheel.py \
  --dist dist/ --profiles core factor-analysis alphalens alphalens-pyfolio all
/Users/yunjinqi/opt/anaconda3/bin/conda run -n base python scripts/check_public_typing.py --dist dist/
/Users/yunjinqi/opt/anaconda3/bin/conda run -n base python scripts/check_release_candidate.py --dist dist/
/Users/yunjinqi/opt/anaconda3/bin/conda run -n base python scripts/verify_attestation.py --dist dist/
```

**Suggested commits:**

- `security: harden repository and release supply chain`
- `docs: publish generated world-class analytics documentation`
- `release: seal verified fincore 1.0 candidate`

## 6. 阶段性里程碑与退出标准

### Milestone A — Correctness Recovery

- T0、T1 完成。
- 所有 P0 数值错误关闭，错误 capability 降级或修复后重新批准。
- 当前 HEAD 有 clean、fresh、可重建质量证据。

### Milestone B — Semantic Convergence

- T2–T6 完成。
- OperationCatalog 成为唯一权威；strict/enhanced profile 明确隔离。
- enhanced 输入、异常、结果和插件调用协议统一。

### Milestone C — Platform Execution

- T7 完成。
- context/report/rolling 使用 Snapshot + DAG，共享中间计算。
- public stubs/docs/API map 可生成且 drift 为 0。

### Milestone D — Domain Excellence

- T8–T10 完成。
- performance/risk/factor/attribution/optimization 具有独立 oracle、推断、诊断和审计模型。

### Milestone E — Performance and 1.0 Candidate

- T11、T12 完成。
- 性能优化不破坏语义；同一 candidate 通过全部自动和人工 gates。
- readiness seal 无 unresolved blocker，才允许讨论 1.0 发布。

## 7. 明确不做的事

- 不把 strict 与 enhanced 强行统一为同一行为。
- 不为了 API “整洁”立刻删除 0.3.x public name；先 alias、警告、文档和完整弃用周期。
- 不在没有 independent oracle 前重写风险、模拟、归因或优化 kernel。
- 不把 Array API、Numba、Rust、GPU 当作目标本身；只有 workload 和 profile 证明收益后才引入。
- 不宣称 GIPS/Basel/监管合规认证；只提供可复核的计算支持和参考报告。
- 不用测试数量、行覆盖率或 benchmark 单点代替模型正确性、研究有效性和发布证据。

## 8. Definition of Done

本计划完成的定义不是“所有任务打勾”，而是同时满足：

1. 数值结论正确且有独立 oracle；
2. strict compatibility 没有回归；
3. enhanced API 的语义、输入、异常、结果、类型和文档来自同一 catalog；
4. context/report/factor/risk/optimization 可以在统一 snapshot/result/provenance 模型中组合；
5. 性能提升由多尺度、同平台、语义先行的 benchmark 证明；
6. 发布的 wheel 就是通过全部门禁的那一份 wheel；
7. 人工许可证审批、分支保护和发布环境保护均有当前证据；
8. 任何一项缺失都会让 readiness seal 明确显示 blocked。

只有达到以上条件，fincore 才具备从“功能丰富的 Beta 包”升级为“可被研究、资管、风控和平台工程团队长期依赖的一流金融分析基础设施”的可信基础。
