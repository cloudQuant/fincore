# Fincore Financial Analytics Platform Excellence (Iteration 0041) Implementation Plan

> **For Codex:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 将 fincore 由功能广、已有兼容层和工程门禁的 Beta 库，迭代为可复现、可验证、可扩展且性能可证明的金融绩效分析、风险计量和因子研究平台。

**Architecture:** 保持现有三类边界：fincore.empyrical、fincore.pyfolio、fincore.alphalens 是冻结来源和语义的 strict façade；fincore.metrics、risk、factor_analysis、attribution、report 是可演进的 enhanced kernel；contracts 负责输入、时序和结果约束。新能力先在 enhanced 层建立结构化结果、来源和验证证据，再决定是否需要进入 strict façade，绝不通过修改兼容行为来顺便升级金融语义。

**Tech Stack:** Python 3.11+、NumPy、pandas、SciPy、statsmodels、pytest/pytest-xdist/pytest-cov、mypy、Ruff、setuptools/PEP 517、MkDocs、现有 fresh-subprocess benchmark runners；只在基准和兼容证明后评估新的可选加速依赖。

---

## 0. 计划状态、审计边界与北极星

- 状态：Proposed
- 审计日期：2026-08-17
- 审计提交：6135890
- 审计工作树：干净；本次只新增本计划，不修改业务实现或生成质量快照。
- 现有相关计划：
  - docs/plans/2026-08-12-fincore-empyrical-pyfolio-convergence.md
  - docs/plans/2026-08-13-fincore-alphalens-integration.md
- 本计划不撤销上述计划中已冻结的 strict façade、C0-C4、wheel 和 provenance 约束；它处理当前剩余的横向产品化、风控可信度与性能治理工作。

### 0.1 北极星定义

一流不以指标数量、历史覆盖率或一次绿色运行衡量，而是同时满足下列可验证结果：

| 维度 | 目标状态 | 证据形式 |
| --- | --- | --- |
| 金融语义 | 每个风险、归因和因子结果都有输入、时间、符号、异常和缺失值语义 | 版本化 contract、数值 oracle、性质测试 |
| 研究可复现 | 报告、因子实验和风控结果可追溯到数据快照、配置、代码和依赖版本 | 不含原始敏感数据的 audit manifest |
| 风险可信 | VaR/ES 不仅能计算，还具备样本外回测、例外统计和失败状态 | RiskEstimate、RiskBacktestResult、固定 fixture |
| 性能可信 | 热点时间、RSS、输出 digest 和基准来源可比较；回归会阻断发布 | 平台标签化 baseline 与 CI compare gate |
| 兼容可控 | strict façade 的承诺与 enhanced API 的创新互不污染 | manifest、profile、C0-C4、fresh wheel |
| 交付可信 | 文档、类型、依赖、许可证和发行物由机器门禁佐证 | clean baseline、docs build、SBOM/NOTICE review、wheel consumer |

### 0.2 本次审计的已核验证据与优先级

| 优先级 | 观察到的事实 | 影响 | 本计划的处理 |
| --- | --- | --- | --- |
| P0 | docs/quality/current-baseline.json 指向旧提交 58a4c08，且记录 dirty=true；当前实际分支覆盖率 ~94%（来自 convergence 计划审计），但 snapshot 中记录的 55% branch coverage 已过期，不能作为当前 master 的发布证据 | 质量数字可能过期或不可复现 | Task 1 先建立 clean-commit freshness gate 和重新生成流程 |
| P0 | 当前 Anaconda base 中，yfinance 0.2.66 和 akshare 1.18.64 均在导入 curl_cffi 0.13.0 后经 eventlet 0.40.2/PyOpenSSL 24.0.0 失败，根异常为 OpenSSL lib 缺少 GEN_EMAIL；当前实际可收集的 24 个 provider failure 节点均失败 | data-yahoo/data-cn 能力在已声明环境不可用，且单元测试依赖真实可选 SDK 导入 | Task 3 隔离 client、统一 DependencyError 并增加 optional-extra import smoke；Task 10 固化兼容依赖矩阵 |
| P0 | benchmarks/factor-analysis-baseline.json 为 candidate-only-not-release-approved，只有 1 repeat、0 warmup、dirty=true；CI 仅上传 factor artifact，release build 不依赖它 | 因子性能没有真正的发布阻断证据 | Task 6 将 approved baseline 与 compare job 收口 |
| P0 | risk 目录目前聚焦 EVT 和 GARCH（`fincore/risk/evt.py` 和 `fincore/risk/garch.py`）；未发现 forecast 与 realized loss 的 VaR/ES 回测公开模块 | 能算风险不等于模型受验证 | Task 4 建立 enhanced 风险结果与回测层，新增 `models.py`、`backtesting.py` 独立于现有 EVT/GARCH 模块 |
| P1 | fetch_ff_factors、fetch_style_factors 需要外部 provider；BrinsonAttribution 的 brinson_hood 是公开选项但抛 NotImplementedError（`brinson.py:291-292`） | 能力宣传、研究数据来源和实际可用性之间有边界 | Task 2/3 明确 capability 状态并补 provider/算法决策 |
| P1 | factor_analysis/calendar.py 在质量快照中产生 non-vectorized DateOffset PerformanceWarning；factor cleaning、round trips、rolling 中仍有 groupby.apply、逐行或循环 concat 热点 | 真实大面板/长交易序列可能退化 | Task 7 先 profile，Task 8 只优化被证实热点 |
| P1 | Ruff 已通过、554 个文件格式正确、默认 mypy 对 115 个文件为 0 error，但 mypy 提示部分无注解函数体未检查 | 当前静态绿不等于类型契约完整 | Task 10 分层提升类型与依赖兼容性 |
| P2 | 报告已有 compute-once/render-many 的 ReportModel（`fincore/report/model.py`、`compute.py`、`format.py`），但没有统一 input/config/code provenance manifest | 研究结论与报告文件难以审计复现 | Task 9 以 enhanced、显式 opt-in 方式加审计包 |

### 0.3 全局决策与非目标

1. 保持 Beta。任何发布说明在 clean baseline、compatibility、风险回测、性能 baseline、wheel、文档和人工许可证审阅完成前，不得宣称 Production、Stable 或 1.0。
2. 不引入外部 empyrical、pyfolio、alphalens 作为 runtime dependency；也不创建顶层同名 alias。
3. 不因为性能优化引入未经证明的 Numba、Polars、Arrow、Cython 或 GPU 后端。每种候选后端先通过 Task 7 的等价性、冷启动、RSS、安装体积和 wheel 矩阵 decision gate。
4. 不隐式下载数据、不将缓存写入源码或 package 目录、不把 API key 写入配置、日志或报告。磁盘缓存必须由调用方显式指定目录。
5. 风险回测是研究与模型治理能力，不构成任何司法辖区的合规认证或投资建议。
6. 不做一次横跨全仓库的大重构；每个变更必须有 strict façade 回归、enhanced 数值断言和安装包消费者证明。

### 0.4 建议分工与依赖

| Track | 负责范围 | 可并行起点 | 主要交付 |
| --- | --- | --- | --- |
| A：证据与质量 | Task 1、10、11、12 | 立即 | 可信基线、类型/依赖矩阵、发布门禁、文档 |
| B：数据与归因 | Task 2、3 | Task 1 capability schema 完成后 | data snapshot、provider contract、归因状态收口 |
| C：风控与报告 | Task 4、9 | Task 1 后；Task 9 可与 4 并行 | 风险回测、audit manifest |
| D：性能与因子 | Task 6、7、8 | Task 1 后；Task 7 可与 6 并行 | approved baseline、热点优化 |
| E：验证策略 | Task 5、11 | Task 1 后；对所有 Track 持续服务 | oracle、性质测试、供应链和 wheel 证据 |

**执行依赖（优化后）：**

```
Task 1 (freshness gate) ── 必须先完成，是所有后续任务的前置
  ├── Task 2 (capability inventory) ── 依赖 Task 1
  │     └── Task 3 (provider contracts) ── 依赖 Task 2
  ├── Task 4 (risk backtesting) ── 依赖 Task 1，可与 Task 2/3/6/10 并行
  │     └── Task 9 (audit manifest) ── 依赖 Task 4
  ├── Task 5 (property tests) ── 依赖 Task 1，可与 2/3/4/6/10 并行，持续服务所有 Track
  ├── Task 6 (factor baseline) ── 依赖 Task 1，可与 Task 2/3/4/10 并行
  │     ├── Task 7 (profile corpus) ── 依赖 Task 6（baseline 协议），但 workload 定义可提前并行
  │     │     └── Task 8 (hotspot optimization) ── 依赖 Task 7
  ├── Task 10 (type/dep matrix) ── 依赖 Task 1，可与 Task 2/3/4/6 并行
  ├── Task 11 (supply chain) ── 依赖 Task 5、9、10
  └── Task 12 (docs) ── 依赖 Task 2、3、4、9（需要能力状态、风险回测、审计 manifest 的产出）
```

**并行化优化说明：**
- Task 7 的 workload 定义（`benchmarks/workloads.py`）不依赖 Task 6 的 approved baseline 协议，可以提前开始；只有 compare-gate 部分需要等 Task 6
- Task 5（性质测试）与 Track B/C/D 的所有任务并行，持续为它们提供验证服务
- Task 10 的依赖矩阵和类型检查独立于金融领域任务，可以立即开始

### 0.5 CI 协调约定

多个任务修改 `.github/workflows/ci.yml`，为避免冲突：
- 每个 Track 在自己的 worktree/branch 中只追加新 job，不重排已有 job
- 最终由 Track A 在 Task 12 前合并所有 CI 变更并验证 job 依赖图完整性
- 新增 job 的 `needs` 列表不得引用尚未合并的 job 名称

### 0.6 部分部署与回滚策略

- 每个 Task 独立可交付、可验证、可 revert
- 如果 Task 3 的数据 provider contract 比预期复杂，可以交付不含 `brinson_hood` 实现的版本（保持 `not_implemented` 状态），后续迭代补充
- 如果 Task 4 的 ES 回测统计需要更多研究，可以先交付 VaR 回测（unconditional coverage + independence），ES 校准分数标记为 `experimental`
- 所有 Task 的变更是增量式（additive），不在同一 PR 中混合多个 Task 的变更

---

## 1. 迭代任务

### Task 1: 建立当前提交可信质量快照与 freshness gate

**Owner:** Track A

**Files:**

- Create: scripts/check_quality_snapshot.py
- Create: tests/quality/test_check_quality_snapshot.py
- Modify: scripts/collect_quality_baseline.py
- Modify: tests/quality/test_collect_quality_baseline.py（已存在，含 4 个已有测试文件）
- Modify: .github/workflows/ci.yml
- Generate at final step only: docs/quality/current-baseline.json
- Generate at final step only: docs/quality/current-baseline.md

**Step 1: Write the failing freshness tests**

```python
def test_rejects_dirty_or_wrong_commit_snapshot(tmp_path: Path) -> None:
    snapshot = {"source": {"commit": "old", "dirty": True}, "outcome": "pass"}
    path = tmp_path / "snapshot.json"
    path.write_text(json.dumps(snapshot))
    assert check_snapshot(path, expected_commit="current") == [
        "source.commit does not match HEAD",
        "source.dirty must be false",
    ]

def test_current_branch_coverage_meets_threshold() -> None:
    """The snapshot must record branch coverage >= the project minimum (60%)."""
    snapshot = load_snapshot(ROOT / "docs" / "quality" / "current-baseline.json")
    assert snapshot["coverage"]["branch"] >= 60.0, (
        f"Branch coverage {snapshot['coverage']['branch']}% below 60% threshold"
    )
```

**Step 2: Verify the test fails because no checker exists**

Run:

```sh
python -m pytest -o addopts='' tests/quality/test_check_quality_snapshot.py -q
```

Expected: FAIL during collection because scripts.check_quality_snapshot is absent.

**Step 3: Implement a fail-closed checker and artifact metadata**

- Check outcome=pass, source.commit equals exact HEAD, source.dirty is false, every declared run has returncode 0 and integrity_ok true, and branch coverage is present and >= 60%.
- Add schema version plus exact selector/command to the collector. It must never claim an old snapshot represents the current commit.
- Use a fresh pytest cache inside the disposable copy or disable the cache provider for baseline collection. Baseline commands must not use the developer-machine lastfailed cache; stale node IDs are diagnostic residue, not failure evidence.
- CI runs the checker after coverage and uploads the generated snapshot.

**Step 4: Run targeted and clean-copy baseline tests**

Run:

```sh
python -m pytest -o addopts='' tests/quality -q --maxfail=0
python scripts/collect_quality_baseline.py
python scripts/check_quality_snapshot.py --snapshot docs/quality/current-baseline.json
```

Expected: all targeted tests pass; generated snapshot is clean, current and complete.

**Step 5: Commit owned files**

```sh
git add scripts/check_quality_snapshot.py scripts/collect_quality_baseline.py tests/quality .github/workflows/ci.yml docs/quality/current-baseline.json docs/quality/current-baseline.md
git commit -m "test: require a fresh clean quality snapshot"
```

**Exit criteria:** No current quality number is accepted unless it passes the checker on the same clean commit.

---

### Task 2: 发布机器可读的能力状态清单

**Owner:** Track B with Track A review

**Files:**

- Create: fincore/capabilities.py
- Create: tests/contracts/test_capabilities.py
- Create: scripts/render_capability_inventory.py
- Create: docs/quality/capability-inventory.md
- Modify: fincore/attribution/__init__.py
- Modify: fincore/risk/__init__.py
- Modify: fincore/data/__init__.py
- Modify: mkdocs_docs/ecosystem.md

**Step 1: Write the failing capability-contract tests**

```python
def test_public_capabilities_have_unique_ids_and_actionable_statuses() -> None:
    rows = list_capabilities()
    assert {row.status for row in rows} <= {
        "stable", "experimental", "provider_required", "not_implemented",
    }
    assert len({row.id for row in rows}) == len(rows)
    assert all(row.docs_path for row in rows)

def test_brinson_hood_is_not_implemented() -> None:
    """Until Task 3 delivers a verified implementation, brinson_hood must be not_implemented."""
    cap = get_capability("attribution.brinson_hood")
    assert cap.status == "not_implemented"
```

**Step 2: Verify the registry is missing**

Run:

```sh
python -m pytest -o addopts='' tests/contracts/test_capabilities.py -q
```

Expected: FAIL because fincore.capabilities does not exist.

**Step 3: Implement a declarative, import-light registry**

- Add immutable Capability records with id, public_path, domain, status, input_contract, output_contract, docs_path and rationale.
- Register EVT/GARCH, strict compatibility façades, report generation, factor analysis, Fama-French/style provider entry points, and both Brinson methods.
- Mark provider functions as provider_required and brinson_hood as not_implemented until Task 3 has a verified implementation.
- Render checked-in Markdown from this registry and reject undocumented public rows.

**Step 4: Verify registry and docs agreement**

Run:

```sh
python scripts/render_capability_inventory.py --check
python -m pytest -o addopts='' tests/contracts/test_capabilities.py -q
python -m mkdocs build --strict
```

Expected: registry, generated inventory and public docs agree exactly.

**Step 5: Commit owned files**

```sh
git add fincore/capabilities.py fincore/attribution/__init__.py fincore/risk/__init__.py fincore/data/__init__.py tests/contracts/test_capabilities.py scripts/render_capability_inventory.py docs/quality/capability-inventory.md mkdocs_docs/ecosystem.md
git commit -m "docs: publish fincore capability states"
```

**Exit criteria:** Users can distinguish stable, experimental, provider-required and unavailable surfaces without reading source or triggering an exception.

---

### Task 3: 给数据来源和归因功能增加可复现的 provider contract

**Owner:** Track B

**Files:**

- Create: fincore/data/contracts.py
- Create: fincore/data/snapshots.py
- Create: tests/test_data/test_snapshots.py
- Create: tests/test_attribution/test_provider_contracts.py
- Create: tests/test_data/test_optional_dependency_health.py
- Modify: fincore/data/providers.py（现有 30KB 模块，注意增量修改）
- Modify: fincore/exceptions.py（已有 DependencyError 等异常类）
- Modify: fincore/attribution/fama_french.py
- Modify: fincore/attribution/style.py
- Modify: fincore/attribution/brinson.py
- Modify: mkdocs_docs/guide/concepts.md
- Modify: .github/workflows/ci.yml

**Step 1: Write failing provider and provenance tests**

```python
def test_snapshot_hash_is_stable_and_excludes_secret_configuration() -> None:
    snapshot = DataSnapshot.from_frame(
        frame=pd.DataFrame({"close": [10.0]}),
        provider="fixture",
        requested_start="2024-01-01",
        requested_end="2024-01-02",
        as_of="2024-01-03T00:00:00Z",
    )
    assert snapshot.content_sha256 == DataSnapshot.from_frame(snapshot.data, **snapshot.identity_kwargs()).content_sha256
    assert "api_key" not in snapshot.to_manifest()

def test_provider_can_use_an_injected_fake_client_when_sdk_is_unavailable() -> None:
    provider = YahooFinanceProvider(client=FakeYahooClient())
    assert provider.validate_dates("2024-01-01", "2024-01-02")[0].year == 2024

def test_broken_optional_sdk_raises_dependency_error_not_attribute_error() -> None:
    """yfinance/akshare import failure must produce fincore.DependencyError, not raw AttributeError."""
    with pytest.raises(DependencyError, match="data-yahoo"):
        YahooFinanceProvider()  # when yfinance is unavailable or broken
```

**Step 2: Verify the contract is absent**

Run:

```sh
python -m pytest -o addopts='' tests/test_data/test_snapshots.py tests/test_attribution/test_provider_contracts.py -q
```

Expected: FAIL because DataSnapshot and the provider protocols do not exist.

**Step 3: Implement explicit data and provider semantics**

- DataSnapshot contains defensive data, source identifier, request interval, as_of timestamp, price-adjustment convention, timezone, schema version and content SHA256.
- Add bounded RequestPolicy with connect/read/total timeouts, max_attempts and deterministic retry classification. Never log credentials or retry caller validation errors.
- Preserve DataProvider.fetch returning DataFrame for compatibility. Add enhanced fetch_snapshot rather than silently changing old call sites.
- Make provider constructors accept a testable injected client/transport. When the optional SDK import itself raises any exception, convert it to a fincore DependencyError that names the required extra and retains the original exception as __cause__; do not leak a raw third-party AttributeError from a constructor.
- Add a clean-environment CI smoke job for data-yahoo and data-cn. It must install each extra from the Task 10 constraints, import its SDK in a new interpreter and run fake-client provider tests; it must never contact a market-data service.
- Make FamaFrenchProvider and StyleFactorProvider explicit protocols with set/clear functions and cache invalidation; do not introduce a default network fetcher here.
- Either implement a formula-tested brinson_hood method with an internal formula specification and oracle fixtures, or remove it from callable options and leave it not_implemented. If implementing, add a dedicated oracle fixture in `tests/test_attribution/fixtures/`.

**Step 4: Run deterministic fake-provider tests, existing suites, AND compat gate**

Run:

```sh
python -m pytest -o addopts='' tests/test_data tests/test_attribution -q --maxfail=0
python -m pytest -o addopts='' tests/compat -q --maxfail=0
```

Expected: no network access; retry, partial failure, cache-copy, provenance and attribution semantics use fake providers. A broken optional SDK produces one controlled DependencyError and cannot invalidate client-injected unit tests. Strict façade compatibility (C0-C4) remains green.

**Step 5: Commit owned files**

```sh
git add fincore/data fincore/exceptions.py fincore/attribution tests/test_data tests/test_attribution mkdocs_docs/guide/concepts.md .github/workflows/ci.yml
git commit -m "feat: add reproducible data provider contracts"
```

**Exit criteria:** Every external-data analysis identifies its data snapshot and request convention; unavailable attribution methods cannot masquerade as usable API.

---

### Task 4: 建立 enhanced 风险结果与样本外回测层

**Owner:** Track C

**Files:**

- Create: fincore/risk/models.py（新增，独立于现有 evt.py/garch.py）
- Create: fincore/risk/backtesting.py（新增）
- Create: tests/test_risk/test_models.py
- Create: tests/test_risk/test_backtesting.py
- Create: tests/test_risk/fixtures/risk_backtest_cases.json
- Modify: fincore/risk/__init__.py（导出新模块的公共符号，不改动现有 EVT/GARCH 导出）
- Modify: mkdocs.yml
- Create: mkdocs_docs/guide/risk-validation.md

**架构说明：** 现有 `fincore/risk/evt.py` 和 `fincore/risk/garch.py` 保持不变。新增的 `models.py` 定义 `RiskEstimate` 结果容器和 `forecast_var`/`forecast_es` 等增强适配器（内部调用现有 EVT/GARCH 函数），`backtesting.py` 实现独立的回测统计逻辑。两者都是增强层，不修改 strict façade。

**Step 1: Write failing deterministic risk-backtest tests**

```python
def test_var_backtest_keeps_time_alignment_and_exception_count() -> None:
    forecast = pd.Series([-0.02, -0.02, -0.02], index=pd.date_range("2024-01-01", periods=3, tz="UTC"))
    realized = pd.Series([-0.01, -0.03, -0.02], index=forecast.index)
    result = backtest_var(forecast, realized, confidence_level=0.99)
    assert result.observations == 3
    assert result.exceptions == 1
    assert result.aligned_index.equals(forecast.index)

def test_risk_estimate_rejects_duplicate_index() -> None:
    idx = pd.date_range("2024-01-01", periods=2, tz="UTC").append(
        pd.date_range("2024-01-01", periods=1, tz="UTC")
    )
    with pytest.raises(ValueError, match="duplicate"):
        RiskEstimate(forecast=pd.Series([-0.01, -0.01, -0.01], index=idx), ...)
```

**Step 2: Verify the enhanced API is missing**

Run:

```sh
python -m pytest -o addopts='' tests/test_risk/test_models.py tests/test_risk/test_backtesting.py -q
```

Expected: FAIL because RiskEstimate and backtest_var are absent.

**Step 3: Implement result contracts before model adapters**

- Add immutable RiskEstimate and RiskBacktestResult with method, confidence level, horizon, forecast timestamp, sign convention, inputs digest, estimate, diagnostics and status.
- Add enhanced rolling forecast adapters for historical, EVT and GARCH without changing existing legacy functions in `evt.py`/`garch.py`.
- Implement VaR exception counting, unconditional coverage (Kupiec) and independence (Christoffersen) statistics with explicit null hypotheses and small-sample status.
- Add a named ES calibration score with documented assumptions. **注意：** ES 回测是风险管理中的开放问题；首版实现将使用基于 bootstrap 的校准检验，标记为 `experimental` 状态，并在文档中说明假设和局限性。
- Reject duplicate, unsorted, timezone-incompatible or non-overlapping inputs through existing contracts. A failed diagnostic is data, not an unhandled warning.

**Step 4: Run risk domain tests, oracle checks, AND compat gate**

Run:

```sh
python -m pytest -o addopts='' tests/test_risk -q --maxfail=0
python -m pytest -o addopts='' tests/compat -q --maxfail=0
```

Expected: legacy EVT/GARCH remains green; fixtures prove sign convention, alignment, edge state and known statistic values. Strict façade compatibility unchanged.

**Step 5: Commit owned files**

```sh
git add fincore/risk tests/test_risk mkdocs.yml mkdocs_docs/guide/risk-validation.md
git commit -m "feat: add risk forecast validation contracts"
```

**Exit criteria:** Risk results say what was forecast, under which convention and horizon, and whether realized outcomes pass, fail or are statistically inconclusive.

---

### Task 5: 将 example assertions 扩展为性质与差分验证

**Owner:** Track E

**Files:**

- Create: tests/property/test_time_series_contracts.py
- Create: tests/property/test_risk_invariants.py
- Create: tests/property/test_factor_invariants.py
- Create: tests/oracles/README.md
- Modify: pyproject.toml（添加 hypothesis 到 test deps 和 pytest marker）
- Modify: requirements-test.txt（添加 `hypothesis>=6.100`）
- Modify: .github/workflows/ci.yml（添加 property test job）

**属性测试框架选型：** 使用 **hypothesis**（最成熟的 Python 属性测试库，与 pytest 深度集成，支持 deterministic seed 和 shrinking）。不使用 pytest-quickcheck（功能较弱）或 st（不成熟）。

**Step 1: Write a minimal property test that exposes an invariant**

```python
from hypothesis import given, strategies as st, settings

@given(st.lists(st.floats(min_value=-0.2, max_value=0.2, allow_nan=False), min_size=3, max_size=40))
@settings(max_examples=200, deadline=None)
def test_cumulative_return_is_unchanged_by_series_copy(values: list[float]) -> None:
    original = pd.Series(values)
    copied = original.copy(deep=True)
    assert cumulative_returns(original) == cumulative_returns(copied)
```

**Step 2: Verify it fails before its dependency and policy are installed**

Run:

```sh
python -m pytest -o addopts='' tests/property -q
```

Expected: FAIL during collection until hypothesis is declared in test dependencies.

**Step 3: Add bounded high-value properties and independent oracles**

- Add hypothesis to test/development dependencies with deterministic seed reporting (`hypothesis>=6.100`) and bounded CI examples (`max_examples=200`).
- Cover alignment, timezone normalization, NaN/empty profiles, return-scale consistency, risk sign convention, factor quantile partitioning and no-mutation boundaries.
- For each critical family keep a small checked-in independent oracle fixture or transparent NumPy reference. It must not call the function under test.
- Add a serial CI execution path (`-n 0`) so shrinking and global seed state remain reproducible.

**Step 4: Verify strict and enhanced layers together**

Run:

```sh
MPLBACKEND=Agg python -m pytest -o addopts='' tests/property tests/contracts tests/compat tests/test_risk tests/test_factor_analysis -q --maxfail=0
```

Expected: properties pass without weakening frozen C0-C4 assertions.

**Step 5: Commit owned files**

```sh
git add tests/property tests/oracles pyproject.toml requirements-test.txt .github/workflows/ci.yml
git commit -m "test: add financial invariants and differential oracles"
```

**Exit criteria:** Critical results are tested for classes of inputs and independent semantics, not only hand-selected examples.

---

### Task 6: 将因子性能 artifact 升级为经审核、可阻断的 baseline

**Owner:** Track D with Track A approval

**Files:**

- Create: benchmarks/factor-analysis-baselines/linux-x86_64.json（从现有 `benchmarks/factor-analysis-baseline.json` 迁移）
- Create: benchmarks/factor-analysis-baselines/darwin-arm64.json
- Create: docs/quality/factor-benchmark-approval.md
- Modify: scripts/run_factor_benchmarks.py
- Modify: scripts/compare_benchmarks.py
- Modify: tests/benchmarks/test_factor_analysis_performance.py
- Modify: .github/workflows/ci.yml
- Modify: docs/quality/release-candidate-checklist.md
- Migrate/Remove: benchmarks/factor-analysis-baseline.json（旧单文件，迁移后删除）

**现有基线迁移：** 当前 `benchmarks/factor-analysis-baseline.json` 是 candidate-only-not-release-approved。Task 6 将其迁移到 `benchmarks/factor-analysis-baselines/` 目录下的平台标签化文件，同时添加 approval 元数据。

**Step 1: Write a failing platform-baseline selection test**

```python
def test_selects_only_an_approved_matching_platform_baseline(tmp_path: Path) -> None:
    baseline = select_baseline(tmp_path, platform_label="linux-x86_64")
    assert baseline.approval.status == "approved"
    assert baseline.provenance.platform_label == "linux-x86_64"

def test_candidate_only_baseline_is_not_a_release_gate() -> None:
    """Existing candidate-only baseline must not silently promote to approved."""
    candidates = list_candidate_baselines()
    approved = [c for c in candidates if c.approval.status == "approved"]
    unapproved = [c for c in candidates if c.approval.status != "approved"]
    assert len(approved) == 0 or all(a.provenance.platform_label for a in approved), (
        f"Found {len(unapproved)} unapproved baselines that must not gate releases"
    )
```

**Step 2: Verify pending Darwin candidate cannot be a release gate**

Run:

```sh
python -m pytest -o addopts='' tests/benchmarks/test_factor_analysis_performance.py -q
```

Expected: new selection test fails until matching approved-baseline workflow exists; current candidate-only artifact never silently promotes.

**Step 3: Implement reviewed-baseline protocol and CI comparison**

- Preserve digest-before-time/RSS comparison and output-shape checks.
- Generate clean candidates using agreed warmup/repeat protocol (minimum 2 warmups, 5 repeats). Store platform label, source commit, dependency versions, digest, candidate SHA256, reviewers and approval time.
- Maintain independent Linux CI and Darwin reference baselines. Platform mismatch uploads evidence but cannot compare unrelated hosts.
- Add factor-performance comparison to release build needs only after its approval record is complete.
- CI never overwrites a baseline; promotion is a reviewed commit.

**Step 4: Verify candidate, rejection and approved path**

Run:

```sh
python scripts/run_factor_benchmarks.py --scenarios small-ci --warmups 2 --repeats 5 --output build/factor-candidate.json
python scripts/compare_benchmarks.py --baseline benchmarks/factor-analysis-baselines/linux-x86_64.json --candidate build/factor-candidate.json --digest-gate sha256
python -m pytest -o addopts='' tests/benchmarks/test_factor_analysis_performance.py -q
```

Expected: unreviewed/wrong-platform baseline fails clearly; reviewed matching baseline performs digest, time and RSS comparison.

**Step 5: Commit owned files**

```sh
git add benchmarks/factor-analysis-baselines scripts/run_factor_benchmarks.py scripts/compare_benchmarks.py tests/benchmarks/test_factor_analysis_performance.py .github/workflows/ci.yml docs/quality
git rm benchmarks/factor-analysis-baseline.json  # 迁移后删除旧文件
git commit -m "perf: gate factor analysis against approved platform baselines"
```

**Exit criteria:** Factor performance is release-blocking only when a matching approved baseline exists; otherwise its absence is explicit rather than a pass.

---

### Task 7: 建立跨领域性能 profile corpus，并先量化再优化

**Owner:** Track D

**Files:**

- Create: benchmarks/workloads.py
- Create: scripts/profile_hotspots.py
- Create: tests/benchmarks/test_workloads.py
- Create: docs/quality/performance-methodology.md
- Modify: benchmarks/bench_metrics.py
- Modify: benchmarks/bench_factor_analysis.py
- Modify: scripts/run_rolling_benchmarks.py
- Modify: scripts/run_round_trip_benchmarks.py

**并行化说明：** workload 定义（`benchmarks/workloads.py`）和 profile 脚本不依赖 Task 6 的 approved baseline。Task 7 可以在 Task 6 完成 baseline 协议的同时开始。只有后续 compare-gate 集成需要 Task 6 的产出。

**Step 1: Write failing deterministic workload tests**

```python
def test_factor_workload_has_fixed_shape_seed_and_output_digest() -> None:
    case = factor_panel_workload("medium", seed=20260817)
    assert case.factor.index.nlevels == 2
    assert case.expected_rows == 630000
    assert len(case.input_digest) == 64
```

**Step 2: Verify common workload factories do not exist**

Run:

```sh
python -m pytest -o addopts='' tests/benchmarks/test_workloads.py -q
```

Expected: FAIL because benchmarks.workloads is absent.

**Step 3: Implement benchmark corpus and profile output**

- Define deterministic small, medium and large workloads for cold import, single-series metrics, rolling metrics, factor panels, transaction FIFO/round trips and report model computation.
- Record wall time, peak RSS, Python allocation peak, output shape/digest, seed, platform and dependency versions in one schema.
- profile_hotspots.py emits machine-readable top cumulative functions plus human Markdown summary; it uses subprocesses for cold import and RSS.
- Document warmup (minimum 2), median-of-N (minimum 5), platform separation, noise rules and no cross-platform comparison.

**Step 4: Produce pre-optimization evidence**

Run:

```sh
python scripts/profile_hotspots.py --scenario medium --output build/hotspots-before.json
python -m pytest -o addopts='' tests/benchmarks/test_workloads.py -q
```

Expected: output identifies whether calendar, factor cleaning, rolling, round trips or report compute is dominant.

**Step 5: Commit owned files**

```sh
git add benchmarks/workloads.py benchmarks/bench_metrics.py benchmarks/bench_factor_analysis.py scripts/profile_hotspots.py scripts/run_rolling_benchmarks.py scripts/run_round_trip_benchmarks.py tests/benchmarks/test_workloads.py docs/quality/performance-methodology.md
git commit -m "perf: add reproducible platform workload corpus"
```

**Exit criteria:** Every proposed optimization has a before profile, fixed workload, numerical digest and success budget.

---

### Task 8: 分批优化已证实热点，保持数值与兼容语义

**Owner:** Track D

**Files:**

- Modify only after Task 7 identifies them: fincore/factor_analysis/calendar.py
- Modify only after Task 7 identifies them: fincore/factor_analysis/data.py
- Modify only after Task 7 identifies them: fincore/metrics/rolling.py
- Modify only after Task 7 identifies them: fincore/metrics/round_trips.py
- Create: tests/performance/test_hotspot_regressions.py
- Modify: tests/test_factor_analysis/test_calendar.py
- Modify: tests/test_factor_analysis/test_data.py
- Modify: tests/test_metrics/test_rolling.py
- Modify: tests/test_metrics/test_round_trips.py

**Step 1: Write semantic and budget test for each selected hotspot**

```python
def test_round_trips_preserve_fifo_result_before_optimization() -> None:
    actual = extract_round_trips(transaction_fixture())
    pd.testing.assert_frame_equal(actual, expected_round_trips())

def test_round_trips_medium_budget(profile_result: dict[str, float]) -> None:
    assert profile_result["wall_seconds"] <= approved_budget("round_trips", "medium")
```

**Step 2: Verify current semantics and expose the measured budget**

Run:

```sh
python -m pytest -o addopts='' tests/test_factor_analysis/test_calendar.py tests/test_metrics/test_round_trips.py tests/performance/test_hotspot_regressions.py -q
```

Expected: current behavior is locked; budget is expected-to-fail only in the dedicated performance branch until an approved target exists.

**Step 3: Implement smallest measured improvement**

- calendar.py: remove non-vectorized DateOffset only if calendar output, timezone and labels are identical.
- data.py: replace groupby.apply only if factor bins, loss accounting, ordering and nullable behavior remain identical.
- rolling.py: use shared vectorized moments only for semantically equivalent volatility/Sortino calculations; preserve ndarray, Series, out-buffer and strict Empyrical projections.
- round_trips.py: accumulate results and concatenate once only if profile proves loop concat dominant; retain FIFO, signed quantity and partial-close semantics.
- Never combine all hotspots in one pull request.

**Step 4: Verify correctness, performance, AND compat**

Run:

```sh
MPLBACKEND=Agg python -m pytest -o addopts='' tests/compat tests/test_factor_analysis tests/test_metrics tests/performance -q --maxfail=0
python scripts/profile_hotspots.py --scenario medium --output build/hotspots-after.json
```

Expected: no strict regression; output digest is unchanged; time/RSS improve or remain within approved noise. Compatibility suite (C0-C4) stays green.

**Step 5: Commit one hotspot at a time**

```sh
git add fincore/factor_analysis/calendar.py tests/test_factor_analysis/test_calendar.py tests/performance/test_hotspot_regressions.py
git commit -m "perf: vectorize factor calendar offsets"
```

Repeat with explicit owned paths for each independently measured hotspot.

**Exit criteria:** No optimization is accepted solely on speed; it preserves numeric/shape semantics and improves a profile corpus case.

---

### Task 9: 为 enhanced 报告建立无敏感数据泄漏的审计 manifest

**Owner:** Track C

**Files:**

- Create: fincore/report/provenance.py
- Create: tests/test_report/test_provenance.py
- Modify: fincore/report/model.py
- Modify: fincore/report/artifacts.py
- Modify: fincore/report/__init__.py
- Modify: fincore/report/render_html.py
- Modify: mkdocs_docs/api/report.md

**注意：** 现有 `fincore/report/compute.py` 和 `fincore/report/format.py` 不需要修改，manifest 信息由 `model.py` 中的 ReportModel 收集，`provenance.py` 负责序列化。

**Step 1: Write a failing report-manifest test**

```python
def test_audit_manifest_contains_hashes_not_raw_returns(tmp_path: Path) -> None:
    result = create_strategy_report(returns(), output=str(tmp_path / "report.html"), return_result=True, audit_manifest=True)
    manifest = json.loads(result.manifest_path.read_text())
    assert manifest["inputs"]["returns"]["sha256"]
    assert "0.001" not in result.manifest_path.read_text()
```

**Step 2: Verify audit option is absent**

Run:

```sh
python -m pytest -o addopts='' tests/test_report/test_provenance.py -q
```

Expected: FAIL because ReportProvenance and audit_manifest do not exist.

**Step 3: Implement opt-in provenance without changing legacy returns**

- ReportProvenance has schema version, code commit/version, dependencies, normalized calculation configuration, input shapes/time bounds/content hashes and optional DataSnapshot references.
- Default create_strategy_report behavior stays unchanged. Only enhanced return_result=True plus audit_manifest=True creates a sidecar JSON in caller-selected output directory.
- Renderers consume precomputed ReportModel and do not mutate it or recompute. Manifest excludes raw returns, positions, transactions, credentials and absolute local paths.

**Step 4: Run report and PDF cleanup tests**

Run:

```sh
python -m pytest -o addopts='' tests/test_report -q --maxfail=0
```

Expected: HTML/PDF retains current result contracts; manifests are deterministic except documented generation time.

**Step 5: Commit owned files**

```sh
git add fincore/report tests/test_report mkdocs_docs/api/report.md
git commit -m "feat: add audit manifests to enhanced reports"
```

**Exit criteria:** A report traces to input identities, configuration and code without copying sensitive data into the artifact.

---

### Task 10: 分层加强类型契约与最低/最新依赖兼容性

**Owner:** Track A

**Files:**

- Create: constraints/minimum.txt
- Create: constraints/latest.txt
- Create: scripts/check_dependency_matrix.py
- Create: tests/quality/test_dependency_matrix.py
- Modify: pyproject.toml
- Modify: .github/workflows/ci.yml
- Modify: fincore/contracts/validation.py
- Modify: fincore/risk/models.py（Task 4 产出，本 Task 只加类型注解）
- Modify: fincore/data/contracts.py（Task 3 产出，本 Task 只加类型注解）

**已知问题处理：** 当前 yfinance/akshare 的 curl_cffi→OpenSSL 兼容性问题（0.2 节 P0 项）必须在 constraints 文件中明确排除已知不兼容版本组合。constraints/minimum.txt 应使用经验证可工作的版本组合。

**Step 1: Write failing matrix and typed-boundary tests**

```python
def test_constraints_cover_each_supported_python_environment() -> None:
    matrix = load_matrix(ROOT / "constraints")
    assert matrix["minimum"]["pandas"]
    assert matrix["latest"]["pandas"]
    assert matrix["minimum"]["numpy"] <= matrix["latest"]["numpy"]

def test_yfinance_import_probe_succeeds_with_minimum_constraints() -> None:
    """The minimum constraints must resolve to a working yfinance import."""
    result = probe_import("yfinance", constraints=ROOT / "constraints" / "minimum.txt")
    assert result.success, f"yfinance import failed: {result.error}"
```

**Step 2: Verify checker does not exist**

Run:

```sh
python -m pytest -o addopts='' tests/quality/test_dependency_matrix.py -q
```

Expected: FAIL because constraints and checker are absent.

**Step 3: Implement staged policy, not a whole-repository strict flip**

- Define supported Python/dependency combinations from pyproject metadata and exercise minimum plus latest resolvers in CI.
- Add separate optional-extra constraints/import probes for yfinance, akshare and their transitive HTTP/TLS stack. A package merely resolving in pip is insufficient; its import must pass in a pristine interpreter.
- Use mypy --check-untyped-defs first for contracts, risk and data; add annotations and narrow Any boundaries in those owned modules before widening scope.
- Keep strict façades callable with pinned signatures. Type improvements cannot change runtime validation/profile behavior.
- Reject floors that cannot install or pass the chosen Python matrix.

**Step 4: Run staged static and matrix gates**

Run:

```sh
python scripts/check_dependency_matrix.py --constraints constraints/minimum.txt
python -m mypy --check-untyped-defs fincore/contracts fincore/risk fincore/data
python -m pytest -o addopts='' tests/quality/test_dependency_matrix.py -q
```

Expected: selected typed boundary is clean; CI runs both constraints without silently resolving only latest packages.

**Step 5: Commit owned files**

```sh
git add constraints scripts/check_dependency_matrix.py tests/quality/test_dependency_matrix.py pyproject.toml .github/workflows/ci.yml fincore/contracts fincore/risk fincore/data
git commit -m "ci: validate supported dependency and type boundaries"
```

**Exit criteria:** Declared support range is executable evidence, and type checks cover public contracts rather than annotated islands.

---

### Task 11: 完成供应链、许可证、wheel 和 release evidence 收口

**Owner:** Track E with human legal/release reviewer

**Files:**

- Create: THIRD_PARTY_NOTICES.md
- Create: docs/quality/license-review.md
- Create: scripts/check_notices.py
- Create: tests/packaging/test_notices.py
- Modify: pyproject.toml
- Modify: .github/workflows/ci.yml
- Modify: .github/workflows/publish.yml
- Modify: docs/quality/release-candidate-checklist.md

**Step 1: Write a failing notice/provenance test**

```python
def test_copied_or_adapted_component_has_notice_and_license_status() -> None:
    notices = load_notices(ROOT / "THIRD_PARTY_NOTICES.md")
    assert notices["alphalens"]["review_status"] in {"pending-human-review", "approved"}
    assert notices["alphalens"]["source_commit"]

def test_empyrical_notice_records_pinned_commit() -> None:
    notices = load_notices(ROOT / "THIRD_PARTY_NOTICES.md")
    assert notices["empyrical"]["source_commit"] == "74655e974ed2935563820c548c339731f1fe0621"
```

**Step 2: Verify no checker validates notices**

Run:

```sh
python -m pytest -o addopts='' tests/packaging/test_notices.py -q
```

Expected: FAIL because notice checker is absent.

**Step 3: Implement evidence, not legal conclusion**

- Inventory imported/adapted Empyrical (commit `74655e9`), Pyfolio (commit `724bbd7`) and Alphalens (commit `3fa17ad`) paths, pinned commits, headers, license references and human-review status. 使用已冻结的本地快照 commit，与 convergence/alphalens 计划一致。
- Add reproducible SBOM/dependency audit command and attach artifact to release CI. Fail malformed/missing inventory, not unresolved legal judgment.
- Retain fresh-wheel profile tests and add notice/SBOM verification before PyPI publish.
- Human reviewer records whether NOTICE/SPDX/header changes are approved; code does not self-certify legal compliance.

**Step 4: Run packaging and release-consistency suite**

Run:

```sh
python -m pytest -o addopts='' tests/packaging -q --maxfail=0
python scripts/check_notices.py
```

Expected: distribution, notices and provenance agree; human-review gaps remain visible release blockers.

**Step 5: Commit owned files**

```sh
git add THIRD_PARTY_NOTICES.md docs/quality/license-review.md scripts/check_notices.py tests/packaging/test_notices.py pyproject.toml .github/workflows/ci.yml .github/workflows/publish.yml docs/quality/release-candidate-checklist.md
git commit -m "build: add release provenance and notice gates"
```

**Exit criteria:** Release candidates have package, dependency and source-provenance evidence; unresolved manual license decisions are explicit blockers.

---

### Task 12: 将真实能力、教程和发布决策同步到公开文档

**Owner:** Track A, all domain owners review

**Files:**

- Create: mkdocs_docs/guide/reproducible-research.md
- Create: mkdocs_docs/guide/performance.md
- Create: tests/docs/test_documented_examples.py
- Modify: mkdocs.yml
- Modify: README.md
- Modify: CHANGELOG.md
- Modify: mkdocs_docs/development/compatibility.md
- Modify: mkdocs_docs/development/api-stability.md
- Modify: docs/quality/release-candidate-checklist.md

**依赖：** 需要 Task 2（能力状态）、Task 3（provider contract）、Task 4（风险回测）、Task 9（审计 manifest）的产出作为文档内容来源。

**Step 1: Write a failing executable documentation example**

```python
def test_risk_validation_quickstart_runs_without_network() -> None:
    namespace = run_markdown_example(DOCS / "guide" / "risk-validation.md", name="minimal-backtest")
    assert namespace["result"].observations > 0
```

**Step 2: Verify named example is absent**

Run:

```sh
python -m pytest -o addopts='' tests/docs/test_documented_examples.py -q
```

Expected: FAIL because named code block and runner are absent.

**Step 3: Document only proven contracts**

- Publish end-to-end offline examples for performance analysis, provider injection/data snapshots, risk backtesting, factor analysis and audit manifest generation.
- State Task 2 statuses and link each public capability to its contract; distinguish strict compatibility from enhanced APIs.
- Replace stale coverage, maturity and performance claims with generated snapshot/approved-baseline references. Do not copy counts into prose.
- Add changelog entries for intentional divergences, deprecations and newly implemented Brinson/provider functionality.

**Step 4: Run docs, examples and full release candidate gates**

Run:

```sh
python -m pytest -o addopts='' tests/docs tests/quality tests/packaging -q --maxfail=0
python -m mkdocs build --strict
python scripts/check_quality_snapshot.py --snapshot docs/quality/current-baseline.json
```

Expected: examples run offline, docs build strictly and each release claim links to current evidence.

**Step 5: Commit owned files and conduct release review**

```sh
git add README.md CHANGELOG.md mkdocs.yml mkdocs_docs tests/docs docs/quality/release-candidate-checklist.md
git commit -m "docs: publish verified analytics platform workflows"
```

**Exit criteria:** Users can run important workflows without ambiguous capability claims, and release messaging says no more than evidence supports.

---

## 2. 阶段门槛与最终验收

### Gate A: 可信事实基础

- Task 1 clean snapshot passes on the release commit, branch coverage >= 60%.
- Task 2 capability inventory renders and has no undocumented public status.
- No baseline, benchmark or docs page is accepted with dirty=true or a different source commit.

### Gate B: 金融研究与风控可信

- Task 3 provider/data snapshot contracts pass without live network.
- Task 3 compat suite (`tests/compat`) remains green.
- Task 4 risk results and backtests cover alignment, sign, horizon, small-sample and inconclusive states.
- Task 4 compat suite (`tests/compat`) remains green.
- Task 5 property/oracle suites pass alongside strict compatibility tests.

### Gate C: 性能可信

- Task 6 has matching approved platform baselines, digest-before-performance comparison and release-build dependency.
- Task 7 before profiles exist for selected workflows.
- Task 8 accepts only reviewed, numerically equivalent improvements.
- Task 8 compat suite (`tests/compat`) remains green.

### Gate D: 产品与发行可信

- Task 9 report manifest is opt-in, privacy-preserving and tested.
- Task 10 minimum/latest dependency matrix and staged type gate pass. yfinance/akshare import probe succeeds with minimum constraints.
- Task 11 fresh wheel, notices/SBOM and manual review status are complete.
- Task 12 strict docs build and executable examples pass.

### Final acceptance command set

以下命令使用 Python 环境（不硬编码 Anaconda 路径，适配不同开发环境）：

```sh
# 1. 核心测试套件（不含 slow/integration/benchmarks）
MPLBACKEND=Agg python -m pytest -o addopts='' tests/ -q --tb=short --maxfail=0 -m "not slow and not integration" --ignore=tests/benchmarks

# 2. 兼容、属性、契约、打包和文档测试
MPLBACKEND=Agg python -m pytest -o addopts='' tests/compat tests/property tests/contracts tests/packaging tests/docs -q --maxfail=0

# 3. 静态检查
python -m ruff check fincore/ tests/ scripts/ examples/ benchmarks/
python -m ruff format --check fincore/ tests/ scripts/ examples/ benchmarks/
python -m mypy fincore --ignore-missing-imports

# 4. 文档构建
python -m mkdocs build --strict

# 5. 质量快照验证
python scripts/check_quality_snapshot.py --snapshot docs/quality/current-baseline.json
```

Release owner also executes configured sdist/wheel, installed-consumer, SBOM/notice and matching-platform benchmark gates in CI. A local green focused suite is diagnostic evidence, not a release declaration.

## 3. Scope-control rules for implementers

1. Use one branch/worktree per Track. Do not revert or stage unrelated shared-worktree changes.
2. Start each task with the failing test above or an equivalent stronger test; never change an expected value merely to turn a test green.
3. Any strict façade change requires frozen manifest suite, enhanced kernel suite and fresh wheel consumer check.
4. Any numerical optimization publishes before/after profile JSON, digest equality or reviewed semantic change; never compare across unmatched platforms.
5. Any provider is testable with fake in-memory transport and makes timeouts, retries, partial results, as-of semantics and adjustment explicit.
6. Any compatibility, performance, license or release claim links to generated or reviewed evidence; historical snapshots remain historical.
7. **Compat gate requirement:** Any task that modifies existing modules in `fincore/metrics/`, `fincore/risk/`, `fincore/data/`, `fincore/attribution/`, `fincore/factor_analysis/`, or `fincore/report/` must run `tests/compat` as part of its Step 4 verification.
8. **CI coordination:** When modifying `.github/workflows/ci.yml`, add new jobs at the end; do not reorder or rename existing jobs. Track A owns final CI integration before Task 12.
9. **Partial deployment:** Each Task is independently deliverable. If a Task's scope proves larger than estimated, deliver the completed portion and file a follow-up issue for the remainder.
10. **Portability:** All commands in this plan use `python -m pytest` (not hardcoded conda paths). Implementers use their own environment activation.