# Fincore 0.5 Unified Core Rewrite Implementation Plan

> **For Codex:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 在允许旧 API、导入路径和类模型发生破坏性变化的前提下，把 Fincore 直接重构为一个统一的新版本；所有现有分析、数据、图表和报告能力必须迁移到唯一 canonical 实现，生产 Python 代码量至少净减少 12%。

**Architecture:** 不创建并存的 fincore.v2，也不发布 Empyrical、Pyfolio、Alphalens 兼容 shim。新版本以领域 kernel 为唯一实现，以 runtime 统一输入、Catalog、执行、结果和 artifact，以 capability ledger 与 golden scenarios 证明功能零缺失；旧 façade、strict profile、registry、动态 wrapper 和兼容测试在对应能力迁移后整套删除。

**Tech Stack:** Python 3.11–3.14、NumPy、pandas、SciPy、statsmodels、Matplotlib、Plotly/Bokeh、pytest、Hypothesis、Ruff、mypy、setuptools。

---

## 0. 决策、版本和边界

### 0.1 迭代身份

- 迭代编号：0043。
- 状态：Proposed，等待执行授权。
- 目标版本：0.5.0.dev0。当前项目仍为 Beta；本迭代不以 1.0、Stable 或 Production Ready 为目标。
- 本文覆盖并替换此前 0043 中“保留旧 API 和 strict façade”的方案。
- 本轮只修改本计划文档与开发准入报告，不修改生产代码、不提交、不推送。
- 本文件是 0043 唯一 canonical 计划；当 0041_LANDED_SHA 满足且 0043 转为 Active 时，只在 docs/迭代计划/README.md 增加指针，不复制第二份 numbered 计划。
- 当前工作树包含 0041 治理和 CI 的未提交改动。0043 不能只接受口头 handoff：0041 必须先落到一个命名 commit/ref，记录 0041_LANDED_SHA；0043 clean base 必须经 git merge-base --is-ancestor 证明是其后代，之后才能开始 D0。
- AGENTS.md 引用的 .joyincode/rules/backend.md 与 frontend.md 当前缺失。D0 必须恢复这些规则或取得明确 waiver，不能在未知规范下启动业务重构。

### 0.2 明确放弃的兼容对象

以下内容不再属于验收契约，可直接删除或重新设计：

1. fincore.empyrical、fincore.pyfolio、fincore.alphalens 及根级 flat metric API。
2. Empyrical、Pyfolio 类、类方法、MRO、descriptor、reload/monkeypatch 形状和 state binding 方式。
3. 旧 import path、__all__、dir()、函数签名、参数顺序、默认值和 __module__。
4. strict-* / enhanced-* profile、PublicBinding 多路径、adapter_ref 和 result projection。
5. 旧 extras 名 pyfolio、alphalens、factor-analysis、datareader，以及兼容 alias。
6. 精确异常类名/消息/优先级、warning/stdout 文案、None projection、out= identity 等上游兼容怪癖。
7. C0/C1 import/signature 兼容门、旧 public API digest、旧 nodeid 超集和 source/wheel 旧 API snapshot 相等。

旧路径在新 wheel 中必须不可导入；这是 0043 成功条件，不是回归。

### 0.3 “功能不减少”的可执行定义

“功能”指用户能够获得的金融计算、数据变换、模型、图表、报告、扩展或执行能力，不等同于旧 API 的表达方式。D0 将每项能力登记为稳定 capability_id；之后路径可以变化，但 capability_id 和 scenario_id 不得未经批准删除。

对每个 required capability，候选版本必须满足：

1. 至少一个新 Catalog operation_id 可解析到唯一 canonical callable。
2. 公式、单位、方向、有效输入域和随机种子策略保持。
3. 数值结果在审核后的 rtol/atol 内一致。
4. Series/DataFrame 的值、shape、index、columns、name、order、timezone 和关键 dtype 保持；允许更换旧容器类名。
5. 输入不被修改；缓存键、随机性、provider 注入和并发隔离不退化。
6. 图表保留相同信息、数据序列、表格、legend 和资源 ownership；不要求像素或 Matplotlib artist 身份完全相同。
7. 报告保留章节、指标、单位、表格、离线 HTML、PDF 内容、provenance 和 artifact 生命周期；不要求 byte-for-byte 相同。
8. 缺少 optional dependency 时仍 lazy、可操作地失败；未调用可选能力时不得导入重依赖。
9. source tree 和同一个候选 wheel 都能执行该能力。
10. 旧 callable 若只是 alias 或 compatibility quirk，可标为 alias_only / legacy_quirk；必须有等价执行证据或“没有独立 observable”的证明，并同时获得 product owner 与独立 reviewer 批准，不能静默丢弃。

业务错误统一为 invalid_input、insufficient_data、unsupported、optional_dependency_missing、numerical_failure、convergence_failure、provider_failure、execution_failure、resource_failure 九类；只冻结类别和可操作诊断字段，不冻结旧异常类名或文案。

### 0.4 非目标

- 不新增金融指标、因子模型、风险模型或报告章节。
- 不重写已经有独立数值 oracle 的算法。
- 不通过缩短变量名、压行、生成大段 Python、删除测试/文档或把实现搬入 JSON 达成减码。
- 不保留双运行时、弃用期、兼容包或隐藏 sys.modules alias。
- 不发布 PyPI、不创建 release/version tag。唯一例外是 hosted D-1 经单独授权预建的非发布技术 tag `fincore-0043-bootstrap-dispatch-v1`，它只用于不可变 workflow dispatch，不能作为版本或发布标记。不自动更改远端分支保护或 Ruleset；bootstrap branch、dispatch tag、collision-branch ruleset与D8b merge enforcement都必须分别取得用户对具体ref/rule的授权，否则停在对应gate并保持BLOCKED。
- 不自动删除 LICENSE、NOTICE、THIRD_PARTY_NOTICES.md 或第三方许可证；即使旧模块删除，派生 kernel 的 provenance 仍需人工复核。

## 1. 当前事实与重构机会

### 1.1 规模基线

当前 fincore/**/*.py：

| 指标 | 当前值 |
| --- | ---: |
| Python 文件 | 153 |
| physical LOC | 49,412 |
| 非空非注释 logical LOC | 约 40,079 |
| 函数/方法 | 约 1,536 |
| OperationCatalog | 176 definitions / 257 bindings |

约 7,703 PLOC 属于可删除或替换的 façade、兼容合同、重复 registry 和动态分派。核心候选包括：

| 集群 | 约 PLOC | 处理 |
| --- | ---: | --- |
| empyrical.py、_empyrical_legacy.py | 854 | 迁移唯一组合公式后删除 |
| pyfolio.py、_pyfolio_impl.py | 1,320 | 工作流落入 portfolio/report 后删除 |
| alphalens/* | 3,032 | 迁移非同名能力到 factor_analysis 后删除 |
| _registry.py、_dispatch.py | 1,122 | 用 canonical-only runtime 替换后删除 |
| api/builtins.py、capabilities.py | 577 | 用单一 operation metadata 替换 |
| compat/profile/workflow/factor contracts | 798 | 迁移必要 schema 后删除 |

这些是 gross 候选，不能直接整批删除：当前 report、tearsheets、utils、context 和 8 个 metrics 模块仍依赖旧 façade/dispatch。

### 1.2 当前“像三个项目集合”的根因

1. 根包同时暴露 flat API、Empyrical 类、Pyfolio 类、Alphalens 子包和新 runtime，用户必须理解多个产品模型。
2. 旧 registry 是实际 source of truth，OperationCatalog 只是反向投影；capabilities.py 又维护另一份清单。
3. 257 个 binding 中大量只是同一 operation 在 module/class/root/context 上的重复路径。
4. metrics 模块通过 module __class__ monkeypatch 注入验证和 projection，真实调用路径难以追踪。
5. Pyfolio OO 层大部分是对 tearsheets/metrics 的手工委托。
6. alphalens 与 factor_analysis 有 49 个同名公开函数；另有 12 个能力可以明确迁入 enhanced kernel。
7. ReportArtifacts、factor artifacts、results.ArtifactBundle、model codec 和 report provenance 多轨并存。
8. report/compute.py 和 tearsheets 仍通过 Empyrical/Pyfolio 访问已有 metrics kernel。
9. 现有 public snapshot 246 项中 244 项 kind=unknown，且静默漏掉 fincore.metrics，不能证明功能完整。
10. capabilities.py 只有 25 行且漏掉 metrics、performance、simulation、optimization、plugin；OperationCatalog 也只覆盖 metrics/performance/report。

### 1.3 本轮实测证据

- Catalog/API/manifest/adapter 聚焦测试：65 passed，1 skipped。
- 排除 pytester/rerun 元测试后，旧三套 strict 核心兼容测试：1232 passed，1 skipped。
- 完整 tests/compat：1373 passed，1 skipped，12 failed；12 项均由受限环境中 pytest-rerunfailures 子进程绑定 socket 失败引起，因此不能登记为完整绿色基线。
- tests/compat 约 3.2 MB、34 个测试文件、约 485 个测试函数；其中包含大量真正的数值、标签、图表和工作流断言，不能先删。
- 当前 docs/quality/current-baseline.md 来自旧 commit，branch coverage 45.0%；任何 97% 或历史 100% 文案都不能作为 0043 证据。

## 2. 方案比较

| 方案 | 做法 | 优点 | 风险 | 结论 |
| --- | --- | --- | --- | --- |
| A. 原地 breaking cutover，按能力分批 | 现有 fincore 直接变成 0.5；先冻结 capability/golden，逐域迁移能力，最后在 T7 原子删除全部旧外壳。 | 不发布 shim；可复用可靠 kernel；领域迁移可独立回滚；最终只有一套代码。 | 实施期需严格控制内部依赖顺序，T7 不可拆分落地。 | 推荐 |
| B. 新建 fincore.v2 并保留旧包 | 新旧 API 并存一个或多个版本。 | 迁移用户风险较低。 | 与用户要求冲突；代码更多；双测试/双文档长期存在。 | 拒绝 |
| C. 绿地重写所有算法 | 只保留测试数据，从零实现统一库。 | 目录最整齐。 | 数值、NaN、时区、图表和报告回归风险最高；耗时不可控。 | 拒绝 |

选择 A。“直接修改成新的”表示最终 wheel 不包含兼容层，不表示跳过能力盘点和数值证明。T2–T6 的实现分支可短暂同时存在 old/new source，但必须保持完整非在线测试绿色，且不得作为 release candidate 发布；T7 在同一个 atomic cutover 中删除全部旧路径、旧测试与治理引用。

## 3. 目标架构

### 3.1 最终包拓扑

~~~text
fincore/
  __init__.py                # version 与领域模块；无 flat functions/classes
  runtime/
    specs.py                 # 单一 OperationSpec
    catalog.py               # operation_id -> canonical callable
    data.py                  # immutable analysis inputs/snapshots
    engine.py                # plan/run/batch
    session.py               # generic state/cache，替代 Empyrical/Pyfolio state
    results.py               # Result/metadata/codec
    artifacts.py             # artifact ownership/lifecycle
    validation.py            # canonical boundary validation
  metrics/                   # 纯时间序列与统计 kernel
  portfolio/                 # positions/transactions/round trips/perf attribution
  performance/               # TWR/MWR/XIRR/cashflow/inference/disclosures
  factor_analysis/           # 唯一因子 prepare/model/analyze/render kernel
  risk/
  attribution/
  optimization/
  simulation/
  report/
    portfolio/               # portfolio report builders/render models
    factor/                  # factor report builders/render models
    renderers/               # matplotlib/html/pdf/xlsx/interactive
  data/
  extensions/               # plugin + hooks
  viz/                       # backend primitives
  exceptions.py
~~~

最终必须不存在：

- fincore/empyrical.py
- fincore/pyfolio.py
- fincore/alphalens/
- fincore/_empyrical_legacy.py
- fincore/_pyfolio_impl.py
- fincore/_registry.py
- fincore/_dispatch.py
- fincore/_compat/
- strict profile、legacy registry、dynamic module monkeypatch 和 compatibility adapter

api、core、contracts、results 中可复用的通用模型迁入 runtime；领域合同与领域代码同目录，跨领域 schema 放入 runtime.specs/runtime.data。最终不保留顶层 contracts 兼容模块。

下列容易遗漏的 support/public clusters 也必须逐文件 disposition；D1 生成的 module ledger 要为每个 fincore/**/*.py 记录 owner、target、keep/move/delete、consumer count 和 capability_ids，0 unmapped：

| 当前模块 | Owner | 最终落位 / 处理 |
| --- | --- | --- |
| backends/numpy_backend.py、_types.py | Runtime | 迁入 runtime backend/specs；验证 backend selection、type/schema 后删除旧路径 |
| constants/periods.py、interesting_periods.py | Metrics + Portfolio | 频率/年化常量与 timing 场景进入对应领域；不保留公共 constants 聚合层 |
| constants/color.py、style.py | Reporting | 迁入 viz/report styles；保留表格、色彩、阈值 observable 后删除旧路径 |
| utils/math_utils.py | Metrics | 数值 helper 私有化到 metrics numeric primitives；由 numerical/property oracle 覆盖 |
| utils/date_utils.py、utils/deprecate.py | Reporting（T6 cleanup owner） | timer/stdout/deprecation 无独立 observable 时以 approved legacy_quirk 删除；若 D1 发现真实诊断能力则迁入 runtime diagnostics。Acceptance/Packaging 在 D1 签署 disposition 后把删除权限显式 handoff 给 T6 |
| utils/data_utils.py | Metrics（T2 owner） | rolling/up/down/array helper 私有化到 metrics primitives；冻结数值、index、dtype 与空窗口语义 |
| utils/common_utils.py | Portfolio + Reporting | 按 print/table/export/legend/asset 职责拆到 portfolio/report；不建立新 catch-all utils |
| data/contracts.py、data/snapshots.py | Runtime + Data | 通用 snapshot 进入 runtime.data，provider schema 留在 data models |
| exceptions.py | Runtime | 保留为九类 error algebra 的唯一公共错误入口，领域错误携带 category/details |

其他领域 support/private 文件默认归对应 domain owner，但也必须出现在 module ledger；不能因未出现在旧 __all__ 或 Catalog 中而跳过。

### 3.2 新公开模型

新版本只提供三类入口：

1. 领域函数：公开路径就是实现路径，例如 fincore.metrics.ratios.sharpe_ratio、fincore.risk.models.forecast_var。
2. runtime 执行：fincore.runtime.run(operation_id, inputs, config) 与 batch；用于 Catalog、DAG、缓存和统一结果。
3. 领域工作流：portfolio、factor_analysis、risk、report 的显式 builder/analyze/render 函数。

不在根包重导出几百个函数，不生成 module/class wrapper。OperationSpec 与 canonical function 共址，每项只记录：

- operation_id
- capability_id
- domain
- callable
- input/output schema
- optional extra
- determinism/RNG policy
- provenance

删除 semantic_profile、PublicBinding、signature text、adapter_ref、projection 和 deprecated aliases。Catalog 通过各领域 operations() 聚合实际 callable；任何 operation_id 必须恰好解析一次。

### 3.3 能力账本

D0 ledger 至少覆盖下列 family：

| family | 必须保留的能力 |
| --- | --- |
| metrics | returns/compounding、ratios、alpha-beta/alignment、drawdown、rolling、risk、stats、consecutive、timing、bayesian |
| portfolio | positions、transactions、round trips、perf attribution、perf stats、capacity |
| performance | TWR、MWR、XIRR、cashflow/fee/FX、inference、disclosure |
| risk | EVT、GARCH family、VaR/ES forecast、backtest、calibration、walk-forward、report |
| simulation | Monte Carlo、GBM、bootstrap、scenario、path sampling |
| attribution | Brinson、Fama-French、style、provider injection/cache |
| optimization | frontier、generic objectives、risk parity、feasibility |
| factor | prepare/forward/calendar、quantile/weights/returns/IC/turnover、portfolio inputs、PIT/multihorizon、cost/capacity、inference/FDR/Fama-MacBeth、events、render/tear workflows |
| report | compute-once model、portfolio/factor builders、offline HTML、PDF、XLSX export、artifact、provenance |
| viz | Matplotlib、HTML、Plotly、Bokeh、caller axes/resource ownership |
| runtime | snapshot/session/cache/export、DAG/rolling、result codec/artifact lifecycle |
| data | provider contract、Yahoo/AlphaVantage/pandas-datareader/CN injected offline clients |
| extensions | registration、discovery、isolation、hooks |

Ledger 不从候选 Catalog 自证。legacy_surface_inventory 必须取以下并集：

1. root/domain/submodule 的 __all__ 和公开 definition；
2. METRIC_REGISTRY、WORKFLOW_REGISTRY、OperationCatalog、capabilities.py；
3. README/docs 中声明的公开路径；
4. tests 中所有非 private import、公开类成员与参数模式；
5. pyproject.toml 的 extras、package-data、entry points 和 wheel 内容契约；
6. examples 与 benchmarks 实际调用的能力和参数模式。

每项 legacy surface 必须映射 capability_id、alias_only 或 legacy_quirk。alias_only 必须有同输入等价执行证明；legacy_quirk 必须证明没有独立分析 observable；两者都需要 product owner 与独立 reviewer 批准。每个 required capability 的 scenario 集必须覆盖适用的 happy path、边界、错误、optional dependency/provider 分支和文档化参数模式，不能只留一个 smoke case。

family_id 只负责把多个 leaf capability 分组，不能被当成实现单元。D1 只冻结 planned_owner、target operation_ids 和 max_candidate_implementation_fingerprints=1，不预填尚不存在的 candidate fingerprint。D3–D8 由独立 acceptance script 从实际 callable 的 module/qualname/source AST 计算 implementation_fingerprint；每个 leaf capability_id 必须恰好一个。它可以暴露 1..N 个 operation_id，但仅限这些 operation 共享同一 canonical callable/fingerprint，或只是经批准的参数化 semantic mode。每个 operation_id 仍必须恰好解析到一个 callable，禁止由候选 Catalog 自报 fingerprint 来证明自身唯一。

### 3.4 数据流

~~~text
caller data / provider
        |
        v
runtime immutable snapshot + canonical validation
        |
        v
OperationCatalog -> planner -> canonical domain callable
        |
        +------> native scalar/Series/DataFrame/model
        |
        v
Result + metadata + artifacts
        |
        +------> portfolio/factor/risk workflows
        |
        v
report model -> matplotlib/html/pdf/interactive renderer
~~~

直接领域函数返回自然 Python/pandas/model 结果；runtime.run 统一包裹 Result。报告只消费 snapshot/result，不访问 session 私有字段，不通过 façade 反向调用 metrics。

### 3.5 量化目标

| 指标 | 强制门 | 目标 | Stretch |
| --- | ---: | ---: | ---: |
| fincore/**/*.py physical LOC | <= 43,480（-12%） | <= 42,000（-15%） | <= 40,520（-18%） |
| logical LOC | <= 35,270（-12%） | <= 34,070（-15%） | — |
| façade/registry/delegate normalized body | 至少 -60% | -80% | — |
| 旧 façade/strict/compat production 文件 | 0 | 0 | 0 |
| required capability_id/scenario_id 删除 | 0 | 0 | 0 |
| 每个 leaf capability 的 implementation_fingerprint 数 | 1 | 1 | 1 |
| 每个 operation 的 canonical callable 数 | 1 | 1 | 1 |
| branch coverage | >= 60%，且不低于 fresh D0 | 同左 | — |
| 性能回退 | median <=10%，p95 <=15%，RSS <=10% | 同左 | — |

测试、fixture、docs、生成物的删减不计入生产代码减码。每个 domain cleanup commit 的 production LOC 必须净负；若为保留真实能力需要暂时净增，必须在同一 milestone 的 consolidation commit 中偿还。

减码门同时使用“相对 D0”与上表当前参考值：candidate physical LOC 必须 <= floor(D0 physical LOC * 0.88)，并且在当前 49,412 PLOC 基线下 <=43,480；logical LOC 必须 <= floor(D0 logical LOC * 0.88)，并且在当前约 40,079 基线下 <=35,270。若 0041 落地后 D0 数值变化，以相对公式为权威，并在 D1 amendment 中更新参考数，不能放宽 12% 降幅。

architecture gate 的算法也在 D0 冻结：physical LOC 是 include manifest 中 fincore/**/*.py 的 splitlines 数；logical LOC 是 tokenize 后包含非空白、非 comment-only token 的物理行，docstring 计入；normalized duplicate fingerprint 对函数/方法 AST 去除位置、type comment 与 docstring，归一化局部标识符和字面量，但保留 call target、控制流和运算结构。baseline 必须记录 include/exclude manifest、schema_version 和脚本 SHA；tests/docs/assets/generated Python 均不计入减码。

## 4. 决策门与依赖

| Gate | 必须回答的问题 | 通过证据 |
| --- | --- | --- |
| D-1 Acceptance bootstrap | 三平台 D0 是否有独立且获授权的 hosted/self-hosted 触发与签名路径？ | hosted `0043_BOOTSTRAP_SHA` 或 self-hosted runner/signer contract、三平台 attestations |
| D0 Clean baseline | 基线 SHA、规则、工作树、依赖与 0041 ownership 是否清楚？ | clean worktree fingerprint、fresh quality snapshot |
| D1 Capability freeze | 是否枚举了全部能力而非只枚举旧 Catalog？ | immutable ledger、surface disposition、goldens |
| D2 API design approval | 0.5 namespace、operation ID 规则、extras、错误类别和 planned API fixture 是否已签署？ | ADR-0043、schema、namespace/extra mapping；此时不要求实现 snapshot |
| D3 Runtime ready | snapshot/catalog/run/result/artifact 是否可单独工作？ | runtime contracts、source/wheel tests |
| D4 Domain parity | 每个领域是否先迁语义测试再删除旧实现？ | per-domain parity report |
| D5 Report/viz parity | 图表、报告、资源 ownership 与离线性是否保留？ | normalized plot/report goldens |
| D6 Installed profiles | 同一 wheel 在所有 extras 组合中是否完成 ledger scenario？ | fresh-consumer matrix |
| D7 Implemented cutover | 旧路径是否消失、contract tests 是否完成迁移、新 API 是否与设计一致，且验收 head/base 未漂移？ | implemented 0.5 API snapshot、forbidden import graph、exact head/base ancestry |
| D8a Technical seal | capability/scenario 0 missing，质量和性能是否全绿？ | attested technical acceptance manifest |
| D8b Remote enforcement | 真实 PR 是否把唯一受信 final workflow/check 强制为无 bypass merge gate？ | D0/D8a-linked signed live Ruleset/workflow/check-run/PR snapshot |

~~~text
D-1 -> D0 -> D1 -> D2 -> D3
                  |
                  +--> T2 metrics/performance
                  +--> T3 portfolio
                  +--> T4 factor
                  +--> T5 risk/attribution/simulation/optimization/data/extensions
                              |
                              v
                       T6 report/result/viz
                              |
                              v
                       T7 contract migration + atomic cutover/deletion
                              |
                              v
                       T8 docs/quality/CI
                              |
                              v
                       T9 technical seal (D8a)
                              |
                              v
                       T10 remote enforcement (D8b)
~~~

验收权威规则：任务中的 candidate-checkout pytest、build、checker 命令用于快速定位；T1–T6 每个 tranche 的正式 handoff 还必须从 detached T0A acceptance worktree，以已验签的仓库外 D0 bundle 运行 frozen runner 的 `--tranche --families ...` 模式。expected bytes 只来自 bundle，candidate 只作为 subprocess actual；没有该 isolated evidence 的 tranche 不得交接。T7–T10 使用各任务写明的同一信任链。

### 4.1 Self-hosted 权威入口

self-hosted mode 不执行后文任何引用 `0043_BOOTSTRAP_SHA` 的 hosted block。T7、T8、T9/D8a、T10/D8b 及强制 merge-time recheck 分别设置 `FINCORE_0043_GATE=t7|t8|d8a|d8b|merge_recheck`。每次运行都由独立 acceptance owner 使用 D-1 预登记私钥从仓库外签署 gate-spec envelope；签名 payload 必须绑定 gate、actual head、expected base、artifact digest、单次 nonce、签发/过期时间和 signer fingerprint。spec 只能包含 actual checkout/artifact 路径和 T0A runner 允许的参数，所有 expected digest/SHA 必须来自 D0 anchor 或 D-1 runner contract；可由环境同时替换的裸 SHA256 不构成授权。以下完整入口先验证 acceptance-owner 公钥、gate-spec 签名、外部 verifier、runner/signer/schema 与撤销状态，再从已验签 payload 导出 gate/head/base/digest，随后验签 D0、检查 T0A→T0B→T0C→actual head/base，最后由外部 verifier 从 detached T0A 运行白名单 entrypoint；candidate 不能选择 checker、expected 文件或子命令：

    set -euo pipefail
    test "$D1_EXECUTION_MODE" = self_hosted
    test -n "$D1_TRUST_VERIFIER_PATH"
    test -n "$D1_TRUST_VERIFIER_SHA256"
    test -n "$D1_RUNNER_CONTRACT_PATH"
    test -n "$D1_RUNNER_CONTRACT_SHA256"
    test -n "$D1_SIGNER_PUBLIC_KEY_FINGERPRINT"
    test -n "$D1_ACCEPTANCE_OWNER_PUBLIC_KEY"
    test -n "$D1_ACCEPTANCE_OWNER_PUBLIC_KEY_FINGERPRINT"
    test -n "$D1_GATE_SPEC_SCHEMA_PATH"
    test -n "$D1_GATE_SPEC_SCHEMA_SHA256"
    test -n "$D1_AUTHORIZATION_LEDGER_CONTRACT_PATH"
    test -n "$D1_AUTHORIZATION_LEDGER_CONTRACT_SHA256"
    test -n "$D1_MERGE_RECEIPT_LEDGER_CONTRACT_PATH"
    test -n "$D1_MERGE_RECEIPT_LEDGER_CONTRACT_SHA256"
    test -n "$FINCORE_0043_GATE"
    test -n "$FINCORE_0043_SELF_HOSTED_GATE_SPEC"
    test -n "$FINCORE_0043_SELF_HOSTED_GATE_SPEC_SIGNATURE"
    test -n "$FINCORE_0043_SELF_HOSTED_EVIDENCE_DIR"
    test -n "$FINCORE_0043_D0_TRUST_DIR"
    test -n "$FINCORE_0043_D0_ANCHOR_SHA256"
    test -n "$FINCORE_0043_D0_EXPECTED_BUNDLE_SHA256"
    test -n "$T0A_TOOLING_SHA"
    test -n "$T0B_CAPABILITY_SHA"
    test -n "$T0C_BASELINE_SHA"
    test -n "$FINCORE_0043_ACTUAL_HEAD_SHA"
    test -n "$EXPECTED_PR_BASE_SHA"
    test "$(shasum -a 256 "$D1_TRUST_VERIFIER_PATH" | cut -d ' ' -f 1)" = "$D1_TRUST_VERIFIER_SHA256"
    "$D1_TRUST_VERIFIER_PATH" verify-and-exec --gate "$FINCORE_0043_GATE" --gate-spec "$FINCORE_0043_SELF_HOSTED_GATE_SPEC" --gate-spec-signature "$FINCORE_0043_SELF_HOSTED_GATE_SPEC_SIGNATURE" --acceptance-owner-public-key "$D1_ACCEPTANCE_OWNER_PUBLIC_KEY" --expected-acceptance-owner-fingerprint "$D1_ACCEPTANCE_OWNER_PUBLIC_KEY_FINGERPRINT" --gate-spec-schema "$D1_GATE_SPEC_SCHEMA_PATH" --expected-gate-spec-schema-sha256 "$D1_GATE_SPEC_SCHEMA_SHA256" --runner-contract "$D1_RUNNER_CONTRACT_PATH" --expected-runner-contract-sha256 "$D1_RUNNER_CONTRACT_SHA256" --expected-signer-fingerprint "$D1_SIGNER_PUBLIC_KEY_FINGERPRINT" --authorization-ledger-contract "$D1_AUTHORIZATION_LEDGER_CONTRACT_PATH" --expected-authorization-ledger-contract-sha256 "$D1_AUTHORIZATION_LEDGER_CONTRACT_SHA256" --merge-receipt-ledger-contract "$D1_MERGE_RECEIPT_LEDGER_CONTRACT_PATH" --expected-merge-receipt-ledger-contract-sha256 "$D1_MERGE_RECEIPT_LEDGER_CONTRACT_SHA256" --require-signed-gate "$FINCORE_0043_GATE" --require-signed-head "$FINCORE_0043_ACTUAL_HEAD_SHA" --require-signed-base "$EXPECTED_PR_BASE_SHA" --require-unexpired --require-atomic-consumption --require-service-attestation-match --trust-anchor "$FINCORE_0043_D0_TRUST_DIR/0043-d0-trust-anchor.json" --trust-attestation "$FINCORE_0043_D0_TRUST_DIR/0043-d0-attestation.json" --trusted-expected-bundle "$FINCORE_0043_D0_TRUST_DIR/0043-d0-expected-bundle.tar.zst" --expected-anchor-sha256 "$FINCORE_0043_D0_ANCHOR_SHA256" --expected-bundle-sha256 "$FINCORE_0043_D0_EXPECTED_BUNDLE_SHA256" --expected-tooling-sha "$T0A_TOOLING_SHA" --expected-capability-sha "$T0B_CAPABILITY_SHA" --expected-baseline-sha "$T0C_BASELINE_SHA" --actual-head "$FINCORE_0043_ACTUAL_HEAD_SHA" --expected-base-sha "$EXPECTED_PR_BASE_SHA" --require-full-history --require-t0c-head-ancestry --deny-revoked --deny-candidate-expected --sanitize-environment --python-isolated --output-trusted-shas "${FINCORE_0043_SELF_HOSTED_GATE_SPEC}.trusted-shas.json" --evidence-dir "$FINCORE_0043_SELF_HOSTED_EVIDENCE_DIR"

`verify-and-exec` 的 t7/t8 entrypoint 分别固定为 T0A `run_0043_cutover_gates.py` full/verify-evidence-only；d8a 固定为 `run_0043_final_gates.py` execute/verify 后由另一台隔离 signer runner 仅对 canonical verdict 做 attest；d8b 固定为 `run_0043_final_gates.py --verify-attested-final-evidence` 后接隔离 capture/verify/sign，merge_recheck 则每次 fresh hydrate bootstrap/D0/D8a/D8b/T0A 后输出短 TTL receipt。执行 runner 无 signer key、OIDC、repository secret 或管理 token；verify runner 从 content-addressed artifact store 按 post-execution handoff 的 ID/digest 下载证据，先用预登记 trust verifier认证bootstrap/D0/T0A，再运行 T0A verify-only checker；signer runner不 checkout 或执行 candidate/T0A，只核对 canonical verdict digest与workflow identity后签名。self-hosted signer 必须对 12 个 matrix、3 个 browser、D8a、D8b和merge receipt分别出具 envelope，字段与 hosted schema 相同；任一 gate spec 签名、signer、runner、隔离边界或 revocation 漂移即停止，不得回退到 hosted env 变量。

## 5. 实施任务

### Task -1: 落地独立的三平台 baseline bootstrap

**Owner:** Acceptance + Governance

**Depends on:** 共同前置是 0041 已落到命名 commit/ref、规则文件已恢复或 waiver 已签署。hosted mode 还必须取得明确的 workflow/默认分支变更授权，以及分别针对 `refs/heads/codex/0043-bootstrap-immutable`、exact非发布dispatch tag名称+目标SHA、tag ruleset ID/0-bypass/rollback、同名collision branch ruleset ID/0-bypass的具体授权；同时要完成 GitHub App、OIDC broker、GET-only API proxy、原子 authorization/merge-receipt ledgers 和 protected environment 的独立安全审批。任一精确远端授权缺失即停在D-1。self-hosted mode 不需要仓库 workflow/ref/rule 授权，但必须先完成仓库外 runner、verifier、acceptance-owner key、signer 与两类原子 ledger 的预登记和独立审批。

**Files（hosted mode）:**

- Create: .github/workflows/0043-baseline.yml
- Create: .github/workflows/0043-acceptance-cell.yml
- Create: .github/workflows/0043-final-seal.yml
- Create: .github/workflows/0043-remote-attestation.yml
- Create: .github/workflows/0043-merge-recheck.yml
- Create: scripts/verify_0043_bootstrap_trust.py
- Create: tests/quality/test_0043_baseline_workflow.py
- Create: tests/quality/test_0043_acceptance_workflows.py
- Create: tests/quality/test_0043_remote_attestation_workflow.py
- Create: tests/quality/test_0043_merge_recheck_workflow.py
- Create: tests/quality/test_verify_0043_bootstrap_trust.py

**External records（mode-specific）:**

- `D1_RUNNER_CONTRACT_SHA256`
- `D1_SIGNER_PUBLIC_KEY_FINGERPRINT`
- `D1_ACCEPTANCE_OWNER_PUBLIC_KEY` + fingerprint
- `D1_TRUST_VERIFIER_URI` + SHA256
- `D1_TRUST_VERIFIER_PATH`
- `D1_RUNNER_CONTRACT_PATH`
- `D1_ATTESTATION_DIR`
- `D1_GATE_SPEC_SCHEMA_URI`、`D1_GATE_SPEC_SCHEMA_PATH` + SHA256（覆盖 t7/t8/d8a/d8b/merge_recheck）
- `D1_AUTHORIZATION_LEDGER_CONTRACT_PATH` + SHA256
- `D1_MERGE_RECEIPT_LEDGER_CONTRACT_PATH` + SHA256
- `D1_TRUSTED_REQUESTER_URI`、`D1_TRUSTED_REQUESTER_PATH` + SHA256
- `D1_INPUT_LOADER_URI`、`D1_INPUT_LOADER_PATH` + SHA256
- `D1_VERDICT_VERIFIER_URI`、`D1_VERDICT_VERIFIER_PATH` + SHA256
- hosted `FINCORE_0043_BOOTSTRAP_BRANCH_REF=refs/heads/codex/0043-bootstrap-immutable`、immutable `FINCORE_0043_BOOTSTRAP_DISPATCH_TAG=fincore-0043-bootstrap-dispatch-v1`，以及exact collision ref `refs/heads/fincore-0043-bootstrap-dispatch-v1` + 三份0-bypass ruleset ID/digest
- `D1_BOOTSTRAP_VERIFIER_SHA256`
- hosted OIDC issuer/audience、broker identity/build digest、GitHub App ID、installation ID、repository allowlist、token permission/TTL/revocation contract
- hosted GET-only proxy identity/build digest与固定 GET endpoint allowlist；atomic authorization ledger identity/schema/consistency contract
- merge-receipt verifier/consumer identity、artifact digest、公钥与atomic receipt-ledger contract
- `D1_MERGE_RECEIPT_VERIFIER_PATH` + SHA256
- ubuntu/macos/windows runner identity + attestation IDs

**Steps:**

1. 在 D-1 签署互斥的 `D1_EXECUTION_MODE=hosted|self_hosted`，实施期不得混用。hosted mode 从 `0041_LANDED_SHA` 的 clean 后代创建仅含上述五个 workflow、bootstrap trust verifier及 tests 的 change；baseline workflow只接受完整40位 tooling/capability/baseline SHA、acceptance-lock digest 与 broker 签发的授权 capability，不接受可变 source ref。`0043-acceptance-cell.yml` 与 `0043-final-seal.yml` 是受信 reusable workflows；所有 caller 的 `uses:` 必须写成 `owner/repo/.github/workflows/<file>@<完整40位0043_BOOTSTRAP_SHA>`，禁止 branch/tag/短 SHA。caller identity 绑定 T8 head/base，called identity 单独绑定 `job_workflow_ref`/`job_workflow_sha=0043_BOOTSTRAP_SHA`，不得把 reusable run 的 `run.head_sha` 误当 called workflow SHA。
2. 每个 cell/final workflow 固定拆成 `preflight-fetch -> execute -> verify -> sign` 四类 job。preflight-fetch从job启动即只有`contents: read/actions: read/id-token: write/attestations: write/artifact-metadata: write`，可只读checkout exact candidate SHA但不执行candidate；它生成content-addressed full-git-bundle/source/dist/D0输入并用D-1锁定完整SHA的attest action认证execution spec，只绑定locked argv、raw artifact name/schema与fresh execution_nonce，不引用未来artifact ID。execute以`permissions: {}`、无secrets/OIDC/signer key运行input-loader；loader先用单次pre-signed read URL下载输入，验签后删除runtime credential，再在隔离容器运行candidate并上传raw artifact。raw upload完成后，平台action outputs与Actions API产生绑定nonce/run/job/artifact ID/digest的post-execution handoff。verify是另一台clean runner，只有`contents: read/actions: read`且无OIDC/attestation write；它先以exact bootstrap verifier认证D0并derive trusted T0A/tree/toolchain，再执行T0A verify-only，输出canonical verdict。sign是第四台clean runner，job启动时即具有`id-token: write/attestations: write/artifact-metadata: write/actions: read`，但不checkout或执行candidate/T0A，只核对content-addressed verdict和前序job identity后签名。workflow contract静态核对权限，D-1还用真实最小artifact做一次attest/verify smoke；四类job不共享workspace、runner、GitHub command files或candidate进程。self-hosted mode执行同一隔离拓扑。
3. `0043-remote-attestation.yml` 是 restricted capture-and-attest workflow；`0043-merge-recheck.yml` 是独立、短 TTL 的 merge-time recheck workflow。两者不 checkout 或执行 candidate，只接受 broker 已签发 capability 以及 D0/D8a/D8b immutable artifact ID/digest。D-1 冻结 OIDC issuer/audience、broker service identity与build digest、GitHub App/installation/repository allowlist、精确 permission、token TTL（最多300秒）、revocation和审计 schema。读取 Ruleset `bypass_actors` 所需 installation token 可具有 `administration: write`，但原始 token 只进入预登记 GET-only proxy；proxy 固定允许的 GET endpoint、拒绝所有非 GET/未列路径，不把 token 暴露给 workflow shell、candidate 或 child process，并要求响应显式包含完整 `bypass_actors` 字段。不得使用长期 PAT，也不得把“字段被权限隐藏”解释为零 bypass。
4. protected broker 从可信 approval/event identity 派生 repository、PR、head、base、actor ID、expiry 与唯一 `jti`，签发短时 capability；外部 strongly-consistent ledger 在 capture/sign 前以 `(repository, actor_id, jti)` 做原子 CAS `unused -> consumed`。只有 CAS winner 可继续，失败、并发、重放或过期 run 不生成 snapshot/attestation。envelope 必须包含可验消费 receipt、ledger record version/digest；测试并发 dispatch 时恰好一个 signer 成功。requester 不自报 actor/expiry，也拿不到 App/Actions token。
5. baseline workflow以 ubuntu/macos/windows + Python 3.11 matrix checkout输入SHA；browser下载与scenario network-deny分阶段执行。每个平台记录 source SHA、caller run/job、called `job_workflow_ref/job_workflow_sha` 或 self-hosted attestation ID、runner image、Python、Playwright/Chromium revision、browser artifact digest、命令/exit code和 record digest。所有第三方 actions（含 checkout/setup-python/upload-artifact/attest）和 gh CLI 版本固定完整 SHA/digest；两种 mode 都在 D-1 冻结完整 caller/reusable job graph、cell/final/remote/merge-recheck envelope schema及验签规则。
6. 仅 hosted mode：经 reviewer 验证 workflow fail-closed、job isolation、权限和 fork/ref 防护后，使用用户分别授权将 tooling-only change 落到默认分支、创建bootstrap branch、预建指向同一SHA的非发布dispatch tag及三份规则。bootstrap branch与exact tag各有0-bypass ruleset；第三份branch-target ruleset精确覆盖`refs/heads/fincore-0043-bootstrap-dispatch-v1`并Restrict creations/update/delete、0 bypass，因而同名branch从未存在。reusable `uses:`只使用完整40位SHA；`workflow_dispatch`只使用immutable tag。broker每次dispatch在同一受信前置中核对tag object、tag ruleset与collision-branch ruleset ID/digest/0-bypass后才触发，actual `run.head_sha`仍等于bootstrap SHA。授权还必须明确是否允许在T0a/T0b/T0c review后只推送exact tooling/capability/baseline SHA到专用immutable ref并触发workflow；未授权则每次远端动作前停下。这是“T8前不创建cutover PR”的唯一例外；不含0.5 production/cutover代码。self-hosted跳过本步骤。
7. self-hosted mode 不产生 `0043_BOOTSTRAP_SHA`：记录 runner-contract digest、acceptance-owner public key fingerprint、signer fingerprint、隔离执行/签名 runner IDs 与三平台 attestations；只有本机 macOS 输出不满足 D-1。hosted mode 必须记录默认分支上的 bootstrap SHA、workflow graph digest、broker/App/proxy/ledger contract；两套证据不可互相代填。

**Verify:**

Hosted mode：

    /Users/yunjinqi/opt/anaconda3/bin/conda run -n base python -m pytest -p no:rerunfailures -o addopts='' tests/quality/test_0043_baseline_workflow.py tests/quality/test_0043_acceptance_workflows.py tests/quality/test_0043_remote_attestation_workflow.py tests/quality/test_0043_merge_recheck_workflow.py tests/quality/test_verify_0043_bootstrap_trust.py -q --tb=short --maxfail=0

    set -euo pipefail
    test -n "$D1_HOSTED_SERVICE_CONTRACT"
    test -n "$D1_HOSTED_SERVICE_CONTRACT_SHA256"
    test -n "$D1_HOSTED_SERVICE_ATTESTATION_DIR"
    test -n "$FINCORE_0043_BOOTSTRAP_SHA"
    test -n "$D1_BOOTSTRAP_VERIFIER_SHA256"
    FINCORE_0043_BOOTSTRAP_VERIFY_ROOT=$(mktemp -d /tmp/fincore-0043-bootstrap-verify.XXXXXX)
    git worktree add --detach "$FINCORE_0043_BOOTSTRAP_VERIFY_ROOT/bootstrap" "$FINCORE_0043_BOOTSTRAP_SHA"
    test "$(shasum -a 256 "$FINCORE_0043_BOOTSTRAP_VERIFY_ROOT/bootstrap/scripts/verify_0043_bootstrap_trust.py" | cut -d ' ' -f 1)" = "$D1_BOOTSTRAP_VERIFIER_SHA256"
    /Users/yunjinqi/opt/anaconda3/bin/conda run -n base python -I "$FINCORE_0043_BOOTSTRAP_VERIFY_ROOT/bootstrap/scripts/verify_0043_bootstrap_trust.py" --mode hosted --service-contract "$D1_HOSTED_SERVICE_CONTRACT" --expected-service-contract-sha256 "$D1_HOSTED_SERVICE_CONTRACT_SHA256" --service-attestation-dir "$D1_HOSTED_SERVICE_ATTESTATION_DIR" --expected-bootstrap-sha "$FINCORE_0043_BOOTSTRAP_SHA" --require-oidc-issuer-audience --require-broker-build --require-app-installation --require-get-only-proxy --require-explicit-bypass-actors-field --require-authorization-ledger --require-merge-receipt-ledger --require-three-rulesets --require-collision-branch-creation-restricted --require-service-attestation-match --require-token-ttl-max-seconds 300 --run-live-cas-race --run-live-receipt-cas-race --expect-single-winner --deny-revoked --output /tmp/fincore-0043-hosted-bootstrap-verdict.json

Self-hosted mode：

    set -euo pipefail
    test -n "$D1_TRUST_VERIFIER_PATH"
    test -n "$D1_TRUST_VERIFIER_SHA256"
    test -n "$D1_RUNNER_CONTRACT_PATH"
    test -n "$D1_RUNNER_CONTRACT_SHA256"
    test -n "$D1_SIGNER_PUBLIC_KEY_FINGERPRINT"
    test -n "$D1_ACCEPTANCE_OWNER_PUBLIC_KEY"
    test -n "$D1_ACCEPTANCE_OWNER_PUBLIC_KEY_FINGERPRINT"
    test -n "$D1_ATTESTATION_DIR"
    test -n "$D1_AUTHORIZATION_LEDGER_CONTRACT_PATH"
    test -n "$D1_AUTHORIZATION_LEDGER_CONTRACT_SHA256"
    test -n "$D1_MERGE_RECEIPT_LEDGER_CONTRACT_PATH"
    test -n "$D1_MERGE_RECEIPT_LEDGER_CONTRACT_SHA256"
    test "$(shasum -a 256 "$D1_TRUST_VERIFIER_PATH" | cut -d ' ' -f 1)" = "$D1_TRUST_VERIFIER_SHA256"
    "$D1_TRUST_VERIFIER_PATH" verify-bootstrap --runner-contract "$D1_RUNNER_CONTRACT_PATH" --expected-runner-contract-sha256 "$D1_RUNNER_CONTRACT_SHA256" --expected-signer-fingerprint "$D1_SIGNER_PUBLIC_KEY_FINGERPRINT" --acceptance-owner-public-key "$D1_ACCEPTANCE_OWNER_PUBLIC_KEY" --expected-acceptance-owner-fingerprint "$D1_ACCEPTANCE_OWNER_PUBLIC_KEY_FINGERPRINT" --attestation-dir "$D1_ATTESTATION_DIR" --gate-spec-schema "$D1_GATE_SPEC_SCHEMA_PATH" --expected-gate-spec-schema-sha256 "$D1_GATE_SPEC_SCHEMA_SHA256" --authorization-ledger-contract "$D1_AUTHORIZATION_LEDGER_CONTRACT_PATH" --expected-authorization-ledger-contract-sha256 "$D1_AUTHORIZATION_LEDGER_CONTRACT_SHA256" --merge-receipt-ledger-contract "$D1_MERGE_RECEIPT_LEDGER_CONTRACT_PATH" --expected-merge-receipt-ledger-contract-sha256 "$D1_MERGE_RECEIPT_LEDGER_CONTRACT_SHA256" --require-platforms ubuntu macos windows --require-gates t7 t8 d8a d8b merge_recheck --require-execution-verifier-signer-isolation --require-signed-gate-spec --require-atomic-authorization-ledger --require-service-attestation-match --run-live-cas-race --run-live-receipt-cas-race --expect-single-winner --require-remote-envelope-schema --deny-revoked

**Acceptance:**

- hosted mode：`0043_BOOTSTRAP_SHA` 是 `0041_LANDED_SHA` 的后代且已在默认分支可见；受保护bootstrap branch、预建immutable dispatch tag及exact collision branch三份独立ruleset均已验ID/digest且0 bypass，tag禁止后续create/update/delete，collision branch规则禁止同名branch的create/update/delete。reusable calls固定40位SHA，dispatch只使用该tag并验证tag/collision rules及run SHA；caller与called workflow identity均可机器验证。
- candidate execution、verify与sign使用不同job/runner/credential boundary；sign不checkout/执行candidate或T0A。broker/App/proxy/authorization-ledger/merge-receipt-ledger contract已冻结并有live service attestation，两类ledger的并发CAS race都恰好一个winner；Ruleset响应显式含`bypass_actors`。
- self-hosted mode：runner-contract digest、外部 verifier digest、acceptance-owner key、gate-spec signature、signer fingerprint、隔离 runner IDs、三平台 attestation IDs、remote envelope schema与撤销状态均验证通过；不要求或伪造 `0043_BOOTSTRAP_SHA`、仓库 workflow tests或默认分支 commit。
- bootstrap trust verifier 的 commit/artifact digest在candidate产生前已登记；wrong signer/repository/workflow/SHA/digest、revoked signer及候选控制的 snapshot均fail closed。互斥 mode 字段不可为空。
- 未获得远端/三平台 runner 权限时明确停在 D-1，不启动 T0。

**Rollback:** hosted mode先按同一授权禁用五个workflow并冻结新dispatch，再回滚bootstrap commit、删除/锁死bootstrap ref并作废全部run/attestation；不影响0041，已产生的审计记录保留但不得复用。self-hosted mode撤销runner/signer/verifier/acceptance-owner/ledger登记并将对应fingerprint/digest加入revocation record；没有仓库commit可回滚。

**Suggested commit:** hosted mode 为 `ci: add the signed 0043 baseline bootstrap`；self-hosted mode 无仓库 commit。

### Task 0: 建立 clean baseline、ADR 与不可变 capability goldens

**Owner:** Architecture + Acceptance

**Depends on:** D-1 通过。hosted mode 记录 `0041_LANDED_SHA` 与 `0043_BOOTSTRAP_SHA`，T0 分支从后者的 clean 后代开始；self-hosted mode 记录 `0041_LANDED_SHA`、runner-contract digest、signer fingerprint 与 attestation IDs，T0 只要求从 0041 的 clean 后代开始。

**Files:**

- Create: docs/architecture/adr/0043-breaking-unified-core.md
- Create: docs/quality/0043-capability-baseline.json
- Create: docs/quality/0043-architecture-baseline.json
- Create: docs/quality/0043-performance-baseline.json
- Create: docs/quality/0043-browser-baseline.json
- Create: docs/quality/0043-acceptance-toolchain.json
- Create: requirements-0043-acceptance.txt
- Create: tests/parity/fixtures/capability-ledger-0043.json
- Create: tests/parity/fixtures/legacy-surface-inventory-0043.json
- Create: tests/parity/fixtures/module-disposition-0043.json
- Create: tests/parity/fixtures/numerical-golden-0043.json
- Create: tests/parity/fixtures/shape-golden-0043.json
- Create: tests/parity/fixtures/plot-golden-0043.json
- Create: tests/parity/fixtures/report-golden-0043.json
- Create: tests/parity/fixtures/test-node-disposition-0043.json
- Create: tests/contracts/fixtures/public-api-0.5.0.planned.json
- Create: tests/parity/scenarios.py
- Create: tests/parity/normalizers.py
- Create: tests/parity/test_ledger.py
- Create: scripts/capture_capability_baseline.py
- Create: scripts/check_feature_parity.py
- Create: scripts/check_architecture_convergence.py
- Create: scripts/run_0043_cutover_gates.py
- Create: scripts/run_0043_final_gates.py
- Create: scripts/verify_0043_trust_anchor.py
- Create: scripts/check_remote_enforcement.py
- Create: scripts/request_0043_remote_attestation.py
- Create: scripts/request_0043_merge_recheck.py
- Modify: scripts/test_installed_wheel.py
- Modify: scripts/check_dependency_matrix.py
- Modify: scripts/check_release_consistency.py
- Create: scripts/check_legacy_references.py
- Create: scripts/check_0043_workflow_graph.py
- Replace: scripts/snapshot_public_api.py
- Modify: benchmarks/workloads.py
- Modify: scripts/profile_workloads.py
- Modify: scripts/profile_hotspots.py
- Modify: scripts/compare_benchmarks.py
- Modify: scripts/check_performance.py
- Modify: scripts/collect_quality_baseline.py
- Modify: scripts/check_quality_snapshot.py
- Test: tests/quality/test_capture_capability_baseline.py
- Test: tests/quality/test_check_feature_parity.py
- Test: tests/quality/test_architecture_convergence.py
- Test: tests/quality/test_0043_profiling_contract.py
- Create: tests/quality/test_run_0043_cutover_gates.py
- Create: tests/quality/test_run_0043_final_gates.py
- Create: tests/quality/test_verify_0043_trust_anchor.py
- Create: tests/quality/test_check_remote_enforcement.py
- Create: tests/quality/test_request_0043_remote_attestation.py
- Create: tests/quality/test_request_0043_merge_recheck.py
- Create: tests/packaging/test_installed_wheel_profiles.py
- Modify: tests/quality/test_dependency_matrix.py
- Modify: tests/quality/test_snapshot_public_api_source.py
- Create: tests/quality/test_legacy_reference_gate.py
- Create: tests/quality/test_offline_network_guard.py
- Create: tests/quality/test_0043_workflow_graph.py
- Create: tests/quality/fixtures/0043-final-workflow-contract.json
- Modify: tests/benchmarks/test_profile_workloads.py
- Modify: tests/quality/test_check_performance_source.py
- Modify: tests/quality/test_check_quality_snapshot.py
- Modify: tests/quality/test_collect_quality_baseline.py
- Create: docs/quality/0043-quality-baseline.json
- Create: docs/quality/0043-quality-baseline.md

**External immutable outputs after T0c:**

- `D0_TRUST_DIR/0043-d0-trust-anchor.json`
- `D0_TRUST_DIR/0043-d0-attestation.json`（hosted attestation 或 self-hosted detached-signature envelope）
- `D0_TRUST_DIR/0043-d0-expected-bundle.tar.zst`

**Steps:**

1. 从满足规则文件前置条件的 clean worktree 开始并验证 D-1 mode。hosted mode 用 `git merge-base --is-ancestor` 验证 `0041_LANDED_SHA -> 0043_BOOTSTRAP_SHA -> HEAD`；self-hosted mode 只验证 `0041_LANDED_SHA -> HEAD`，并验证仓库外 runner-contract/signing records。记录 mode、base SHA、Python/platform/依赖、source digest，并确认无 tracked/untracked 变化。
2. T0a 只实现 capture/compare/normalizer/profiling/offline-network、installed-wheel/dependency/public-API/legacy/release/trust-anchor/remote-enforcement/workflow-graph checkers、remote-attestation/merge-recheck requester、两个 fail-fast gate runner（含 full-run、execute-untrusted、verify-evidence-only、verify-post-execution-handoff、emit-canonical-verdict 与 verify-attested-final-evidence 模式）、独立 evidence schema/workflow contract、schema tests、ADR 草案和 requirements-0043-acceptance.txt。`0043-final-workflow-contract.json` 冻结完整 caller/reusable graph：event、permissions、preflight-fetch/execute/verify/sign job/needs与隔离约束、runner label、3 OS x 4 Python matrix、3个browser cells、trusted reusable workflow path@完整40位bootstrap SHA、caller head/base、called `job_workflow_ref/job_workflow_sha`、full-git-bundle/source input、pre-execution spec、post-execution handoff、canonical verdict、完整action SHA、T0A argv、artifact名/schema和每格attestation subject；不只冻结final job hash。锁文件固定 build/test frontend 的精确版本与 hashes、Playwright version/browser revision、GitHub CLI及checkout/setup-python/upload-artifact/attest action完整SHA；minimum/latest resolver 在每次 gate 另行输出带 hashes 的 resolution manifest。所有 acceptance checkers、requester 与两个 runner 的 gate 清单、required/forbidden argv、证据 schema、fail-closed 行为与 source SHA 在 T0A_TOOLING_SHA 冻结，T1–T10 不得由候选修改。requester只能向D-1 broker提交已签capability，不能持有token、上传/签署客户端snapshot或改写远端输出。T0a 的 installed-wheel/profile tests 使用合成 wheel/ledger 验证 harness contract；真实 0.5 wheel scenarios 到 T7 才应通过。
3. 写失败测试：inventory 必须扫描 3.3 节完整并集与全部 fincore/**/*.py；architecture/performance/quality CLI 必须 fail closed；trust verifier 必须拒绝 unsigned/wrong signer/wrong workflow/wrong SHA/digest/ancestry/candidate-supplied anchor及self-hosted裸digest gate spec；workflow checker必须拒绝少一个matrix/browser cell、改needs/runner/permission/action SHA/command、reusable branch/tag/短SHA、mutable dispatch ref/tag或collision-branch ruleset漂移、缺`artifact-metadata: write`、caller/called identity混淆、candidate execution与verify/sign同job/runner/credential、无full-git-bundle、preflight预填未来artifact ID、post-execution handoff nonce/run/job/digest错配、candidate-local reusable workflow、伪造manifest、job conclusion/API provenance不匹配或任一cell缺attestation；remote verifier必须拒绝fork、`pull_request_target` candidate result、错误 PR head/base SHA/app/workflow ID、重复同名 context、非 required workflow/context、pending evidence、缺失/隐藏`bypass_actors`、D8a attestation不匹配与 ETag 漂移；requester必须拒绝unsigned/wrong/reused/expired capability、错误immutable dispatch tag/head、非D-1 signer、客户端snapshot upload，以及并发同一jti两个signer成功。input-loader测试必须拒绝任意candidate路径/hash替换、未从D0认证bundle materialize的T0A runner及tree/toolchain digest漂移。merge-recheck测试必须证明fresh runner重新hydrate信任链，receipt verifier逐项比较本次request_jti、actor/repository/PR/head/base/rule/check，跨请求替换任一字段均失败；使用专用test receipt验证并发CAS只有一个winner和重放拒绝，真实merge receipt在未执行expected-head merge前不得转为consumed。offline lane 默认禁止外网 socket，只允许 browser lane 显式声明的 loopback/pipe，browser 下载发生在测试执行前。
4. T0a 工具测试全绿后提交独立 tooling commit，记录 T0A_TOOLING_SHA；提交后 worktree/index 必须 clean，后续不得 amend 或修改这些工具。T0a 只提交工具、工具/schema 测试、normalizer/scenario capture 支撑、ADR 草案与 acceptance lock；`tests/parity/test_ledger.py`、planned API fixture、全部 `tests/parity/fixtures/*` 和 `docs/quality/0043-*-baseline.*` 均不得提前进入 T0a，它们分别在 T0b/T0c materialize。每层 commit 都必须让其当时应存在的完整 `not integration_online` 测试集保持绿色，不能依赖尚未提交的 fixture。
5. 从 clean T0A_TOOLING_SHA 运行 capture，所有输出先写入 repo 外临时目录。先生成 acceptance-toolchain manifest，记录 T0A SHA、runner/checker/test/schema 每文件 SHA256、acceptance environment lock 与允许调用的子进程协议；再生成 legacy-surface inventory、全模块 disposition 和全 pytest node disposition，每项分别填写 capability/alias/quirk 与 owner/target/action/consumer count，0 unclassified/unmapped。
6. 在外部 capture 目录建立两级 ledger：family 行保证领域完整，leaf 行包含 capability_id、family_id、planned_owner、target operation_ids、max_candidate_implementation_fingerprints=1、scenario_ids、observables、tolerance、wheel_profiles、legacy_evidence、status 和 reviewer，不包含候选自行声明的 fingerprint。
7. 捕获固定 seed/frequency/unit/timezone 的 numerical/shape goldens，NaN/Inf 使用 canonical encoding。Matplotlib/Agg 记录按 `os + architecture + Python + Matplotlib + FreeType + renderer` 分平台索引；每条冻结 figure size、DPI、locale、timezone、backend、font family、实际 font file SHA256、fontconfig/renderer version、axis limits、artist visibility、clip/bounding boxes、non-background pixel ratio 与容差化 perceptual hash，不比较 PNG bytes。缺少匹配环境记录时 fail as incomparable，禁止临时扩大 tolerance；minimum/latest 若 renderer tuple 不同，只运行数据/artist/label semantic gate并显式标记 pixel gate not-applicable，不能冒充视觉等价。
8. D0 使用与 candidate 相同的两阶段真实 browser lane：隔离 PLAYWRIGHT_BROWSERS_PATH，先验证 Chromium 缺失错误，再按 acceptance lock 安装固定 revision。真实加载 HTML/PDF，冻结 Playwright/Chromium revision、按平台 browser artifact digest、DOM/console/pageerror、offline asset、交互控件、截图 perceptual digest、PDF page/bookmark/text/metadata 和 cleanup；禁止用 mock browser 生成 baseline。D1 签署前，在独立 reviewer 核对 T0A diff 后，按已获授权把 exact T0A_TOOLING_SHA 推到专用 immutable ref，再通过 D-1 已落默认分支的 restricted `workflow_dispatch`（只接受该 SHA）或等价三台受信 self-hosted runner，由 ubuntu/macos/windows 的 Python 3.11 jobs 各生成一份 platform record/attestation；同一 jobs 同时生成第 7 步的 platform-indexed Agg records，最终合并 browser/plot baselines。缺任一平台不得冻结，也不能以当前 macOS 单机记录代替。
9. Product owner 与独立 reviewer 签署 D1 capability freeze 和 D2 planned 0.5 API；D2 明确 Python support contract 为 `requires-python = ">=3.11,<3.15"`，classifiers 精确列出 3.11/3.12/3.13/3.14，任何未来 3.15+ 在新增矩阵前不可安装。独立 checker 同时核验三平台记录均来自 T0A_TOOLING_SHA、相同 acceptance lock，并匹配 hosted baseline workflow contract 或 self-hosted runner/signing contract。把外部 capture、acceptance-toolchain manifest 与 browser baseline 的精确 bytes materialize 到仓库，连同 planned API fixture 提交为 T0b capability commit，记录 T0B_CAPABILITY_SHA；提交后必须 clean，禁止 amend。
10. 从 clean T0B_CAPABILITY_SHA 捕获 T0c，输出仍先写 repo 外。profiling harness 对 metrics、rolling、transactions、factor、risk、report 的 small/medium/large workload 记录 input_digest、semantic output_digest、median、p95、RSS，并冻结机器/OS/Python/依赖、seed、BLAS threads、warmup/repeat/sample size。
11. 按 3.5 节确定性算法捕获 PLOC/logical LOC、delegation AST、normalized duplicate body 和 import edges；固化 include/exclude manifest、schema_version、T0A tool SHA、当前绝对参考值与 12% 相对降幅公式。
12. 用最终相同的 not integration_online selector、network deny、coverage omit 和完整 pytest/coverage argv 捕获 quality baseline；记录 collector SHA256、schema_version、argv digest、Python/platform/依赖 lock digest，比较器逐项拒绝漂移并要求 candidate branch coverage >= max(60, D0)。
13. 把外部 architecture/performance/quality 输出的精确 bytes materialize 后提交为 T0c baseline commit，记录 T0C_BASELINE_SHA。provenance 必须分开记录：`D0_CAPABILITY_SOURCE_SHA=T0A_TOOLING_SHA`，每个平台 browser record 记录同一 tooling SHA 与各自 workflow run/job/attestation ID，`D0_ARCH_PERF_QUALITY_SOURCE_SHA=T0B_CAPABILITY_SHA`，`D0_BASELINE_COMMIT=T0C_BASELINE_SHA`。checker 还必须证明 T0A→T0B 的 `fincore/**` production-tree digest 完全相同；不再使用会混淆捕获来源的单一 `D0_SOURCE_SHA`。提交后 clean，禁止 amend。
14. 从 clean T0C_BASELINE_SHA 生成仓库外 D0 expected bundle。hosted mode 由默认分支上 `0043_BOOTSTRAP_SHA` 对应 workflow 的 finalize job checkout/fetch exact T0A/T0B/T0C refs，并用 `git show <trusted-sha>:<path>` 收集 T0A acceptance tools/tests/schema/lock、T0B ledger/goldens/browser/capability/toolchain/planned API、T0C architecture/performance/quality baselines；self-hosted mode 由预登记 signer 从三个 detached commits 执行同一 manifest。bundle 不读取未来 candidate worktree。
15. 独立 signer/hosted attestation 生成仓库外 trust anchor，绑定 D1 execution mode、0041/bootstrap或runner-contract身份、`D1_BOOTSTRAP_VERIFIER_SHA256`、T0A/T0B/T0C SHA与ancestry、production-tree digest、workflow/action full SHA、signer/trust-root、input-loader/verdict-verifier/merge-receipt-verifier digest、三平台 attestation IDs、bundle/file-tree SHA256、acceptance lock 与 schema version；同时绑定 D-1 hosted service-contract digest、OIDC issuer/audience、broker/proxy build digest、App/installation ID、authorization/merge-receipt ledger ID+schema+consistency contract、bootstrap branch/immutable dispatch tag/exact collision branch三份ruleset ID+digest及live service attestation，或self-hosted等价runner/key/ledger digests。记录 `D0_TRUST_ANCHOR_SHA256`、`D0_EXPECTED_BUNDLE_SHA256`、不可变 artifact URI/run ID；这些值由 protected CI input或人工签署 handoff 提供，candidate tracked file/CLI 不能覆盖。hosted mode 必须先用 `0043_BOOTSTRAP_SHA` 中已提前落地主线且digest匹配的 bootstrap verifier验签并导出 trusted T0A/T0B/T0C，再进入 trusted T0A 的 `verify_0043_trust_anchor.py` 逐字节复核 bundle；每个T7–T10 checker还必须验证当前service attestation与D0绑定值完全相同且未撤销。self-hosted mode先用预登记 digest 的仓库外 verifier完成同一信任根步骤。两层都通过后 D0 才完成，T1 才可开始。
16. D1 后 capability/module/test ledgers、goldens、browser/toolchain lock 与所有 D0 baselines 不允许修改或追加。发现漏项时停止实现，正式重开 D1，追加有 PO+reviewer 的 amendment commit并生成新的外部 anchor/bundle；旧 baseline、anchor、bundle 全部保留且作废标记可追溯，不能覆写。

**Verify:**

    /Users/yunjinqi/opt/anaconda3/bin/conda run -n base python -m pytest -p no:rerunfailures -o addopts='' tests/quality/test_capture_capability_baseline.py tests/quality/test_check_feature_parity.py -q --tb=short --maxfail=0

    /Users/yunjinqi/opt/anaconda3/bin/conda run -n base python -m pytest -p no:rerunfailures -o addopts='' tests/quality/test_architecture_convergence.py tests/quality/test_0043_profiling_contract.py tests/benchmarks/test_profile_workloads.py tests/quality/test_check_performance_source.py tests/quality/test_check_quality_snapshot.py tests/quality/test_collect_quality_baseline.py tests/quality/test_offline_network_guard.py -q --tb=short --maxfail=0

    /Users/yunjinqi/opt/anaconda3/bin/conda run -n base python -m pytest -p no:rerunfailures -o addopts='' tests/packaging/test_installed_wheel_profiles.py tests/quality/test_dependency_matrix.py tests/quality/test_snapshot_public_api_source.py tests/quality/test_legacy_reference_gate.py tests/quality/test_0043_workflow_graph.py tests/quality/test_run_0043_cutover_gates.py tests/quality/test_run_0043_final_gates.py tests/quality/test_verify_0043_trust_anchor.py tests/quality/test_check_remote_enforcement.py tests/quality/test_request_0043_remote_attestation.py tests/quality/test_request_0043_merge_recheck.py -q --tb=short --maxfail=0

    set -euo pipefail
    FINCORE_0043_D0_CAPTURE=$(mktemp -d /tmp/fincore-0043-d0-capture.XXXXXX)
    MPLBACKEND=Agg MPLCONFIGDIR=/tmp/fincore-0043-mpl /Users/yunjinqi/opt/anaconda3/bin/conda run -n base python scripts/capture_capability_baseline.py --inventory $FINCORE_0043_D0_CAPTURE/legacy-surface-inventory-0043.json --module-disposition $FINCORE_0043_D0_CAPTURE/module-disposition-0043.json --test-disposition $FINCORE_0043_D0_CAPTURE/test-node-disposition-0043.json --ledger $FINCORE_0043_D0_CAPTURE/capability-ledger-0043.json --fixture-dir $FINCORE_0043_D0_CAPTURE --output $FINCORE_0043_D0_CAPTURE/0043-capability-baseline.json --acceptance-lock requirements-0043-acceptance.txt --real-html --real-pdf --provision-browser chromium --browser-evidence $FINCORE_0043_D0_CAPTURE/0043-browser-baseline.json --deny-network-during-scenarios

上述 capture 经 D1/D2 审核后先 materialize 并提交 T0b；确认新 HEAD clean 后，才执行以下 T0c 命令。不得在同一 dirty worktree 连续采集。

    /Users/yunjinqi/opt/anaconda3/bin/conda run -n base python -m pytest -p no:rerunfailures -o addopts='' tests/parity/test_ledger.py tests/quality/test_capture_capability_baseline.py tests/quality/test_check_feature_parity.py -q --tb=short --maxfail=0

    /Users/yunjinqi/opt/anaconda3/bin/conda run -n base python scripts/check_architecture_convergence.py --package fincore --capture $FINCORE_0043_D0_CAPTURE/0043-architecture-baseline.json

    /Users/yunjinqi/opt/anaconda3/bin/conda run -n base python scripts/profile_workloads.py --sizes small medium large --kinds metrics rolling transactions factor risk report --require-output-digest --output $FINCORE_0043_D0_CAPTURE/0043-performance-baseline.json

    /Users/yunjinqi/opt/anaconda3/bin/conda run -n base python scripts/collect_quality_baseline.py --selector 'not integration_online' --deny-network --json $FINCORE_0043_D0_CAPTURE/0043-quality-baseline.json --markdown $FINCORE_0043_D0_CAPTURE/0043-quality-baseline.md

T0c 提交后先通过 hosted finalize job 或 self-hosted signer 生成并下载外部 anchor/bundle。以下是 hosted mode 的完整两层验证；self-hosted mode用预登记 digest 的仓库外 verifier替代 bootstrap checkout，然后执行相同的 detached T0A deep verification。所有变量必须由外部 handoff/protected input绑定，禁止指向 candidate 内文件：

    set -euo pipefail
    test -n "$FINCORE_0041_LANDED_SHA"
    test -n "$FINCORE_0043_BOOTSTRAP_SHA"
    test -n "$T0A_TOOLING_SHA"
    test -n "$T0B_CAPABILITY_SHA"
    test -n "$T0C_BASELINE_SHA"
    test -n "$FINCORE_0043_D0_TRUST_DIR"
    test -n "$FINCORE_0043_D0_ANCHOR_SHA256"
    test -n "$FINCORE_0043_D0_EXPECTED_BUNDLE_SHA256"
    test -n "$FINCORE_GITHUB_REPOSITORY"
    FINCORE_0043_D0_VERIFY_ROOT=$(mktemp -d /tmp/fincore-0043-d0-verify.XXXXXX)
    git cat-file -e "$FINCORE_0043_BOOTSTRAP_SHA^{commit}"
    git merge-base --is-ancestor "$FINCORE_0041_LANDED_SHA" "$FINCORE_0043_BOOTSTRAP_SHA"
    git merge-base --is-ancestor "$FINCORE_0043_BOOTSTRAP_SHA" "$T0A_TOOLING_SHA"
    git merge-base --is-ancestor "$T0A_TOOLING_SHA" "$T0B_CAPABILITY_SHA"
    git merge-base --is-ancestor "$T0B_CAPABILITY_SHA" "$T0C_BASELINE_SHA"
    git worktree add --detach "$FINCORE_0043_D0_VERIFY_ROOT/bootstrap" "$FINCORE_0043_BOOTSTRAP_SHA"
    /Users/yunjinqi/opt/anaconda3/bin/conda run -n base python -I "$FINCORE_0043_D0_VERIFY_ROOT/bootstrap/scripts/verify_0043_bootstrap_trust.py" --anchor "$FINCORE_0043_D0_TRUST_DIR/0043-d0-trust-anchor.json" --attestation "$FINCORE_0043_D0_TRUST_DIR/0043-d0-attestation.json" --bundle "$FINCORE_0043_D0_TRUST_DIR/0043-d0-expected-bundle.tar.zst" --expected-anchor-sha256 "$FINCORE_0043_D0_ANCHOR_SHA256" --expected-bundle-sha256 "$FINCORE_0043_D0_EXPECTED_BUNDLE_SHA256" --expected-repository "$FINCORE_GITHUB_REPOSITORY" --expected-workflow .github/workflows/0043-baseline.yml --expected-workflow-sha "$FINCORE_0043_BOOTSTRAP_SHA" --expected-tooling-sha "$T0A_TOOLING_SHA" --expected-capability-sha "$T0B_CAPABILITY_SHA" --expected-baseline-sha "$T0C_BASELINE_SHA" --output "$FINCORE_0043_D0_VERIFY_ROOT/trusted-shas.json"
    git worktree add --detach "$FINCORE_0043_D0_VERIFY_ROOT/t0a" "$T0A_TOOLING_SHA"
    /Users/yunjinqi/opt/anaconda3/bin/conda run -n base python -I "$FINCORE_0043_D0_VERIFY_ROOT/t0a/scripts/verify_0043_trust_anchor.py" --anchor "$FINCORE_0043_D0_TRUST_DIR/0043-d0-trust-anchor.json" --attestation "$FINCORE_0043_D0_TRUST_DIR/0043-d0-attestation.json" --bundle "$FINCORE_0043_D0_TRUST_DIR/0043-d0-expected-bundle.tar.zst" --expected-anchor-sha256 "$FINCORE_0043_D0_ANCHOR_SHA256" --expected-bundle-sha256 "$FINCORE_0043_D0_EXPECTED_BUNDLE_SHA256" --expected-tooling-sha "$T0A_TOOLING_SHA" --deny-repo-local-input

**Acceptance:**

- D-1 execution mode 已冻结：hosted 的 `0043_BOOTSTRAP_SHA`/tooling refs/workflow IDs，或 self-hosted 的 runner-contract/signer/attestation IDs，可解析且均指向预期 immutable SHA/digest。
- 每个 legacy surface 有 disposition，0 unclassified。
- 每个生产 Python 模块有 owner/target/action/capability disposition，0 unmapped；backends/constants/utils/data support modules 不得漏账。
- D0 完整 collection 的每个 test nodeid 都有 migrate/replace/retire disposition 与 owner；不要求旧 nodeid 在最终仍存在。
- 每个 required capability 覆盖适用的 happy/boundary/error/optional/provider/文档参数场景，每个 scenario 有 golden。
- D1 只冻结唯一性上限与 planned owner；候选 fingerprint 在实现后由独立 checker 计算。每个 leaf capability 恰好一个，多 operation 只能共享该 fingerprint 或作为批准的参数化 mode。
- baseline 不依赖候选 Catalog 自报完整性。
- docs/quality/0043-capability-baseline.json 记录当前 49,412 PLOC 或 clean base 上经审核的新值。
- architecture/performance baseline 与 capability fixtures 都记录 digest，实施期不得重写。
- quality baseline 与 candidate 的 collector/schema/selector/omit/完整 argv/Python/依赖协议必须可逐项比较；任一漂移 fail closed。
- T0A_TOOLING_SHA、T0B_CAPABILITY_SHA、T0C_BASELINE_SHA、D0_CAPABILITY_SOURCE_SHA、D0_ARCH_PERF_QUALITY_SOURCE_SHA 与三平台 browser run/attestation ID 全部记录；T0A→T0B production-tree digest 相同。每次 capture 开始于 clean commit、输出先在 repo 外，三层 commit 无 amend。
- D0 real-browser baseline 已验证缺 browser 错误与真实 HTML/PDF，冻结 Playwright/Chromium revision、分平台 browser digest 和视觉/交互/PDF semantic digests。
- Agg plot baseline 含三 OS 的 renderer/font/DPI/locale tuple 与 font-file digest；候选只能使用匹配记录，renderer 漂移不会被宽容差掩盖。
- 仓库外 trust anchor/attestation/expected bundle 验签通过，绑定 T0A/T0B/T0C、全部 expected bytes、bootstrap verifier digest及D-1 broker/App/proxy/authorization-ledger/merge-ledger/dispatch-tag/三份ruleset service contract与live attestation；后续服务身份或digest漂移fail closed。candidate copy只能作为actual，不能成为expected或覆盖外部路径。

**Rollback:** 按 T0c -> T0b -> T0a 逆序整笔回滚；不得 amend、重写旧 baseline、只删失败 capability 行，或把三个阶段 squash 后冒充 clean capture。

**Suggested commits:**

- test: add immutable 0043 capture and comparison tooling
- test: freeze 0043 capability visual and browser goldens
- test: record clean 0043 architecture performance and quality baselines

### Task 1: 建立 canonical-only runtime、Catalog 与结果模型

**Owner:** Runtime

**Depends on:** T0 / D1 / D2。

**Files:**

- Create: fincore/runtime/__init__.py
- Create: fincore/runtime/specs.py
- Create: fincore/runtime/catalog.py
- Create: fincore/runtime/data.py
- Create: fincore/runtime/validation.py
- Create: fincore/runtime/engine.py
- Create: fincore/runtime/session.py
- Create: fincore/runtime/results.py
- Create: fincore/runtime/artifacts.py
- Modify: fincore/exceptions.py
- Move: fincore/backends/numpy_backend.py -> fincore/runtime/backends.py
- Move: fincore/_types.py -> fincore/runtime/types.py
- Modify (type-import handoff only): fincore/metrics/alpha_beta.py
- Delete after consumer migration: fincore/backends/
- Delete after consumer migration: fincore/_types.py
- Create: tests/runtime/test_catalog.py
- Create: tests/runtime/test_data.py
- Create: tests/runtime/test_engine.py
- Create: tests/runtime/test_session.py
- Create: tests/runtime/test_results.py
- Create: tests/runtime/test_artifacts.py
- Modify/Retarget: tests/test_core/test_backends.py
- Modify/Retarget: tests/test_smoke_import.py
- Modify: fincore/api/catalog.py
- Modify: fincore/api/specs.py
- Modify: fincore/api/invoke.py
- Modify: fincore/api/builtins.py
- Modify: fincore/core/context.py
- Modify: fincore/core/snapshot.py
- Modify: fincore/core/engine.py
- Modify: fincore/core/execution.py
- Modify: fincore/core/planner.py
- Modify: fincore/core/rolling_moments.py
- Modify: fincore/contracts/analysis.py
- Modify: fincore/results/base.py
- Modify: fincore/results/artifacts.py
- Modify: fincore/results/serialization.py

**Steps:**

1. 写失败测试：OperationSpec 不包含 profile、public binding、adapter 或 string signature。
2. 实现 frozen OperationSpec；operation_id 唯一，callable 必须真实可调用。Catalog 可展示 fingerprint，但不得接收手填值；acceptance checker 必须从实际 callable 独立重算并拒绝同一 leaf capability 的重复实现。
3. 实现显式 domain operations() 聚合；不扫描 sys.modules，不 monkeypatch module class，不生成 wrapper。
4. 把 AnalysisSnapshot 迁为 runtime.data 的 immutable input；保留 copy-on-ingest、digest、timezone、mutation isolation。
5. 实现 run/batch/plan；直接调用 canonical callable，统一 Result metadata 和业务错误类别。
6. 把 AnalysisContext 的通用 state/cache 能力迁为 AnalysisSession；不保留 Empyrical/Pyfolio 方法面。
7. 合并 Result、ArtifactBundle、serialization primitives；artifact close 必须幂等且 ownership 明确。
8. 先让旧 api/core/results 作为内部临时消费者；不得创建面向用户的 re-export shim。
9. 增加 source 与临时 wheel runtime smoke。
10. 记录 runtime 新增 PLOC，作为后续旧平台删除时的净额。
11. 按 module-disposition 将 NumPyBackend 与公共 type/schema 迁入 runtime。Metrics owner 在 T1 开始时给 Runtime 一次窄 handoff，只允许把 `metrics/alpha_beta.py` 的 `TYPE_CHECKING` 类型 import 从 `fincore._types` 改到 `fincore.runtime.types`；连同 core、smoke 与全部其他消费者/tests retarget 后运行完整 `not integration_online` suite，确认 consumer count=0 才删除 fincore.backends 与 fincore._types。source/wheel negative import 必须通过，不创建 alias。
12. 将 `exceptions.py` 收敛为不依赖领域 constants 的九类 error algebra；period 合法值由 metrics boundary 作为结构化 details 提供，异常层不得反向 import metrics/constants。保留序列化、可操作诊断与 error-category parity。

**Verify:**

    /Users/yunjinqi/opt/anaconda3/bin/conda run -n base python -m pytest -p no:rerunfailures -o addopts='' tests/runtime tests/contracts/test_artifact_lifecycle.py tests/contracts/test_result_protocol.py -q --tb=short --maxfail=0

    FINCORE_0043_TRANCHE_DIST=$(mktemp -d /tmp/fincore-0043-runtime.XXXXXX)
    /Users/yunjinqi/opt/anaconda3/bin/conda run -n base python -m build --wheel --outdir $FINCORE_0043_TRANCHE_DIST
    /Users/yunjinqi/opt/anaconda3/bin/conda run -n base python scripts/check_feature_parity.py --baseline docs/quality/0043-capability-baseline.json --ledger tests/parity/fixtures/capability-ledger-0043.json --fixture-dir tests/parity/fixtures --families runtime --dist $FINCORE_0043_TRANCHE_DIST

**Acceptance:**

- 同一 operation_id 恰好一个 callable；同一 leaf capability_id 恰好一个 implementation_fingerprint。
- Catalog 覆盖 ledger 中首批 runtime capabilities。
- run 的 source/wheel 结果 normalized digest 相同。
- runtime 不 import fincore._registry、_dispatch、empyrical、pyfolio 或 alphalens。
- backend/type capabilities 在 runtime 可达，fincore.backends 与 fincore._types source/wheel 均不可导入。
- exceptions error algebra 不 import constants/metrics，九类 category/details/serialization parity 通过。

**Rollback:** 回滚整个 runtime commit；不得在旧 registry 中增加新 0.5 binding。

**Suggested commit:** refactor: introduce canonical fincore runtime

### Task 2: 统一 metrics、performance 并完成 Empyrical 能力迁移

**Owner:** Metrics + Performance

**Depends on:** T1。

**Files:**

- Modify: fincore/metrics/__init__.py
- Modify: fincore/metrics/alpha_beta.py
- Modify: fincore/metrics/basic.py
- Modify: fincore/metrics/consecutive.py
- Modify: fincore/metrics/drawdown.py
- Modify: fincore/metrics/perf_attrib.py
- Modify: fincore/metrics/ratios.py
- Modify: fincore/metrics/returns.py
- Modify: fincore/metrics/risk.py
- Modify: fincore/metrics/rolling.py
- Modify: fincore/metrics/stats.py
- Modify: fincore/metrics/timing.py
- Modify: fincore/metrics/yearly.py
- Create: fincore/metrics/frequencies.py
- Create: fincore/metrics/interesting_periods.py
- Create: fincore/metrics/_numeric.py
- Create: fincore/metrics/_rolling.py
- Modify/Move source: fincore/utils/math_utils.py
- Modify/Move source: fincore/utils/data_utils.py
- Delete after consumer migration: fincore/utils/math_utils.py
- Delete after consumer migration: fincore/utils/data_utils.py
- Modify: fincore/performance/__init__.py
- Modify: fincore/performance/cashflows.py
- Modify: fincore/performance/returns.py
- Modify: fincore/performance/inference.py
- Create: fincore/metrics/operations.py
- Create: fincore/performance/operations.py
- Create: tests/parity/test_metrics.py
- Create: tests/parity/test_performance.py
- Modify: tests/test_metrics/**
- Modify: tests/test_empyrical/**
- Modify/Retarget: tests/test_constants/test_periods.py
- Modify/Retarget: tests/test_utils/test_math_utils.py
- Modify/Retarget: tests/test_metrics/coverage_gaps/test_data_utils_edge_cases.py
- Modify/Retarget: tests/contracts/test_metric_surface_profiles.py
- Delete after semantic extraction: tests/contracts/test_metric_surface_profiles.py
- Retain until T7 atomic cutover: fincore/empyrical.py
- Retain until T7 atomic cutover: fincore/_empyrical_legacy.py

**Steps:**

1. 将 Empyrical 唯一组合公式 regression_annual_return 迁入 metrics.alpha_beta；确认 annual_active_risk 等 alias 已有 canonical 等价物。
2. 把 old strict/profile tests 中 numerical、alignment、rolling、index、perf-attrib edge cases 复制或 retarget 到 tests/parity/test_metrics.py；test_metric_surface_profiles.py 的真实 validation/alignment/compute-once 断言迁出后，退休其 registry/profile/multi-surface 断言。纯签名、descriptor、MRO 和 state-binding 测试暂时保留到 T7，与 façade 同一 cutover 删除。
3. 移除 8 个 metrics 模块末尾 install_metric_module_surface 调用；公开函数直接执行自身 kernel。
4. 将 _dispatch 暗含的 validation、alignment 和 output normalization 移到明确的 canonical boundary；对 pandas shape/timezone/mutation 写回归测试。
5. 为 metrics/performance 每个 leaf 注册 OperationSpec，不建立 root/class/module 多 binding。
6. 运行 numerical/property/parity，并运行仍存在的 Empyrical compatibility suite，证明本 tranche 没有制造红色中间提交。
7. 生成 Empyrical deletion-readiness 清单；此时不删除 façade，因为 _pyfolio_impl.Pyfolio 仍继承 Empyrical。
8. 记录 tranche PLOC before/after；若 canonical 迁移暂时净增，把偿还额度登记到 T7 cutover，不以提前删除 façade 伪造净减。
9. 把 period/annualization 与 historical timing periods 落入 metrics.frequencies/interesting_periods，把 nan/rolling/up/down helpers 落入 metrics private primitives；显式 retarget alpha_beta/basic/consecutive/ratios/returns/risk/rolling/stats/timing/yearly 及其全部 tests，再删除 utils.math_utils/data_utils。T2 gate 对上述模块执行 import graph，旧 constants consumer count 必须为 0。其余 constants consumers 按 module ledger 在 T3/T5/T6 迁移，顶层 constants 在 T7 删除。

**Verify:**

    MPLBACKEND=Agg MPLCONFIGDIR=/tmp/fincore-0043-mpl /Users/yunjinqi/opt/anaconda3/bin/conda run -n base python -m pytest -p no:rerunfailures -o addopts='' tests/parity/test_metrics.py tests/parity/test_performance.py tests/test_metrics tests/test_empyrical tests/numerical tests/property/test_time_series_contracts.py -q --tb=short --maxfail=0

    MPLBACKEND=Agg MPLCONFIGDIR=/tmp/fincore-0043-mpl /Users/yunjinqi/opt/anaconda3/bin/conda run -n base python -m pytest -p no:rerunfailures -o addopts='' -m 'not integration_online' tests -q --tb=short --maxfail=0

    FINCORE_0043_TRANCHE_DIST=$(mktemp -d /tmp/fincore-0043-metrics.XXXXXX)
    /Users/yunjinqi/opt/anaconda3/bin/conda run -n base python -m build --wheel --outdir $FINCORE_0043_TRANCHE_DIST
    /Users/yunjinqi/opt/anaconda3/bin/conda run -n base python scripts/check_feature_parity.py --baseline docs/quality/0043-capability-baseline.json --ledger tests/parity/fixtures/capability-ledger-0043.json --fixture-dir tests/parity/fixtures --families metrics performance --dist $FINCORE_0043_TRANCHE_DIST

**Acceptance:**

- metrics/performance required capability 0 missing。
- numerical、shape/label、timezone、mutation goldens 0 未批准差异。
- 新 runtime、metrics、performance 不引用 Empyrical 或 fincore._empyrical_legacy；旧 façade 仅为待切换测试保留。
- Empyrical semantic cases 已在 parity 中有新 operation 对应；旧 façade 与旧测试仍保持可运行直到 T7。
- fincore.utils.math_utils/data_utils 的 canonical implementation 只在 metrics，旧路径 source/wheel 不可导入；frequency semantics 无变化。
- alpha_beta/basic/consecutive/ratios/returns/risk/rolling/stats/timing/yearly 对旧 constants 聚合层 consumer count=0。

**Rollback:** 回滚整个 domain tranche；禁止恢复 wrapper 作为临时发布方案。

**Suggested commits:**

- refactor: make metric kernels the public implementation
- refactor: migrate empyrical-only analytical capability

### Task 3: 建立 portfolio 域并完成 Pyfolio 工作流迁移

**Owner:** Portfolio

**Depends on:** T1、T2 canonical metrics。

**Files:**

- Create: fincore/portfolio/__init__.py
- Create: fincore/portfolio/positions.py
- Create: fincore/portfolio/transactions.py
- Create: fincore/portfolio/round_trips.py
- Create: fincore/portfolio/attribution.py
- Create: fincore/portfolio/capacity.py
- Create: fincore/portfolio/models.py
- Create: fincore/portfolio/settings.py
- Create: fincore/portfolio/operations.py
- Create: tests/parity/test_portfolio.py
- Modify/Move source: fincore/metrics/positions.py
- Modify/Move source: fincore/metrics/transactions.py
- Modify/Move source: fincore/metrics/round_trips.py
- Modify/Move source: fincore/metrics/perf_attrib.py
- Modify/Move source: fincore/metrics/perf_stats.py
- Delete after consumer migration: fincore/metrics/positions.py
- Delete after consumer migration: fincore/metrics/transactions.py
- Delete after consumer migration: fincore/metrics/round_trips.py
- Delete after consumer migration: fincore/metrics/perf_attrib.py
- Delete after consumer migration: fincore/metrics/perf_stats.py
- Modify/Move source: fincore/tearsheets/*.py
- Modify: fincore/_pyfolio_impl.py
- Modify: fincore/_registry.py
- Modify: tests/test_tearsheets/**
- Retain until T7 atomic cutover: fincore/pyfolio.py
- Retain until T7 atomic cutover: fincore/_pyfolio_impl.py
- Retain until T7 atomic cutover: fincore/tearsheets/
- Retain until T7 atomic cutover: fincore/contracts/workflows.py

**Steps:**

1. T2 完成后把上述五个 metrics modules 的 ownership 交给 T3；按 Files 中的逐文件 mapping（执行时用 apply_patch/add-delete）把 positions/transactions/round trips/perf attribution/perf stats 迁为 portfolio canonical functions，retarget _pyfolio_impl、旧 registry 和全部生产/测试消费者后删除原 metrics 路径，避免双实现或 re-export。
2. 将 11 个 tear-sheet workflow 拆为纯 model builder 与 renderer 请求；功能名可以变化，但每个 workflow 的章节/图表 capability 必须在 ledger 可达。
3. 迁移 Pyfolio tests 中 full/risk/drawdown/perf-attrib/positions/transactions/review-fixes 的语义断言；只有 pure public API/signature 断言和 façade 保留到 T7。若 adapter-only None/show/tuple projection 因新模型失效，必须先证明没有独立 observable 并在本 tranche retarget/retire，不能留下红测试。
4. 新 API 用 AnalysisSession + PortfolioInput 承担通用缓存与组合能力；旧 Pyfolio OO state 仅为待切换测试隔离保留。
5. 将原 _pyfolio_impl 中 compute_*_exposures 等调用直接指向 portfolio kernels。
6. 贝叶斯、capacity 和 interesting-times 保持 optional dependency/lazy behavior。
7. report renderer 接口准备好之前，builder 输出 runtime Artifact 请求；不得新建 Pyfolio compatibility object。
8. source parity 与临时 wheel parity 通过后生成 Pyfolio/tearsheets/workflow registry deletion-readiness 清单，不在本任务提前删除。
9. 运行仍存在的 Pyfolio compatibility suite，保证中间提交绿色；T6 迁移最后的 report/tearsheets 消费者，T7 原子删除旧层。
10. source 与 wheel 都断言 fincore.metrics.positions/transactions/round_trips/perf_attrib/perf_stats 不可导入，且 import graph 对这些旧路径的 consumer count=0。
11. 将 sector/cap bucket、capacity、turnover 等 portfolio 业务常量迁入 portfolio.settings；报告展示名称/色彩仍交 T6。旧 constants.style 只为尚未迁移的 T6 consumer 暂留，T7 删除。

**Verify:**

    MPLBACKEND=Agg MPLCONFIGDIR=/tmp/fincore-0043-mpl /Users/yunjinqi/opt/anaconda3/bin/conda run -n base python -m pytest -p no:rerunfailures -o addopts='' tests/parity/test_portfolio.py tests/test_tearsheets tests/test_report -q --tb=short --maxfail=0

    MPLBACKEND=Agg MPLCONFIGDIR=/tmp/fincore-0043-mpl /Users/yunjinqi/opt/anaconda3/bin/conda run -n base python -m pytest -p no:rerunfailures -o addopts='' -m 'not integration_online' tests -q --tb=short --maxfail=0

    FINCORE_0043_TRANCHE_DIST=$(mktemp -d /tmp/fincore-0043-portfolio.XXXXXX)
    /Users/yunjinqi/opt/anaconda3/bin/conda run -n base python -m build --wheel --outdir $FINCORE_0043_TRANCHE_DIST
    /Users/yunjinqi/opt/anaconda3/bin/conda run -n base python scripts/check_feature_parity.py --baseline docs/quality/0043-capability-baseline.json --ledger tests/parity/fixtures/capability-ledger-0043.json --fixture-dir tests/parity/fixtures --families portfolio --dist $FINCORE_0043_TRANCHE_DIST

**Acceptance:**

- 11 类 portfolio workflow、全部 positions/transactions/round-trip/capacity/bayesian 能力可达。
- 新 portfolio/runtime/report path 不构造或 import Pyfolio；旧 façade 只为待切换测试保留。
- builder 不创建隐藏 stateful façade；同一输入只计算一次共享指标。
- figure ownership、caller axes、close/no-orphan goldens 通过。
- 五个旧 metrics portfolio modules 的 source/wheel negative import 通过，且实现 fingerprint 只指向 fincore.portfolio。

**Rollback:** 回滚整个 portfolio tranche；不得将旧 Pyfolio 接入新 runtime 作为 fallback。

**Suggested commits:**

- refactor: establish canonical portfolio analytics
- refactor: replace pyfolio state with portfolio workflows

### Task 4: 合并 Alphalens 与 factor_analysis 为唯一因子实现

**Owner:** Factor Analysis

**Depends on:** T1；可与 T3 并行，不修改 portfolio owned files。

**Files:**

- Modify: fincore/factor_analysis/__init__.py
- Modify: fincore/factor_analysis/data.py
- Modify: fincore/factor_analysis/performance.py
- Modify: fincore/factor_analysis/portfolio.py
- Modify: fincore/factor_analysis/analysis.py
- Modify: fincore/factor_analysis/models.py
- Modify: fincore/factor_analysis/render_matplotlib.py
- Modify: fincore/factor_analysis/tears.py
- Create: fincore/factor_analysis/operations.py
- Create: tests/parity/test_factor_analysis.py
- Create: tests/parity/test_factor_plots.py
- Modify: tests/test_factor_analysis/**
- Modify: tests/benchmarks/test_factor_analysis_performance.py
- Retain until T7 atomic cutover: fincore/alphalens/
- Retain until T7 atomic cutover: fincore/contracts/factor_analysis.py
- Retain until T7 atomic cutover: fincore/contracts/factor_workflows.py

**Steps:**

1. 以 factor_analysis 为 canonical kernel；不要把 strict code 原样复制进去。
2. 迁移 Alphalens 12 个非同名能力：
   - get_clean_factor -> prepare_factor_data_from_forward_returns
   - get_clean_factor_and_forward_returns -> prepare_factor_data
   - demean_forward_returns -> 提升为 performance public helper
   - rate_of_return/std_conversion -> 提升到 performance
   - 四个 plot_*_table -> render_matplotlib 的四个 build_*_table
   - print_table -> report table renderer
   - rethrow/non_unique_bin_edges_error -> canonical quantize_factor 的业务错误类别
3. 将 factor strict tests 中 forward returns、cleaning、performance、portfolio、plotting、bridge、tearsheet 场景复制或 retarget 到 parity；pure signature/public 断言保留到 T7。adapter-only legacy projection/empty-edge 断言若因 canonical 设计失效，必须先证明没有独立 observable 并在本 tranche retarget/retire，不能留下红测试。
4. 保留 prepare、calendar、quantile、IC、returns、weights、turnover、PIT、多 horizon、cost/capacity、FDR/Fama-MacBeth、events、portfolio inputs 和 7 个 report workflow。
5. 合并 model codec 到 runtime results；factor model 只保留领域字段。
6. renderer 输出 normalized artifact；不要求旧 Matplotlib artist 布局完全一致。
7. source + factor/factor-visualization wheel profiles 通过后生成 alphalens 与 factor compatibility contracts deletion-readiness 清单，不在本任务提前删除。
8. 运行仍存在的 Alphalens compatibility suite，保持中间提交绿色；negative import 与实际删除在 T7 原子完成。

**Verify:**

    MPLBACKEND=Agg MPLCONFIGDIR=/tmp/fincore-0043-mpl /Users/yunjinqi/opt/anaconda3/bin/conda run -n base python -m pytest -p no:rerunfailures -o addopts='' tests/parity/test_factor_analysis.py tests/parity/test_factor_plots.py tests/test_factor_analysis tests/numerical/test_factor_costs.py tests/numerical/test_factor_inference.py tests/property/test_factor_invariants.py -q --tb=short --maxfail=0

    MPLBACKEND=Agg MPLCONFIGDIR=/tmp/fincore-0043-mpl /Users/yunjinqi/opt/anaconda3/bin/conda run -n base python -m pytest -p no:rerunfailures -o addopts='' -m 'not integration_online' tests -q --tb=short --maxfail=0

    FINCORE_0043_TRANCHE_DIST=$(mktemp -d /tmp/fincore-0043-factor.XXXXXX)
    /Users/yunjinqi/opt/anaconda3/bin/conda run -n base python -m build --wheel --outdir $FINCORE_0043_TRANCHE_DIST
    /Users/yunjinqi/opt/anaconda3/bin/conda run -n base python scripts/check_feature_parity.py --baseline docs/quality/0043-capability-baseline.json --ledger tests/parity/fixtures/capability-ledger-0043.json --fixture-dir tests/parity/fixtures --families factor --dist $FINCORE_0043_TRANCHE_DIST

**Acceptance:**

- factor ledger leaf 0 missing，Alphalens 独有真实能力全部落位。
- factor numerical/shape/plot/report goldens 0 未批准差异。
- 新 factor_analysis/runtime path 不 import fincore.alphalens、strict factor profile 或 adapter-only contract；旧 façade 只为待切换测试保留。
- factor performance benchmark 没有少算功能；artifact 同时记录 input_digest 与 output_digest。

**Rollback:** 回滚整个 factor tranche；禁止通过恢复 alphalens wrapper 补缺。

**Suggested commits:**

- refactor: promote factor analysis as the canonical kernel
- refactor: migrate remaining alphalens capabilities

### Task 5: 接入 risk、attribution、simulation、optimization、data 与 extensions

**Owner:** Domain Integrations

**Depends on:** T1；可按不重叠目录并行。

**Files:**

- Create: fincore/risk/operations.py
- Create: fincore/attribution/operations.py
- Create: fincore/simulation/operations.py
- Create: fincore/optimization/operations.py
- Modify: fincore/optimization/risk_parity.py
- Create: fincore/data/operations.py
- Create: fincore/data/models.py
- Modify: fincore/data/providers.py
- Modify: fincore/data/__init__.py
- Move/Modify: fincore/data/contracts.py -> fincore/data/models.py
- Move/Modify: fincore/data/snapshots.py -> fincore/runtime/data.py
- Delete after consumer migration: fincore/data/contracts.py
- Delete after consumer migration: fincore/data/snapshots.py
- Create: fincore/extensions/__init__.py
- Create: fincore/extensions/operations.py
- Move: fincore/plugin/specs.py -> fincore/extensions/specs.py
- Move: fincore/plugin/registry.py -> fincore/extensions/registry.py
- Move: fincore/plugin/discovery.py -> fincore/extensions/discovery.py
- Move: fincore/hooks/events.py -> fincore/extensions/events.py
- Delete after test migration: fincore/plugin/
- Delete after test migration: fincore/hooks/
- Modify: fincore/validation.py
- Modify: fincore/runtime/validation.py
- Delete after consolidation: fincore/validation.py
- Modify/Retarget: fincore/core/context.py
- Modify/Retarget: fincore/core/engine.py
- Modify/Retarget: fincore/viz/base.py
- Modify: fincore/contracts/validation.py
- Modify: fincore/contracts/time_series.py
- Modify: fincore/contracts/portfolio.py
- Create: tests/parity/test_risk.py
- Create: tests/parity/test_attribution.py
- Create: tests/parity/test_simulation.py
- Create: tests/parity/test_optimization.py
- Create: tests/parity/test_data_extensions.py
- Create: tests/test_data/test_pandas_datareader_provider.py
- Modify/Retarget: tests/test_plugin/**
- Modify/Retarget: tests/test_hooks/**
- Modify/Retarget: tests/test_core/test_engine_equivalence.py
- Modify/Retarget: tests/contracts/test_plugin_overlay_snapshot.py
- Modify/Retarget: tests/contracts/test_returns_schema.py
- Modify/Retarget: tests/quality/test_release_gate_changed_lines.py
- Modify/Retarget: tests/test_validation.py
- Modify/Retarget: tests/test_smoke_import.py

**Steps:**

1. 为所有 leaf capability 建 OperationSpec，不能只登记 capabilities.py 当前 25 个高层行。
2. T1 在 D3 handoff 后把 runtime/validation.py 的独占 ownership 交给 T5；合并 fincore.validation 与旧 contracts validation，retarget 消费者后删除 fincore/validation.py，validation 只发生在 canonical boundary。
3. 修复当前两个不可解析 provider capability path；在 fincore/data/providers.py 新增 canonical PandasDataReaderProvider，在 data.operations 注册 fetch operation，并从 data.__init__ 导出。reader 未注入时才 lazy import pandas_datareader.data.DataReader；source/wheel 测试把该真实入口替换为 injected fake，执行 column/index/exception normalization，全程不访问网络。
4. 保留 EVT/GARCH/VaR/ES、backtest、walk-forward、calibration 与 risk report 的独立数值 oracle。
5. 保留 Brinson/Fama-French/style 的 provider/cache/inference 能力。
6. 固定 simulation seed/RNG policy；Monte Carlo、GBM、bootstrap、scenario 的 output_digest 可复现。
7. 保留 optimization feasibility、objective、frontier 和 risk parity；把 `risk_parity.py` 的 annualization constant 改到 canonical metrics frequency owner，T5 import graph 对旧 constants consumer count=0。
8. Runtime、Reporting、Packaging owners 对 Files 中列出的 core/context.py、core/engine.py、viz/base.py 及 core/contracts/quality/smoke tests 做一次性窄 handoff。按上述逐文件 mapping（执行时用 apply_patch/add-delete）合并 plugin + hooks 为 extensions，retarget 这些消费者及 integration tests 后删除旧目录，不创建旧路径 re-export。注册、发现、隔离和 hook event 仍是 ledger capability，并增加 source/wheel negative import。
9. 把 data/contracts.py 的 provider schema 合并到 data.models，把 data/snapshots.py 的通用 immutable snapshot 合并到 runtime.data；retarget 全部 consumers/tests 后删除旧文件，避免 data/runtime 双 snapshot。
10. 迁移其余领域合同到各自 models/specs；删除只服务旧 profile 的合同。

**Verify:**

    /Users/yunjinqi/opt/anaconda3/bin/conda run -n base python -m pytest -p no:rerunfailures -o addopts='' tests/parity/test_risk.py tests/parity/test_attribution.py tests/parity/test_simulation.py tests/parity/test_optimization.py tests/parity/test_data_extensions.py tests/test_data/test_pandas_datareader_provider.py tests/test_risk tests/test_attribution tests/test_simulation tests/test_optimization tests/test_data tests/test_plugin tests/test_hooks tests/integration/test_plugin_end_to_end.py tests/numerical tests/property -q --tb=short --maxfail=0

    /Users/yunjinqi/opt/anaconda3/bin/conda run -n base python -m pytest -p no:rerunfailures -o addopts='' -m 'not integration_online' tests -q --tb=short --maxfail=0

    FINCORE_0043_TRANCHE_DIST=$(mktemp -d /tmp/fincore-0043-domains.XXXXXX)
    /Users/yunjinqi/opt/anaconda3/bin/conda run -n base python -m build --wheel --outdir $FINCORE_0043_TRANCHE_DIST
    /Users/yunjinqi/opt/anaconda3/bin/conda run -n base python scripts/check_feature_parity.py --baseline docs/quality/0043-capability-baseline.json --ledger tests/parity/fixtures/capability-ledger-0043.json --fixture-dir tests/parity/fixtures --families risk attribution simulation optimization data extensions --dist $FINCORE_0043_TRANCHE_DIST

**Acceptance:**

- 五个领域与 extensions required capability 0 missing。
- provider profile 全部使用 fake/offline client。
- data-pandas-datareader profile 从 wheel lazy import 真实依赖入口、注入 fake reader 并执行 canonical fetch operation；不是 import-only smoke。
- fincore.plugin、fincore.hooks 与 fincore.validation source/wheel import 均失败，且新生产消费者为 0。
- fincore.data.contracts/snapshots source/wheel import 失败；provider model 与 immutable snapshot 各只有一个 canonical implementation。
- optional dependency 未调用时 0 eager import，调用缺依赖时错误类别正确。
- numerical/property tests 0 未批准回归。

**Rollback:** 按领域回滚；共享 validation 失败时回滚整个 validation tranche。

**Suggested commits:**

- refactor: register canonical analytical domains
- refactor: unify validation contracts
- refactor: consolidate plugins and hooks as extensions

### Task 6: 统一 report、result、artifact、codec 与 renderer

**Owner:** Reporting + Runtime

**Depends on:** T2–T5 的 domain models。

**Files:**

- Create: fincore/report/portfolio/__init__.py
- Create: fincore/report/portfolio/builders.py
- Create: fincore/report/factor/__init__.py
- Create: fincore/report/factor/builders.py
- Create: fincore/report/renderers/__init__.py
- Create: fincore/report/renderers/xlsx.py
- Create: fincore/report/styles.py
- Create: fincore/report/tables.py
- Move/Modify: fincore/report/render_html.py
- Move/Modify: fincore/report/render_pdf.py
- Move/Modify: fincore/factor_analysis/render_matplotlib.py
- Modify: fincore/report/compute.py
- Modify: fincore/report/model.py
- Modify: fincore/report/artifacts.py
- Modify: fincore/report/provenance.py
- Modify: fincore/results/*.py
- Modify: fincore/viz/*.py
- Modify/Move source: fincore/constants/color.py
- Modify/Move source: fincore/constants/style.py
- Modify/Move source: fincore/utils/common_utils.py
- Delete after semantic disposition: fincore/utils/date_utils.py
- Delete after semantic disposition: fincore/utils/deprecate.py
- Modify: fincore/tearsheets/bayesian.py
- Modify: fincore/tearsheets/returns.py
- Modify: fincore/tearsheets/sheets.py
- Modify: fincore/tearsheets/transactions.py
- Create: fincore/report/operations.py
- Create: tests/parity/test_plots.py
- Create: tests/parity/test_reports.py
- Create: tests/parity/test_report_xlsx.py
- Create: tests/parity/test_report_pdf_contract.py
- Modify: tests/test_report/**
- Modify: tests/test_viz/**
- Modify: tests/test_utils/test_export_destination.py

**Steps:**

1. 写失败 trace：report 一次构建只能对共享 operation 计算一次，且不能 import Empyrical/Pyfolio 或访问 session._*。
2. 用 runtime ResultMetadata/ArtifactBundle/codec 替换 ReportArtifacts、factor artifact 和 results 多套模型。
3. 区分 report.portfolio 与 report.factor builder，避免两个 full tear-sheet 名称冲突。
4. 将 portfolio/factor builder 输出统一 report model，再由 matplotlib/html/pdf/xlsx/interactive renderer 消费。
5. HTML 保持离线、自包含、无外部 URL；PDF 契约冻结 page count/bookmarks/extracted text/metadata；XLSX 保持显式输出目录、sheet/table 内容且绝不在 import/默认调用时写文件。
6. PDF 单元测试覆盖 Playwright Python package 缺失与 Chromium binary 缺失两条 lazy failure；后者返回 optional_dependency_missing、component=chromium 和可执行安装提示。真实 Chromium 渲染必须由 T7/T9 的同 wheel browser lane 完成，mock browser 不能作为 D6 证据。
7. Artifact close/cleanup/ownership 幂等；成功、browser startup failure、render failure 路径均无临时文件和 orphan figure。
8. 将 Plotly/Bokeh/HTML/Matplotlib 后端作为 renderer 能力注册，不在 root 暴露旧 façade。
9. 将 report/tearsheets/utils 的内部 Empyrical/Pyfolio 调用改为 canonical metrics/portfolio/runtime；parity 通过后把旧 results、tearsheets 和重复 codec/artifact 标为 deletion-ready，实际删除留到 T7 原子 cutover。
10. 按 module ledger 将 color/display/stat-name constants 迁入 report.styles，将 print/table/export/legend/asset helpers 拆入 report.tables/renderers 与 portfolio helpers；timer/deprecate 若无独立 observable 按已批准 disposition 删除。retarget 全部 tests 后，constants/utils 聚合目录标为 T7 deletion-ready，不创建新 catch-all。

**Verify:**

    MPLBACKEND=Agg MPLCONFIGDIR=/tmp/fincore-0043-mpl /Users/yunjinqi/opt/anaconda3/bin/conda run -n base python -m pytest -p no:rerunfailures -o addopts='' tests/parity/test_plots.py tests/parity/test_reports.py tests/parity/test_report_xlsx.py tests/parity/test_report_pdf_contract.py tests/test_report tests/test_viz tests/test_utils/test_export_destination.py tests/contracts/test_artifact_lifecycle.py -q --tb=short --maxfail=0

    FINCORE_0043_TRANCHE_DIST=$(mktemp -d /tmp/fincore-0043-report.XXXXXX)
    /Users/yunjinqi/opt/anaconda3/bin/conda run -n base python -m build --wheel --outdir $FINCORE_0043_TRANCHE_DIST
    /Users/yunjinqi/opt/anaconda3/bin/conda run -n base python scripts/check_feature_parity.py --baseline docs/quality/0043-capability-baseline.json --ledger tests/parity/fixtures/capability-ledger-0043.json --fixture-dir tests/parity/fixtures --families report viz --dist $FINCORE_0043_TRANCHE_DIST

**Acceptance:**

- report/plot capability 0 missing，normalized golden 0 未批准差异。
- 同一工作流共享计算次数符合 trace。
- 新 runtime/report 只有一套 active result metadata、artifact lifecycle 和 serialization codec；旧文件被隔离且无 canonical consumer。
- report/compute.py 不 import 旧 façade、不访问私有 context 字段。
- mock PDF contract 与 missing-Chromium failure 通过；D6 仍保持未通过，直到 T7 同一 wheel 的真实 Chromium lane 产生并验证 PDF。
- constants/utils 中所有真实 observable 已落到 metrics/portfolio/runtime/data/report；旧聚合层 consumer count=0，timer/deprecate 的 retire disposition 已批准。

**Rollback:** 统一模型和 renderer 迁移作为一个 tranche 回滚；禁止恢复旧 result projection adapter。

**Suggested commits:**

- refactor: unify report models and artifact lifecycle
- refactor: consolidate portfolio and factor renderers
- refactor: disconnect legacy report and tearsheet consumers

### Task 7: 迁移旧 contract tests，并原子完成 breaking cutover

**Owner:** Runtime + Packaging + Governance

**Depends on:** T2–T6 的 source/unit parity 通过；D6 installed-wheel 与 real-PDF gate 在本任务 exact-tree candidate 上完成后才算通过。

**Files:**

- Modify: fincore/__init__.py
- Modify: pyproject.toml
- Modify: README.md
- Modify: CONTRIBUTING.md
- Modify: GOVERNANCE.md
- Modify: CHANGELOG.md
- Create: docs/MIGRATION-0.5.md
- Modify: docs/API_STABILITY.md
- Modify: docs/governance/branch-governance.md
- Create: tests/contracts/fixtures/public-api-0.5.0.dev0.json
- Modify/Retarget: tests/** except T0A frozen acceptance tests and T0b immutable parity fixtures/tests
- Modify/Retarget: tests/contracts/** except tests/contracts/fixtures/public-api-0.5.0.planned.json
- Modify: tests/packaging/test_extras_union.py
- Modify: tests/packaging/test_optional_extras.py
- Modify: tests/packaging/test_release_candidate.py
- Modify: tests/packaging/test_release_consistency.py
- Modify: tests/packaging/test_wheel_contents.py
- Modify: examples/**
- Modify: mkdocs_docs/**
- Modify: .github/workflows/ci.yml
- Modify: .github/workflows/publish.yml
- Modify: .github/workflows/pr-governance.yml
- Modify: .github/workflows/test-priority.yml
- Modify: .github/CODEOWNERS
- Modify: .github/pull_request_template.md
- Modify: .github/ISSUE_TEMPLATE/bug_report.yml
- Modify: .github/ISSUE_TEMPLATE/feature_request.yml
- Modify: .github/governance/labels.json
- Create: .github/governance/0043/metrics-schema.json
- Modify: scripts/ci/classify_pr_risk.py
- Modify: scripts/ci/verify_github_governance.py
- Modify: tests/quality/test_classify_pr_risk.py
- Modify: tests/quality/test_github_community_files.py
- Modify: tests/quality/test_release_gate_changed_lines.py
- Modify: tests/quality/test_verify_github_governance.py
- Modify: tests/quality/fixtures/github_governance/matching.json
- Modify: tests/quality/fixtures/github_governance/drift.json
- Delete: fincore/empyrical.py
- Delete: fincore/pyfolio.py
- Delete: fincore/alphalens/
- Delete: fincore/_empyrical_legacy.py
- Delete: fincore/_pyfolio_impl.py
- Delete: fincore/api/
- Delete: fincore/core/
- Delete after schema migration: fincore/contracts/
- Delete: fincore/results/
- Delete: fincore/capabilities.py
- Delete: fincore/_compat/
- Delete: fincore/_registry.py
- Delete: fincore/_dispatch.py
- Delete: fincore/tearsheets/
- Delete after module disposition: fincore/constants/
- Delete after module disposition: fincore/utils/
- Delete if not already removed in T1: fincore/backends/
- Delete if not already removed in T1: fincore/_types.py
- Delete after semantic migration: tests/compat/
- Delete: scripts/generate_compat_manifest.py
- Delete: scripts/generate_alphalens_oracle.py
- Delete: scripts/generate_alphalens_upstream_test_inventory.py
- Delete: scripts/check_alphalens_upstream_test_migration.py
- Delete: tests/contracts/test_compat_adapter_isolation.py
- Delete: tests/contracts/test_public_api_behavior_probes.py
- Delete: tests/contracts/fixtures/public-api-0.3.x.json
- Delete: tests/contracts/fixtures/public-api-probes-0.3.x.json
- Delete: tests/contracts/fixtures/public-api-0.4.0.dev0.json

**Steps:**

1. 在删 source 前，对 D0 完整 collection 的每个 nodeid 核销 disposition，不限 tests/compat/contracts：tests/test_pyfolio、test_core、coverage suites、smoke imports、quality/release、plugin/integration 等全部 migrate、replace 或 retire，0 unclassified。语义断言必须已有 parity 对应，pure path/signature tests 与 façade 在同一 commit 删除。
2. 将仍依赖 api/core/contracts/results/旧 façade 的所有 tests retarget 到 runtime/domain；先使完整 not integration_online suite 绿色。
3. 生成只针对 0.5 设计的 public API snapshot 并与 D2 planned fixture 核对；它防止未来漂移，不与 0.4 snapshot 比较。
4. 把 root __init__.py 缩为 version 和领域模块发现；切到 0.5.0.dev0，删除 flat API 和旧 lazy loader，保留 Beta classifier。pyproject 同步设为 `requires-python = ">=3.11,<3.15"` 并精确声明 3.11–3.14 classifiers；metadata/checker 拒绝无上界或遗漏 3.14 的配置。
5. 功能 extras 定为 factor、visualization、factor-visualization、interactive、report-pdf、report-xlsx、bayesian、data-yahoo、data-alphavantage、data-pandas-datareader、data-cn、all；core 是无 extra 的 consumer profile。
6. 删除 pyfolio、alphalens、factor-analysis、datareader 兼容 extras；不提供 alias。使用 T0A 已冻结的 test_installed_wheel profiles/flags（--dist、--ledger、--profiles、--assert-legacy-absent、--real-html、--real-pdf、--provision-browser、--browser-baseline、--browser-evidence、--acceptance-lock、--deny-network）和 check_dependency_matrix --dist/--lane/--ledger contract 验证同一候选 wheel；候选只提供 pyproject extras、Catalog operations 与 ledger scenarios，不能修改 harness/checker/parser/schema tests。把 twine 加入 dev/release 验证依赖。
7. 将 api/core/contracts/results 中最后的消费者迁入 runtime/domain；对 old cluster 做 rg/import graph，只有 consumer count=0 才进入删除步骤。
8. 在同一个 atomic cutover head 中同步迁移 0041 治理 schema：ci/publish/pr-governance、risk classifier、CODEOWNERS、PR/Issue templates、labels、governance fixtures/tests 从 compat/0.4/旧 profile 改为 capability parity/0.5/新 profile；publish 必须依赖完整 0043 gate。candidate CI只能是符合T0A `0043-final-workflow-contract.json` 的caller graph：single-build产生唯一wheel后，12个OS/Python cells与3个real-browser cells分别调用 `owner/repo/.github/workflows/0043-acceptance-cell.yml@<完整40位0043_BOOTSTRAP_SHA>`，final job同样以完整SHA调用`0043-final-seal.yml`；不得使用branch/tag/短SHA，也不得在candidate workflow内重写cell/final命令。每个受信reusable workflow严格使用preflight-fetch/execute/verify/sign四段：preflight认证full-git-bundle和输入，execute在无secret/OIDC/repository token环境运行candidate并输出raw artifact，verify在clean runner先认证bootstrap/D0/T0A再核验post-execution handoff并产canonical verdict，sign在第四runner不执行candidate/T0A而只签verdict。caller T8与called `job_workflow_ref/job_workflow_sha`分别绑定，四类job IDs、matrix tuple、T8/base、同一wheel、argv/exit与artifact digest进入subject。最终唯一稳定check名`0043-final-seal`依赖并逐格验签全部15份attestation，同时用Actions API核对真实job identity/conclusion/runner label；少一格、同runner凭据污染、假manifest或caller graph/blob漂移均失败。workflow ID/path/ref/run SHA与expected GitHub App ID进入evidence；治理测试禁止仓库内任何其他workflow/job产生同名context。保留0041 baseline和历史metrics不变，新建versioned0043 schema。
9. 删除 Empyrical、Pyfolio、Alphalens、tearsheets、旧 api/core/contracts/results、registry/dispatch、已 disposition 的 constants/utils/backends/_types support 聚合层，以及已完成语义迁移的 tests/compat、manifest/oracle generators。生产、测试、workflow/governance 变更不可拆成两个可单独落地主线 head。
10. 更新 README、CONTRIBUTING、GOVERNANCE、docs/governance/branch-governance、CHANGELOG、MIGRATION、API_STABILITY、examples 与 MkDocs；migration guide 只提供 capability -> 新 operation/import 映射，不包含 executable shim。0041 baseline 与历史 metrics 保持原样。
11. 使用 T0A 冻结的 fail-closed legacy reference gate 扫描 production、逐文件 maintained docs、examples、.github/workflows 与当前治理文件，拒绝旧 module/class/flat API/strict profile/0.4 active snapshot/compat gate，也拒绝 fincore.plugin/hooks/validation/constants/utils/backends/_types、data.contracts/snapshots 和五个旧 metrics portfolio modules。历史 allowlist 必须逐文件列出；MIGRATION/CHANGELOG 与 versioned 0041 文件可整文件允许，license/oracle 只允许显式标注 provenance 的行。禁止 allow docs/quality 或 .github 这类目录级豁免。候选只能提供被扫描内容与显式 allowlist input，不能修改扫描器；“0 match”返回 0，扫描错误或不可读路径返回非零。
12. browser fresh consumer 使用隔离 PLAYWRIGHT_BROWSERS_PATH 和 D0 browser baseline：先在未安装 browser 的 venv 验证 actionable optional_dependency_missing，再按 acceptance lock 安装同一 Playwright/Chromium revision并校验分平台 artifact digest。随后在外网 socket deny 下真实加载 HTML、检查 console/pageerror/offline assets、操作 Plotly/Bokeh 控件并做 perceptual screenshot；真实 PDF 还要验证 page/bookmark/text/metadata 与 success/failure cleanup。仅 import Playwright、DOM digest 或 mock Chromium 不得通过 D6。
13. 不删除第三方 notices；单独运行 provenance/license review。任何远端 label/Ruleset 变更仍需另行授权。
14. T7 必须在从已验收 `ACCEPTED_T6_SHA` 创建的独立 clean candidate worktree 中完成，并另建独立 bootstrap/T0A acceptance worktree；各路径、Python import path 与临时目录不得重叠。根据 0041 已签署的真实 PR base ref，用只读远端证据解析并冻结 `EXPECTED_PR_BASE_SHA`，要求它是 `ACCEPTED_T6_SHA` 的祖先；仓库必须有完整 history 和 exact T0A/T0B/T0C/base 对象。所有 source/test/governance 删除与迁移按 owned-path allowlist 暂存并形成一个仅本地的 CUTOVER_CANDIDATE_SHA，且其唯一 parent 必须是 `ACCEPTED_T6_SHA`；记录 HEAD/tree/base SHA，工作树和 index 必须 clean，禁止 push/merge/rebase/amend该 candidate，直到以下 exact-tree gates 全绿。
15. 从 CUTOVER_CANDIDATE_SHA 的精确 tree 按 acceptance lock/hash constraints 只构建一个 wheel；minimum/latest 先各解析一次并记录包版本、来源与 artifact hashes，随后冻结该 resolution 执行真实 scenario，禁止测试途中再次访问 resolver。再跑全量 feature parity、全部 profiles、legacy negative imports、真实 HTML/PDF、完整非在线 suite、branch coverage、architecture LOC/duplicate 和 performance semantic/budget comparison；Agg visual gate 必须先匹配当前 OS/architecture/Python/Matplotlib/FreeType/font/DPI record，环境不匹配则 fail as incomparable。任何失败或真实 base tip 漂移都立即把该 SHA、tree、wheel 和 evidence 标为 discarded；不得在其上追加 fix commit。实现失败时，替代 candidate 必须在新的 worktree/branch 中直接从同一个 accepted T6 SHA 重建，把经审核的 owned-path diff 与修复重新 materialize 成一个新的 atomic cutover commit；base 漂移时则必须先在新 base 上重建并重新验收 T1–T6 受影响 tranches，得到新的 `ACCEPTED_T6_SHA` 后再做 T7。checker 拒绝 ancestry 含任一 discarded candidate SHA。之后从头重跑所有 T7 gates，不能复用旧 wheel 或部分日志。
16. hosted mode 的权威命令必须先从独立 `0043_BOOTSTRAP_SHA` worktree运行提前落地主线的 bootstrap verifier，以外部 trust root验签并导出 trusted T0A/T0B/T0C；self-hosted mode必须设置`FINCORE_0043_GATE=t7`并完整执行4.1节`verify-and-exec`，不得引用任何bootstrap SHA。只有随后才能从独立 T0A acceptance worktree 以 `python -I` 和 sanitized environment 执行 T0A 冻结的 run_0043_cutover_gates.py；candidate 仅作为 `--candidate-repo` data/probe target。runner 不得把 candidate root 加入自身 sys.path，不得在自身进程 import `fincore`，只可在 cwd 位于所有 worktree 之外、无 PYTHONPATH/PYTHONHOME/user-site 的 fresh subprocess/venv 中探测 candidate source/wheel。所有 expected tool/ledger/golden/baseline/lock bytes 只从仓库外已验签 D0 bundle 解包；candidate 同路径文件仅作为 actual 做逐字节/语义比较，CLI 禁止 candidate-supplied expected。runner 把 source/tree/wheel/browser/ledger/fixture/quality/architecture/performance digest、完整 argv 和 exit code写入 repo 外 evidence，前后检查全部 worktree clean；同一 wheel SHA 传给全部 consumer/dependency lanes且禁止子命令重建。T8 只审核/materialize 该 exact-SHA evidence，不能修改 production/scripts/tests/config；T8 完成前不得开 cutover PR 或落地主线，Task -1 的 bootstrap PR 是唯一前置例外。

**Verify:**

前两条 candidate-checkout 命令仅用于快速诊断；以下完整块只适用于hosted mode。self-hosted mode设置`FINCORE_0043_GATE=t7`及actual head/base后完整运行4.1节入口，由外部verifier执行同一T0A runner；不得剪贴hosted block。只有该mode对应的信任根验证后才能生成权威T7 evidence。

    /Users/yunjinqi/opt/anaconda3/bin/conda run -n base python -m pytest -p no:rerunfailures -o addopts='' tests/packaging/test_installed_wheel_profiles.py tests/packaging/test_optional_extras.py tests/quality/test_dependency_matrix.py tests/quality/test_snapshot_public_api_source.py tests/quality/test_legacy_reference_gate.py tests/quality/test_run_0043_cutover_gates.py tests/quality/test_run_0043_final_gates.py -q --tb=short --maxfail=0

    /Users/yunjinqi/opt/anaconda3/bin/conda run -n base python scripts/check_legacy_references.py --production fincore --maintained-doc README.md CONTRIBUTING.md GOVERNANCE.md docs/API_STABILITY.md docs/governance/branch-governance.md .github/CODEOWNERS .github/pull_request_template.md .github/governance/README.md .github/governance/labels.json .github/governance/0043/metrics-schema.json --maintained-root mkdocs_docs examples .github/workflows .github/ISSUE_TEMPLATE --allow-history CHANGELOG.md docs/MIGRATION-0.5.md .github/governance/0041/baseline.json .github/governance/0041/decisions.md .github/governance/0041/metrics-schema.json

    set -euo pipefail
    test -n "$FINCORE_0041_LANDED_SHA"
    test -n "$FINCORE_0043_BOOTSTRAP_SHA"
    test -n "$T0A_TOOLING_SHA"
    test -n "$T0B_CAPABILITY_SHA"
    test -n "$T0C_BASELINE_SHA"
    test -n "$ACCEPTED_T6_SHA"
    test -n "$EXPECTED_PR_BASE_SHA"
    test -n "$CUTOVER_CANDIDATE_SHA"
    test -n "$FINCORE_0043_D0_TRUST_DIR"
    test -n "$FINCORE_0043_D0_ANCHOR_SHA256"
    test -n "$FINCORE_0043_D0_EXPECTED_BUNDLE_SHA256"
    test -n "$FINCORE_GITHUB_REPOSITORY"
    FINCORE_0043_ACCEPTANCE_ROOT=$(mktemp -d /tmp/fincore-0043-acceptance.XXXXXX)
    test "$(git rev-parse HEAD)" = "$CUTOVER_CANDIDATE_SHA"
    test "$(git rev-parse --is-shallow-repository)" = false
    test "$(git rev-parse "$CUTOVER_CANDIDATE_SHA^")" = "$ACCEPTED_T6_SHA"
    git cat-file -e "$FINCORE_0043_BOOTSTRAP_SHA^{commit}"
    git cat-file -e "$T0A_TOOLING_SHA^{commit}"
    git cat-file -e "$T0B_CAPABILITY_SHA^{commit}"
    git cat-file -e "$T0C_BASELINE_SHA^{commit}"
    git cat-file -e "$EXPECTED_PR_BASE_SHA^{commit}"
    git merge-base --is-ancestor "$FINCORE_0041_LANDED_SHA" "$FINCORE_0043_BOOTSTRAP_SHA"
    git merge-base --is-ancestor "$FINCORE_0043_BOOTSTRAP_SHA" "$T0A_TOOLING_SHA"
    git merge-base --is-ancestor "$T0A_TOOLING_SHA" "$T0B_CAPABILITY_SHA"
    git merge-base --is-ancestor "$T0B_CAPABILITY_SHA" "$T0C_BASELINE_SHA"
    git merge-base --is-ancestor "$T0C_BASELINE_SHA" "$CUTOVER_CANDIDATE_SHA"
    git merge-base --is-ancestor "$EXPECTED_PR_BASE_SHA" "$ACCEPTED_T6_SHA"
    git worktree add --detach "$FINCORE_0043_ACCEPTANCE_ROOT/bootstrap" "$FINCORE_0043_BOOTSTRAP_SHA"
    /Users/yunjinqi/opt/anaconda3/bin/conda run -n base python -I "$FINCORE_0043_ACCEPTANCE_ROOT/bootstrap/scripts/verify_0043_bootstrap_trust.py" --anchor "$FINCORE_0043_D0_TRUST_DIR/0043-d0-trust-anchor.json" --attestation "$FINCORE_0043_D0_TRUST_DIR/0043-d0-attestation.json" --bundle "$FINCORE_0043_D0_TRUST_DIR/0043-d0-expected-bundle.tar.zst" --expected-anchor-sha256 "$FINCORE_0043_D0_ANCHOR_SHA256" --expected-bundle-sha256 "$FINCORE_0043_D0_EXPECTED_BUNDLE_SHA256" --expected-repository "$FINCORE_GITHUB_REPOSITORY" --expected-workflow .github/workflows/0043-baseline.yml --expected-workflow-sha "$FINCORE_0043_BOOTSTRAP_SHA" --expected-tooling-sha "$T0A_TOOLING_SHA" --expected-capability-sha "$T0B_CAPABILITY_SHA" --expected-baseline-sha "$T0C_BASELINE_SHA" --output "$FINCORE_0043_ACCEPTANCE_ROOT/trusted-shas.json"
    git worktree add --detach "$FINCORE_0043_ACCEPTANCE_ROOT/t0a" "$T0A_TOOLING_SHA"
    test "$(git -C "$FINCORE_0043_ACCEPTANCE_ROOT/t0a" rev-parse HEAD)" = "$T0A_TOOLING_SHA"
    test -f "$FINCORE_0043_ACCEPTANCE_ROOT/t0a/scripts/run_0043_cutover_gates.py"
    test -f "$FINCORE_0043_D0_TRUST_DIR/0043-d0-trust-anchor.json"
    test -f "$FINCORE_0043_D0_TRUST_DIR/0043-d0-attestation.json"
    test -f "$FINCORE_0043_D0_TRUST_DIR/0043-d0-expected-bundle.tar.zst"
    FINCORE_0043_CUTOVER_EVIDENCE=$(mktemp -d /tmp/fincore-0043-cutover-evidence.XXXXXX)
    PYTHONNOUSERSITE=1 MPLBACKEND=Agg MPLCONFIGDIR=/tmp/fincore-0043-mpl /Users/yunjinqi/opt/anaconda3/bin/conda run -n base python -I "$FINCORE_0043_ACCEPTANCE_ROOT/t0a/scripts/run_0043_cutover_gates.py" --candidate-repo "$PWD" --trust-anchor "$FINCORE_0043_D0_TRUST_DIR/0043-d0-trust-anchor.json" --trust-attestation "$FINCORE_0043_D0_TRUST_DIR/0043-d0-attestation.json" --trusted-expected-bundle "$FINCORE_0043_D0_TRUST_DIR/0043-d0-expected-bundle.tar.zst" --expected-anchor-sha256 "$FINCORE_0043_D0_ANCHOR_SHA256" --expected-bundle-sha256 "$FINCORE_0043_D0_EXPECTED_BUNDLE_SHA256" --deny-repo-local-expected --require-service-attestation-match --selector 'not integration_online' --profiles core factor visualization factor-visualization interactive report-pdf report-xlsx bayesian data-yahoo data-alphavantage data-pandas-datareader data-cn all --dependency-lanes minimum latest --real-html --real-pdf --provision-browser chromium --sanitize-environment --probe-mode subprocess-only --deny-network --allow-browser-loopback --assert-legacy-absent --max-ploc-ratio 0.88 --max-logical-loc-ratio 0.88 --min-duplicate-reduction 0.60 --max-time-regression 0.10 --max-p95-regression 0.15 --max-rss-regression 0.10 --min-branch-coverage 60 --evidence-dir "$FINCORE_0043_CUTOVER_EVIDENCE"

**Acceptance:**

- 旧 façade、strict profile、compat registry、旧 extras 为 0。
- 新 public snapshot 只包含 canonical 0.5 API。
- 每个 ledger operation source path 可解析且恰好一个 callable；每个 leaf capability_id 恰好一个 implementation_fingerprint。
- 旧 contract/compat tests 0 unclassified；所有保留 observable 已进入 parity/domain tests。
- D0 全部 test nodeids disposition 已核销，完整非在线 suite 与本地 governance/publish schema tests 全绿。
- exact CUTOVER_CANDIDATE_SHA 的唯一 wheel 已通过全 ledger、全部 profiles、minimum/latest、negative imports 和 real-Chromium PDF；通过后仍只能交给 T8 审核，T8 完成前不得 PR/落地。
- CUTOVER_CANDIDATE_SHA 的唯一 parent 是 ACCEPTED_T6_SHA，trusted T0C 与 EXPECTED_PR_BASE_SHA 都是其祖先；仓库不是 shallow，所有 exact object 均存在。
- 同一 exact SHA 已通过 relative/absolute LOC、duplicate、performance output-digest/budget 与 quality/coverage 比较；任何后续生产改动使整套 T7 evidence 失效。
- real PDF 的 page/bookmark/text/metadata/cleanup 与 missing-browser actionable error 均有 browser version/digest 证据。
- Migration 文档完整但 wheel 无兼容代码。

**Rollback:** 回滚整个 cutover commit；发布回滚使用上一版 0.4 artifact，不在 0.5 wheel 中恢复 shim。

**Suggested commit:** refactor!: cut over to the unified fincore 0.5 API

### Task 8: 锁定架构、减码、性能、文档与 CI 门

**Owner:** Acceptance + Quality

**Depends on:** T7。

**Files:**

- Create: docs/quality/0043-architecture-candidate.json
- Create: docs/quality/0043-performance-candidate.json
- Create: docs/quality/0043-performance-comparison.json
- Create: docs/quality/0043-quality-candidate.json
- Create: docs/quality/0043-feature-parity-candidate.json
- Create: docs/quality/0043-browser-candidate.json
- Create: docs/quality/0043-cutover-evidence-manifest.json
- Modify: docs/quality/performance-budget.md
- Modify: docs/quality/performance-methodology.md
- Modify: docs/quality/capability-inventory.md
- Modify: docs/quality/1.0-readiness.json
- Modify: docs/architecture/public-api-map.md
- Modify: docs/已实现函数索引.md
- Modify: docs/quality/1.0-readiness.md
- Modify: docs/quality/factor-benchmark-approval.md
- Modify: docs/quality/numerical-oracle-register.md
- Modify: docs/quality/license-review.md
- Modify: docs/quality/current-baseline.json
- Modify: docs/quality/current-baseline.md
- Modify: docs/quality/release-candidate-checklist.md
- Move: docs/quality/2026-08-21-unified-platform-acceptance.md -> docs/quality/history/2026-08-21-unified-platform-acceptance.md

**Steps:**

1. 接收 T7 repo 外 evidence 目录、外部 D0 trust anchor/attestation/bundle。hosted mode先从 `0043_BOOTSTRAP_SHA` 独立 worktree运行提前落地主线的 bootstrap verifier，self-hosted mode先运行预登记 digest 的仓库外 verifier；由它验签并导出 trusted T0A/T0B/T0C 后，才从 T0A_TOOLING_SHA 新建独立 clean acceptance worktree。所有权威 verify-only 检查从该 T0A worktree以 `python -I`/sanitized environment运行，candidate/T8 checkout只作为actual数据路径；再校验 manifest/evidence-dir digest、CUTOVER_CANDIDATE_SHA/tree SHA、唯一 wheel SHA、trusted tool/ledger/browser/acceptance-lock SHA 和每个 gate exit code，缺项或 source SHA 不一致立即失败。
2. 只把 T7 exact-SHA evidence 的精确 bytes materialize 为本任务列出的 0043 candidate JSON；不得重新 profile、重新解析依赖或用 T8 HEAD 生成替代证据。
3. 用 T0 frozen checker 对 materialized architecture evidence 做 verify-only：physical <= floor(D0 physical * 0.88) 且当前参考 <=43,480；logical <= floor(D0 logical * 0.88) 且当前参考 <=35,270；delegate/registry duplicate至少 -60%，forbidden legacy import/file=0，tool/schema 漂移=0。
4. verify-only performance evidence 必须先匹配 metrics/rolling/transactions/factor/risk/report 全 workload semantic output_digest，再满足 median <=10%、p95 <=15%、RSS <=10% 与 absolute dispatch/DAG budget。
5. verify-only quality evidence必须匹配 collector/schema/selector/network deny/coverage omit/完整 argv/Python/lock protocol，并满足 branch coverage >= max(60, D0)。feature/browser evidence必须证明 capability/scenario 0 missing、visual/interactive/PDF semantic 通过。
6. 任一 gate 失败时废弃 CUTOVER_CANDIDATE_SHA，回 owning tranche 修复并从 T7 全量重跑；禁止在 T8 修改 production、scripts、tests、pyproject、workflow 或阈值。
7. 更新 public API map、函数索引、capability inventory、readiness JSON/MD、performance budget/methodology、benchmark approval、numerical oracle、release checklist 与 current-baseline schema；把旧 unified-platform acceptance 移入 history，移除“Catalog 尚未落地”和 active compat/0.4 文案。
8. license/oracle 中只有逐行标注 provenance 的旧名称可保留；运行逐文件 legacy scanner，禁止 docs/quality 或 .github 目录级 allowlist。
9. 复核 CUTOVER_CANDIDATE_SHA 的 workflow 已包含 capability parity、new API snapshot、architecture/performance/quality、same-wheel profiles、real browser 与 3 OS x Python 3.11–3.14 matrix，且 integration_online 仍为显式手动/独立任务。
10. 将 candidate evidence + docs 作为唯一 T8 evidence-only child commit，记录 T8_EVIDENCE_SHA 且 parent=CUTOVER_CANDIDATE_SHA；evidence manifest同时冻结实际 PR base ref与 `EXPECTED_PR_BASE_SHA`，diff allowlist 只能是本任务列出的 docs/quality 与 API index docs。到此才允许创建 PR，CUTOVER candidate 不得单独落地。
11. 创建 PR 前先从 API 断言 `head.repo.id == base.repo.id`；fork PR 立即标记 BLOCKED，不启动 T8/T9，也禁止任何 `pull_request_target` job checkout/执行 candidate。PR CI 的受信 preflight-fetch 以 `ref: github.event.pull_request.head.sha`、`fetch-depth: 0`、`persist-credentials: false` checkout T8_EVIDENCE_SHA并生成认证full-git-bundle/source payload；execute job不再联网checkout，而从单次只读input URL下载、验签、移除credential后本地materialize完整history。事件/API `LIVE_PR_BASE_SHA` 必须等于 `EXPECTED_PR_BASE_SHA`，否则按T7 base-drift重建。在任何 T0A verifier操作前执行外部trust-anchor验证和trusted T0C→T8 ancestry。T0A workflow checker逐字段核对caller graph/blob；每个reusable `uses:`固定完整40位bootstrap SHA，并分别核验caller head/base与called identity。先证明T8只是approved docs evidence child，再按lock构建一次sdist/wheel；同一wheel交给12个matrix和3个browser cells。每个cell采用preflight-fetch/execute/verify/sign四段，preflight spec只绑定nonce/name/schema，raw upload后才由post-execution handoff绑定artifact ID/digest；verify clean runner认证bootstrap/D0/T0A并产verdict，sign只签verdict。final workflow通过Actions API核对15个真实job并逐格验签。任一cell/final/attestation、隔离/graph/base或handoff失败都回T7重建，不沿用旧evidence。

**Verify:**

下列单项命令用于定位失败；最终权威判定还必须先验证独立信任根，再调用 T0A frozen runner 的 `--verify-evidence-only`，并验证 candidate 未进入 acceptance process import path。完整块只展示hosted mode；self-hosted mode设置`FINCORE_0043_GATE=t8`并完整运行4.1节入口。

    /Users/yunjinqi/opt/anaconda3/bin/conda run -n base python scripts/check_architecture_convergence.py --baseline docs/quality/0043-architecture-baseline.json --candidate docs/quality/0043-architecture-candidate.json --verify-only --evidence-manifest docs/quality/0043-cutover-evidence-manifest.json --max-ploc 43480 --max-logical-loc 35270 --max-ploc-ratio 0.88 --max-logical-loc-ratio 0.88 --min-duplicate-reduction 0.60

    /Users/yunjinqi/opt/anaconda3/bin/conda run -n base python scripts/compare_benchmarks.py --baseline docs/quality/0043-performance-baseline.json --candidate docs/quality/0043-performance-candidate.json --max-time-regression 0.10 --max-p95-regression 0.15 --max-rss-regression 0.10 --digest-gate sha256 --report docs/quality/0043-performance-comparison.json

    /Users/yunjinqi/opt/anaconda3/bin/conda run -n base python scripts/check_performance.py

    /Users/yunjinqi/opt/anaconda3/bin/conda run -n base python scripts/check_quality_snapshot.py --snapshot docs/quality/0043-quality-candidate.json --baseline docs/quality/0043-quality-baseline.json --min-branch-coverage 60 --evidence-manifest docs/quality/0043-cutover-evidence-manifest.json

    /Users/yunjinqi/opt/anaconda3/bin/conda run -n base python scripts/check_feature_parity.py --baseline docs/quality/0043-capability-baseline.json --ledger tests/parity/fixtures/capability-ledger-0043.json --fixture-dir tests/parity/fixtures --verify-evidence docs/quality/0043-feature-parity-candidate.json --browser-evidence docs/quality/0043-browser-candidate.json --evidence-manifest docs/quality/0043-cutover-evidence-manifest.json

    /Users/yunjinqi/opt/anaconda3/bin/conda run -n base python scripts/check_legacy_references.py --maintained-doc docs/quality/1.0-readiness.json docs/quality/1.0-readiness.md docs/quality/capability-inventory.md docs/quality/current-baseline.json docs/quality/current-baseline.md docs/quality/factor-benchmark-approval.md docs/quality/performance-budget.md docs/quality/performance-methodology.md docs/quality/release-candidate-checklist.md docs/architecture/public-api-map.md docs/已实现函数索引.md --allow-provenance-lines docs/quality/license-review.md docs/quality/numerical-oracle-register.md --allow-history docs/quality/history/2026-08-21-unified-platform-acceptance.md CHANGELOG.md docs/MIGRATION-0.5.md .github/governance/0041/baseline.json .github/governance/0041/decisions.md .github/governance/0041/metrics-schema.json

    /Users/yunjinqi/opt/anaconda3/bin/conda run -n base python -m pytest -p no:rerunfailures -o addopts='' tests/quality/test_architecture_convergence.py tests/quality/test_0043_profiling_contract.py tests/benchmarks/test_profile_workloads.py tests/quality/test_check_performance_source.py tests/quality/test_check_quality_snapshot.py tests/quality/test_collect_quality_baseline.py -q --tb=short --maxfail=0

    set -euo pipefail
    test -n "$FINCORE_0041_LANDED_SHA"
    test -n "$FINCORE_0043_BOOTSTRAP_SHA"
    test -n "$T0A_TOOLING_SHA"
    test -n "$T0B_CAPABILITY_SHA"
    test -n "$T0C_BASELINE_SHA"
    test -n "$T8_EVIDENCE_SHA"
    test -n "$EXPECTED_PR_BASE_SHA"
    test -n "$LIVE_PR_BASE_SHA"
    test -n "$FINCORE_0043_D0_TRUST_DIR"
    test -n "$FINCORE_0043_D0_ANCHOR_SHA256"
    test -n "$FINCORE_0043_D0_EXPECTED_BUNDLE_SHA256"
    test -n "$FINCORE_GITHUB_REPOSITORY"
    test -n "$FINCORE_0043_BASE_REPOSITORY_ID"
    test -n "$FINCORE_0043_HEAD_REPOSITORY_ID"
    FINCORE_0043_T8_ACCEPTANCE_ROOT=$(mktemp -d /tmp/fincore-0043-t8-acceptance.XXXXXX)
    test "$(git rev-parse HEAD)" = "$T8_EVIDENCE_SHA"
    test "$(git rev-parse --is-shallow-repository)" = false
    test "$FINCORE_0043_HEAD_REPOSITORY_ID" = "$FINCORE_0043_BASE_REPOSITORY_ID"
    test "$LIVE_PR_BASE_SHA" = "$EXPECTED_PR_BASE_SHA"
    git cat-file -e "$FINCORE_0043_BOOTSTRAP_SHA^{commit}"
    git cat-file -e "$T0A_TOOLING_SHA^{commit}"
    git cat-file -e "$T0B_CAPABILITY_SHA^{commit}"
    git cat-file -e "$T0C_BASELINE_SHA^{commit}"
    git cat-file -e "$EXPECTED_PR_BASE_SHA^{commit}"
    git merge-base --is-ancestor "$FINCORE_0041_LANDED_SHA" "$FINCORE_0043_BOOTSTRAP_SHA"
    git merge-base --is-ancestor "$FINCORE_0043_BOOTSTRAP_SHA" "$T0A_TOOLING_SHA"
    git merge-base --is-ancestor "$T0A_TOOLING_SHA" "$T0B_CAPABILITY_SHA"
    git merge-base --is-ancestor "$T0B_CAPABILITY_SHA" "$T0C_BASELINE_SHA"
    git merge-base --is-ancestor "$T0C_BASELINE_SHA" "$T8_EVIDENCE_SHA"
    git merge-base --is-ancestor "$EXPECTED_PR_BASE_SHA" "$T8_EVIDENCE_SHA"
    git worktree add --detach "$FINCORE_0043_T8_ACCEPTANCE_ROOT/bootstrap" "$FINCORE_0043_BOOTSTRAP_SHA"
    /Users/yunjinqi/opt/anaconda3/bin/conda run -n base python -I "$FINCORE_0043_T8_ACCEPTANCE_ROOT/bootstrap/scripts/verify_0043_bootstrap_trust.py" --anchor "$FINCORE_0043_D0_TRUST_DIR/0043-d0-trust-anchor.json" --attestation "$FINCORE_0043_D0_TRUST_DIR/0043-d0-attestation.json" --bundle "$FINCORE_0043_D0_TRUST_DIR/0043-d0-expected-bundle.tar.zst" --expected-anchor-sha256 "$FINCORE_0043_D0_ANCHOR_SHA256" --expected-bundle-sha256 "$FINCORE_0043_D0_EXPECTED_BUNDLE_SHA256" --expected-repository "$FINCORE_GITHUB_REPOSITORY" --expected-workflow .github/workflows/0043-baseline.yml --expected-workflow-sha "$FINCORE_0043_BOOTSTRAP_SHA" --expected-tooling-sha "$T0A_TOOLING_SHA" --expected-capability-sha "$T0B_CAPABILITY_SHA" --expected-baseline-sha "$T0C_BASELINE_SHA" --output "$FINCORE_0043_T8_ACCEPTANCE_ROOT/trusted-shas.json"
    git worktree add --detach "$FINCORE_0043_T8_ACCEPTANCE_ROOT/t0a" "$T0A_TOOLING_SHA"
    PYTHONNOUSERSITE=1 /Users/yunjinqi/opt/anaconda3/bin/conda run -n base python -I "$FINCORE_0043_T8_ACCEPTANCE_ROOT/t0a/scripts/run_0043_cutover_gates.py" --candidate-repo "$PWD" --trust-anchor "$FINCORE_0043_D0_TRUST_DIR/0043-d0-trust-anchor.json" --trust-attestation "$FINCORE_0043_D0_TRUST_DIR/0043-d0-attestation.json" --trusted-expected-bundle "$FINCORE_0043_D0_TRUST_DIR/0043-d0-expected-bundle.tar.zst" --expected-anchor-sha256 "$FINCORE_0043_D0_ANCHOR_SHA256" --expected-bundle-sha256 "$FINCORE_0043_D0_EXPECTED_BUNDLE_SHA256" --deny-repo-local-expected --require-service-attestation-match --verify-evidence-only --evidence-manifest "$PWD/docs/quality/0043-cutover-evidence-manifest.json" --sanitize-environment --probe-mode subprocess-only

**Acceptance:**

- T7 已完成 compat 删除，T8 的 negative audit 未发现残留或隐藏 shim。
- physical/logical/delegate 三个门都通过。
- benchmark 每项有 input_digest 与 semantic output_digest，relative/absolute budgets 都通过。
- quality candidate 的 selector/config 与 D0 一致，branch coverage >= max(60%, D0)。
- 除 Task -1 独立 bootstrap 外，cutover CI 变更只发生在 0041_LANDED_SHA、D-1 mode evidence 与 D0 sources 均已记录并验证之后；hosted 检查 0043_BOOTSTRAP_SHA ancestry，self-hosted 检查 runner/signer contract digest，bootstrap 与 cutover职责分开审计。
- T8 materialized evidence 全部指向同一 CUTOVER_CANDIDATE_SHA/tree/wheel；T8 commit 是其 docs-only child，0 production/script/test/config diff，T8 前未创建/落地 cutover PR（Task -1 bootstrap 例外已单独审计）。
- PR checkout的实际 HEAD 恰为 T8_EVIDENCE_SHA，LIVE_PR_BASE_SHA 恰为冻结的 EXPECTED_PR_BASE_SHA；trusted T0C与base都是T8祖先，任一base drift都未被未测试merge掩盖。
- active quality/governance docs 0 stale legacy reference；历史与 provenance 豁免均为逐文件/逐行 allowlist。
- PR merge gate 保留3 OS x Python 3.11–3.14 same-wheel matrix；12个matrix和3个browser cell均通过固定40位bootstrap SHA的reusable workflow调用，各有隔离preflight-fetch/execute/verify/sign jobs、post-execution handoff、独立attestation并与Actions API真实job结论一致；caller T8 identity 与 called workflow identity分别匹配，`0043-final-seal`逐格验证且check名稳定。fork 或 `pull_request_target` candidate execution 为0。若尚未获准把该workflow/context配成远端required gate，0043保持BLOCKED且禁止merge。

**Rollback:** 架构/性能失败回到 owning tranche；T8 docs commit 可独立回滚，T7 的 CI/governance schema 只能随 atomic cutover 整笔回滚；不调整 baseline 或阈值掩盖失败。

**Suggested commit:** docs: materialize 0043 exact-candidate evidence and API documentation

### Task 9: 全量 source、wheel、质量与功能技术 seal（D8a）

**Owner:** Independent Acceptance

**Depends on:** T0–T8。

**Files:**

- No tracked file changes.
- External artifact: `FINAL_EVIDENCE_DIR/0043-final-acceptance.json`
- External artifact: `FINAL_EVIDENCE_DIR/0043-final-acceptance.md`
- External artifact: `FINAL_EVIDENCE_DIR/0043-final-acceptance-attestation.json`
- External artifact: `FINAL_EVIDENCE_DIR/logs/**`
- Input artifact: `FINCORE_0043_CI_DIST/{sdist,wheel,manifest}`
- Input artifact: `FINCORE_0043_MATRIX_EVIDENCE`（`matrix-manifest.json` 的绝对路径）
- Input artifact: `FINCORE_0043_CELL_ATTESTATIONS`（12 matrix + 3 browser envelopes）
- Input artifact: attested execution spec + full-git-bundle/source payload
- Transient artifact: post-execution handoff + raw evidence + canonical verdict
- Input artifact: protected external `FINCORE_0043_D0_TRUST_DIR/{anchor,attestation,expected-bundle}`
- Input checkout: detached `T0A_TOOLING_SHA` acceptance worktree + `0043-acceptance-toolchain.json`

**Steps:**

1. `0043-final-seal` caller 只能用完整40位 `0043_BOOTSTRAP_SHA` 调用 trusted reusable workflow，并分别绑定 caller repository/T8 head/base 与 called `job_workflow_ref/job_workflow_sha`；同名 branch/tag 不参与 reusable resolution。它只在 evidence-diff、single-build-dist、12份same-wheel matrix attestation与3份real-browser attestation全绿后启动，不能先等待包含自身在内的“全部 required checks”。
2. hosted final workflow固定四段。`preflight-fetch`不执行candidate，从job启动即以`contents: read/actions: read/id-token: write/attestations: write/artifact-metadata: write`核对caller/called graph和15份cell attestation，从exact T8/full history生成content-addressed full-git-bundle与source payload，并attest只绑定输入artifact IDs/digests、expected SHA、D1 input-loader digest、D0认证的T0A tree/toolchain digest、locked argv、raw name/schema和fresh execution_nonce的spec；绝不预填未来raw artifact ID/digest。`execute-final`使用disposable runner与`permissions: {}`，无secrets/OIDC/signer/API token；input-loader用一次性只读URL下载并验签输入后删除runtime credential，从D0 bundle按T0A SHA只读materialize并复核runner/tree/toolchain，再在隔离容器从local git bundle materialize T8/完整history，candidate仅作为subprocess/venv probe target。raw upload后由action outputs+Actions API形成post-execution handoff，绑定execution_nonce、run/job与actual artifact ID/digest。
3. `verify-final`使用另一台clean runner，job从启动起只有`contents: read/actions: read`，无OIDC/attestation write。它从exact bootstrap SHA独立运行bootstrap verifier，验签D0 anchor/bundle、导出trusted T0A/T0B/T0C、核对T0A tree/toolchain digest，然后才执行authenticated T0A verify-only；同时核对post-execution handoff、15份cell envelopes和raw evidence，输出content-addressed canonical verdict。`sign-final`是第四台clean runner，job从启动起具有`actions: read/id-token: write/attestations: write/artifact-metadata: write`，但不checkout或执行candidate/T0A，只验证verdict digest、verify job identity和subject字段后上传/签名。四段不共享runner/workspace/command files/process。self-hosted mode执行同样拓扑；任何对象/head/base/ancestry/spec/handoff/service attestation漂移立即失败。
4. execution job 以 CI dist 为唯一 package input，不重建或重新解析依赖；所有 expected ledger/goldens/toolchain/browser/quality/architecture/performance bytes 只从已验签外部 bundle读取。它 fail-fast 运行 parity、numerical、property、全部 domain、integration_offline、runtime、report、viz、packaging、branch coverage、Ruff、format、mypy、MkDocs strict、notice、release candidate、release consistency 和 twine check；integration_online 明确排除且 offline tests deny outbound socket。
5. execution job 对同一 CI wheel重跑全部profiles；Agg先验证renderer/font/DPI tuple再做perceptual+semantic比对。真实browser重复missing-browser与pinned Chromium两阶段，加载HTML/JS控件和PDF，验证perceptual/console/pageerror/page/bookmark/text/metadata/cleanup。provider scenarios使用injected fake，data-pandas-datareader经过真实lazy dependency entry。
6. minimum/latest 使用已冻结且带 hashes 的 resolution manifests 跑真实 ledger scenarios，不重新访问 resolver；核对3x4 matrix每格使用同一wheel SHA，三OS Python 3.11有real-browser evidence。每个record必须有受信cell sign job的GitHub artifact attestation或self-hosted signer envelope，绑定caller/called workflow、preflight/execute/verify/sign job IDs、execution_nonce/handoff、matrix tuple、head/base、wheel、argv/exit与digest；aggregate manifest不能替代逐格证明。
7. execute job生成raw JSON/MD/log manifest，不自行声明PASS或签名。verify job在bootstrap认证后用T0A verify-only重验T7/T8 feature、architecture、performance、quality、legacy evidence以及每条argv/exit/source/tree/dist/browser/dependency digest；独立reviewer核对capability 0 missing、old path 0 present、PLOC、删除清单、license provenance和逐文件豁免，成功才输出canonical verdict。任一失败停止且sign job不启动；修复必须回T7新建candidate并重跑T7–T9。
8. sign job用D-1锁定完整SHA的upload-artifact action上传canonical final JSON/MD/log manifest，再用锁定完整SHA且权限合同已实跑验证的artifact-attestation action实际生成`0043-final-acceptance-attestation.json`；该合同明确要求`artifact-metadata: write`。attestation subject绑定verdict/final JSON/bundle digest、D0、T8/base、caller/called workflow、四类job IDs、execution_nonce/handoff、dist/matrix/15 cell digests；随后用锁定GitHub CLI与bootstrap verifier验证attestation subject和verdict digest，不运行T0A。缺任何阶段/权限边界、full-git-bundle、post-execution handoff、canonical verdict或独立验签时D8a不能PASS。
9. 缺隔离job、完整40位 reusable SHA、action step、权限、attestation文件或独立验签时D8a不能PASS。`0043-final-seal`同名successful status只证明技术验收，不宣称已受远端Ruleset强制；Task10/D8b未通过前总体保持BLOCKED。本任务不创建acceptance commit，发布/tag/merge仍需单独授权。

**Verify:**

以下三段分别属于 `execute-final`、`verify-final` 与 `sign-final`；不得合并。self-hosted mode设置`FINCORE_0043_GATE=d8a`并运行第4.1节等价隔离入口。execute只消费preflight生成的本地输入，不做checkout/fetch/API调用：

    set -euo pipefail
    test -n "$D1_INPUT_LOADER_PATH"
    test -n "$D1_INPUT_LOADER_SHA256"
    test -n "$FINCORE_0043_EXECUTION_SPEC"
    test -n "$FINCORE_0043_EXECUTION_SPEC_ATTESTATION"
    test -n "$FINCORE_0043_FULL_GIT_BUNDLE"
    test -n "$FINCORE_0043_SOURCE_PAYLOAD"
    test -n "$FINCORE_0043_CI_DIST"
    test -n "$FINCORE_0043_D0_TRUST_DIR"
    test -n "$FINCORE_0043_D0_ANCHOR_SHA256"
    test -n "$FINCORE_0043_D0_EXPECTED_BUNDLE_SHA256"
    test -n "$T0A_TOOLING_SHA"
    test -n "$T8_EVIDENCE_SHA"
    test -n "$EXPECTED_PR_BASE_SHA"
    test "$(shasum -a 256 "$D1_INPUT_LOADER_PATH" | cut -d ' ' -f 1)" = "$D1_INPUT_LOADER_SHA256"
    FINCORE_0043_EXEC_ROOT=$(mktemp -d /tmp/fincore-0043-execute.XXXXXX)
    "$D1_INPUT_LOADER_PATH" materialize --execution-spec "$FINCORE_0043_EXECUTION_SPEC" --execution-spec-attestation "$FINCORE_0043_EXECUTION_SPEC_ATTESTATION" --full-git-bundle "$FINCORE_0043_FULL_GIT_BUNDLE" --source-payload "$FINCORE_0043_SOURCE_PAYLOAD" --ci-dist "$FINCORE_0043_CI_DIST" --trust-anchor "$FINCORE_0043_D0_TRUST_DIR/0043-d0-trust-anchor.json" --trusted-expected-bundle "$FINCORE_0043_D0_TRUST_DIR/0043-d0-expected-bundle.tar.zst" --expected-anchor-sha256 "$FINCORE_0043_D0_ANCHOR_SHA256" --expected-bundle-sha256 "$FINCORE_0043_D0_EXPECTED_BUNDLE_SHA256" --expected-loader-sha256 "$D1_INPUT_LOADER_SHA256" --expected-t0a-sha "$T0A_TOOLING_SHA" --materialize-t0a "$FINCORE_0043_EXEC_ROOT/trusted-t0a" --verify-t0a-tree-and-toolchain --expected-head "$T8_EVIDENCE_SHA" --expected-base "$EXPECTED_PR_BASE_SHA" --remove-runtime-credential --output-trust-manifest "$FINCORE_0043_EXEC_ROOT/trust-manifest.json" --output "$FINCORE_0043_EXEC_ROOT"
    test "$(git -C "$FINCORE_0043_EXEC_ROOT/candidate" rev-parse HEAD)" = "$T8_EVIDENCE_SHA"
    test "$(git -C "$FINCORE_0043_EXEC_ROOT/candidate" rev-parse --is-shallow-repository)" = false
    test -f "$FINCORE_0043_EXEC_ROOT/trusted-t0a/scripts/run_0043_final_gates.py"
    test -f "$FINCORE_0043_EXEC_ROOT/trust-manifest.json"
    PYTHONNOUSERSITE=1 MPLBACKEND=Agg MPLCONFIGDIR=/tmp/fincore-0043-mpl /Users/yunjinqi/opt/anaconda3/bin/conda run -n base python -I "$FINCORE_0043_EXEC_ROOT/trusted-t0a/scripts/run_0043_final_gates.py" --mode execute-untrusted --execution-spec "$FINCORE_0043_EXECUTION_SPEC" --trust-manifest "$FINCORE_0043_EXEC_ROOT/trust-manifest.json" --candidate-repo "$FINCORE_0043_EXEC_ROOT/candidate" --ci-dist-dir "$FINCORE_0043_EXEC_ROOT/dist" --selector 'not integration_online' --profiles core factor visualization factor-visualization interactive report-pdf report-xlsx bayesian data-yahoo data-alphavantage data-pandas-datareader data-cn all --dependency-lanes minimum latest --real-html --real-pdf --sanitize-environment --probe-mode subprocess-only --deny-network --allow-browser-loopback --assert-legacy-absent --min-branch-coverage 60 --raw-evidence-dir "$FINCORE_0043_EXEC_ROOT/raw-evidence"

raw upload完成后，受信workflow用action outputs和Actions API生成`FINCORE_0043_POST_EXECUTION_HANDOFF`。verify-final从clean runner先认证bootstrap/D0/T0A，再运行verify-only：

    set -euo pipefail
    test -n "$FINCORE_0043_VERIFY_BOOTSTRAP_CHECKOUT"
    test -n "$FINCORE_0043_VERIFY_T0A_CHECKOUT"
    test -n "$FINCORE_0043_POST_EXECUTION_HANDOFF"
    test -n "$FINCORE_0043_RAW_FINAL_EVIDENCE"
    test -n "$FINCORE_0043_CELL_ATTESTATIONS"
    test -n "$FINCORE_0043_CANONICAL_VERDICT_DIR"
    "$FINCORE_0043_CI_PYTHON" -I "$FINCORE_0043_VERIFY_BOOTSTRAP_CHECKOUT/scripts/verify_0043_bootstrap_trust.py" --anchor "$FINCORE_0043_D0_TRUST_DIR/0043-d0-trust-anchor.json" --attestation "$FINCORE_0043_D0_TRUST_DIR/0043-d0-attestation.json" --bundle "$FINCORE_0043_D0_TRUST_DIR/0043-d0-expected-bundle.tar.zst" --expected-anchor-sha256 "$FINCORE_0043_D0_ANCHOR_SHA256" --expected-bundle-sha256 "$FINCORE_0043_D0_EXPECTED_BUNDLE_SHA256" --expected-workflow-sha "$FINCORE_0043_BOOTSTRAP_SHA" --expected-tooling-sha "$T0A_TOOLING_SHA" --verify-t0a-tree "$FINCORE_0043_VERIFY_T0A_CHECKOUT" --require-service-attestation-match --output "$RUNNER_TEMP/fincore-0043-verify-trusted-shas.json"
    "$FINCORE_0043_CI_PYTHON" -I "$FINCORE_0043_VERIFY_T0A_CHECKOUT/scripts/run_0043_final_gates.py" --mode verify-post-execution-handoff --execution-spec "$FINCORE_0043_EXECUTION_SPEC" --post-execution-handoff "$FINCORE_0043_POST_EXECUTION_HANDOFF" --raw-evidence-dir "$FINCORE_0043_RAW_FINAL_EVIDENCE" --cell-attestation-dir "$FINCORE_0043_CELL_ATTESTATIONS" --require-attested-cells 15 --trust-anchor "$FINCORE_0043_D0_TRUST_DIR/0043-d0-trust-anchor.json" --trusted-expected-bundle "$FINCORE_0043_D0_TRUST_DIR/0043-d0-expected-bundle.tar.zst" --expected-head "$T8_EVIDENCE_SHA" --expected-base-sha "$EXPECTED_PR_BASE_SHA" --deny-candidate-expected --emit-canonical-verdict "$FINCORE_0043_CANONICAL_VERDICT_DIR/0043-final-acceptance.json"

sign-final不运行candidate/T0A；它只使用预登记的最小verdict verifier核对digest/identity，随后执行pinned upload/attest actions并交叉验签：

    set -euo pipefail
    test -n "$D1_VERDICT_VERIFIER_PATH"
    test -n "$D1_VERDICT_VERIFIER_SHA256"
    test "$(shasum -a 256 "$D1_VERDICT_VERIFIER_PATH" | cut -d ' ' -f 1)" = "$D1_VERDICT_VERIFIER_SHA256"
    "$D1_VERDICT_VERIFIER_PATH" verify --verdict "$FINCORE_0043_CANONICAL_VERDICT_DIR/0043-final-acceptance.json" --expected-verify-job "$FINCORE_0043_VERIFY_JOB_ID" --expected-head "$T8_EVIDENCE_SHA" --expected-base "$EXPECTED_PR_BASE_SHA" --expected-called-workflow-sha "$FINCORE_0043_BOOTSTRAP_SHA" --deny-code-execution --output "$FINCORE_0043_FINAL_EVIDENCE/verdict-verification.json"
    test -f "$FINCORE_0043_FINAL_EVIDENCE/0043-final-acceptance-attestation.json"
    "$FINCORE_0043_GH_BIN" attestation verify "$FINCORE_0043_CANONICAL_VERDICT_DIR/0043-final-acceptance.json" --repo "$FINCORE_GITHUB_REPOSITORY"
    "$D1_VERDICT_VERIFIER_PATH" verify-attestation --verdict "$FINCORE_0043_CANONICAL_VERDICT_DIR/0043-final-acceptance.json" --attestation "$FINCORE_0043_FINAL_EVIDENCE/0043-final-acceptance-attestation.json" --expected-subject-digest "$FINCORE_0043_CANONICAL_VERDICT_SHA256" --expected-workflow-run "$FINCORE_0043_FINAL_WORKFLOW_RUN_ID" --deny-code-execution

**Acceptance:**

- baseline required capability_id 与 scenario_id 0 删除。
- 每项 operation source/wheel 可执行，golden 0 未批准差异。
- old path/import/class/extra 0 present，negative imports 全部通过。
- physical/logical LOC 同时满足相对 D0 -12% 与当前 43,480/35,270 参考上限，delegate/registry duplicate 至少 -60%。
- branch coverage >=60% 且不低于 fresh D0。
- performance median/p95/RSS 未超过预算。
- wheel profiles、minimum/latest、pip check、optional dependency 与 offline provider 全绿。
- real Chromium HTML/PDF 与 missing-browser 两条路径全绿，console/pageerror/interaction/visual/PDF semantic evidence 齐全。
- Agg perceptual evidence 使用匹配的三平台 renderer/font/DPI record；所有不匹配环境均 fail/显式 N/A，不存在放宽 tolerance 的隐式通过。
- 3 OS x Python 3.11–3.14 matrix 与 final runner 使用同一 CI wheel SHA；resolver/browser/build artifacts 均有 pinned revision/hash；wheel metadata 拒绝 Python 3.15+。
- final evidence 指向 T8_EVIDENCE_SHA、CUTOVER parent 与 CI dist SHA，不含旧基线冒充，且 final worktree 前后 clean。
- actual PR head/base 精确匹配 T8_EVIDENCE_SHA/EXPECTED_PR_BASE_SHA，trusted T0C 是 T8 祖先；base drift或synthetic merge-ref checkout均不能生成D8a PASS。
- 15个prerequisite cell均有独立可验attestation，Actions API真实job graph与T0A contract完全匹配；candidate aggregate manifest不能自证。
- 15个cell及final seal均使用preflight-fetch/execute/verify/sign四段隔离；execute无token/OIDC/secrets且只从认证full-git-bundle运行，verify先认证bootstrap/D0/T0A并核对post-execution handoff，sign不checkout/执行candidate或T0A。caller T8与called bootstrap workflow identity分别验证。
- `0043-final-acceptance-attestation.json`已由有明确OIDC/attest权限的clean sign job实际生成，GitHub CLI与hash-pinned verdict verifier双重验签通过，artifact ID/URI/digest已记录。

**Rollback:** 最终 seal 不修实现、不写 tracked 文件。失败项回到 T7/owning tranche并使旧 evidence 全部失效；若已发布需要回滚，恢复上一版 0.4 artifact，不向 0.5 注入兼容 shim。

**Suggested commit:** none；D8a 是 signed external technical artifact + check run，不创建会改变 release SHA 的验收提交。

### Task 10: 验证真实 PR 的远端 merge enforcement（D8b）

**Owner:** Governance + Independent Acceptance

**Depends on:** T9/D8a PASS；真实 0043 PR 已存在且 `head.repo.id == base.repo.id`；用户已对D8b live capture-and-attest capability签发明确授权。hosted broker/App/proxy/atomic ledger或self-hosted等价设施均已通过D-1。若Ruleset/branch-protection需要变更，还必须另有针对具体repository/base/rule ID的写授权；前一项授权不自动包含后一项。

**Files:**

- No tracked file changes.
- External artifact: `REMOTE_EVIDENCE_DIR/0043-remote-enforcement.json`
- External artifact: `REMOTE_EVIDENCE_DIR/0043-remote-enforcement-attestation.json`
- External artifact: `REMOTE_EVIDENCE_DIR/0043-remote-merge-recheck.json`
- External artifact: `REMOTE_EVIDENCE_DIR/0043-remote-merge-recheck-attestation.json`
- External artifact: broker-signed authorization capability + atomic consumption receipt
- Input artifact: external D8a `0043-final-acceptance.json` + `0043-final-acceptance-attestation.json`
- Input artifact: protected external `FINCORE_0043_D0_TRUST_DIR/{anchor,attestation,expected-bundle}`

**Steps:**

1. 先复用 D8a 的 mode-specific 信任根：hosted mode从 `0043_BOOTSTRAP_SHA` 运行提前落地主线的 bootstrap verifier，self-hosted mode运行预登记 digest 的仓库外 verifier；验签 D0 anchor/attestation/bundle并取得 trusted T0A/T0B/T0C。要求完整 history、workspace HEAD=T8_EVIDENCE_SHA、trusted T0C→T8、live base tip=EXPECTED_PR_BASE_SHA且base→T8；任一漂移立即 BLOCKED。
2. 从 authenticated T0A worktree运行 frozen final runner 的 `--verify-attested-final-evidence`，验证 D8a JSON及其attestation、CI dist/matrix digest、workflow ID/path/ref/run SHA、expected GitHub App ID与D0 digest。只有 D8a signature chain完整才可查询远端 merge enforcement。
3. 从 0041 已确认的实际 branch role/PR base解析 repository ID、base ref/SHA、PR number、T8_EVIDENCE_SHA、`0043-final-seal` check-run ID、expected GitHub App/integration ID、workflow ID/path/ref/run SHA；先断言 `head.repo.id == base.repo.id`，fork立即BLOCKED。禁止使用示例PR、旧head、synthetic merge SHA、`pull_request_target`执行结果或其他workflow产生的同名check。
4. 优先要求 Ruleset 的 required workflow绑定 frozen final workflow ID/path与实际 base；若平台/套餐只能用 required status check，则规则必须同时绑定 context与expected App/integration ID，frozen verifier拒绝同一 head上任何重复 `0043-final-seal` context/check-suite。若远端尚未满足，先报告精确缺口并停止；只有取得具体写授权后，治理 owner才能按0041流程做最小变更，不允许admin merge/bypass代替。
5. 从 authenticated T0A worktree运行 frozen `check_remote_enforcement.py`，经D-1预登记GET-only proxy做本地诊断抓取。proxy内部短时installation token可以具有读取完整Ruleset bypass信息所需的`administration: write`permission，但只接受固定allowlist的GET；原始token不进入checker环境。checker要求响应显式包含`bypass_actors`并验证exact head/base、unique workflow/check identity、conclusion=success、rule active、无bypass、0 pending evidence；字段缺失即fail closed。该preflight不作为待签snapshot，也不创建artifact。
6. hosted mode先由protected broker依据本次人工approval与可信event identity签发capability，payload绑定repository/PR/head/base/actor_id/expiry/唯一jti、D0/D8a digest、目标workflow ID和immutable dispatch tag。requester只提交该capability，不自报actor/expiry，也不持有GitHub token。broker在外部strongly-consistent ledger对`(repository, actor_id, jti)`原子CAS后，再在同一受信事务路径核对tag object=bootstrap SHA、bootstrap branch/tag/exact collision branch三份ruleset ID/digest均与D0一致、0 bypass、tag与collision ref禁止create/update/delete且同名branch不存在，随后才以`FINCORE_0043_BOOTSTRAP_DISPATCH_TAG`触发frozen `0043-remote-attestation.yml`。CAS loser、规则漂移、重放或过期请求不得运行signer。
7. remote workflow不执行candidate，并拆成capture/verify/sign隔离jobs：capture只能经GET-only proxy读取live状态，无OIDC/attestation write；verify在clean runner认证bootstrap/D0/T0A、D8a和atomic receipt后输出canonical snapshot verdict，同样无OIDC/write；sign job不执行candidate/T0A或查询live API，只以job启动时固定的`actions: read/id-token: write/attestations: write/artifact-metadata: write`签verdict。requester验证workflow ID/path、actual `run.head_sha=0043_BOOTSTRAP_SHA`、artifact ID/digest。self-hosted mode使用签名gate spec、等价原子ledger和隔离capture/verify/signer runner。envelope绑定D0、D8a、snapshot、PR/head/base/rule/check-run/app、dispatch tag/三份ruleset/run、actor_id/jti/expiry以及ledger consumption receipt/version/digest；checker只验证receipt，不在客户端做“consume nonce”。
8. frozen checker验签remote envelope和atomic consumption receipt后，立即经GET-only proxy再次拉取live状态；要求PR head/base、ETag、rule digest、check-run/workflow identity、完整bypass字段与pending-evidence均和signed snapshot相同。任何状态变化都使D8b失效；base漂移回T7重建，其他状态修复后重新capture/sign/recheck，旧jti不得复用。
9. merge前不复用本地temp/worktree。D8b 阶段调用独立protected `0043-merge-recheck.yml`（self-hosted为预登记 `merge_recheck` gate），由它在新runner上fresh fetch完整history、重新验bootstrap/D0/D8a/D8b、下载immutable artifacts、创建detached T0A并做live recheck；broker签发的capability预先绑定唯一inspection `receipt_jti`，隔离sign job只能签出绑定当前repository/PR/head/base/rule/check、request_jti、actor_id与该`receipt_jti`的短TTL receipt。hash-pinned external verifier必须先验capability/digest，再把这些字段逐项与D8b expected值比较，跨请求receipt不可串用。D8b只验证inspection receipt及独立test receipt的CAS并发/重放语义，不消费真实merge receipt，也不执行merge。0043 的授权边界到此结束。
10. 真正merge是另一次显式授权和独立受保护操作，不属于D8b自动动作。hosted模式必须先通过GitHub API证明repository owner为Organization且当前套餐/规则支持merge queue；当前 `cloudQuant` 是User owner，因此本计划的真实merge路径明确保持BLOCKED，不把不可用的queue写成已具备能力。组织迁移会改变repository identity、OIDC/App installation、allowlist、ruleset与service attestation，必须视为新的trust epoch：先撤销旧capability/receipt与service records，重做hosted D-1登记/规则验收并重新签发D0 anchor+attestation；只有commit、expected-bundle bytes与全部身份都重新验证后，bundle payload才可复用。随后必须在新repository身份下重跑并重签D8a，再重新捕获D8b。只有完成这条链，或由后续独立计划证明等价的服务端strict/up-to-date方案，才可启用merge queue并让`merge_group`候选重新通过同一final seal，从服务端绑定实际base tip、rule与checks；不得静默回退为普通merge。届时broker也不能宣称外部ledger与GitHub API跨系统原子：它持久化`operation_id`及`unused -> reserved -> committing -> consumed|indeterminate`，每次状态转换绑定repository/PR/head/base/rule/check/merge-group SHA。API超时、进程崩溃或响应歧义一律进入`indeterminate`；独立reconciler查询`merged_at`、`merge_commit_sha`、head/base/merge-group/rule/check后再finalize，只有确认未合并且reservation已取消才可`invalidated`。self-hosted若不能提供等价服务端strict queue与reconciler，也保持BLOCKED。request capability jti、inspection/test/real receipt_jti与merge operation_id分名、分ledger状态；禁止“先consume、以后再merge”。

**Verify:**

以下完整块只适用于hosted mode；self-hosted mode设置`FINCORE_0043_GATE=d8b`、actual head/base及验签过的gate-spec envelope后完整运行4.1节入口，由隔离capture/signer runner执行同一deep verification/live capture/recheck，不得引用bootstrap/requester变量。hosted requester通过broker API提交capability，环境中没有raw App/installation/Actions token：

    set -euo pipefail
    test -n "$FINCORE_0041_LANDED_SHA"
    test -n "$FINCORE_0043_BOOTSTRAP_SHA"
    test -n "$FINCORE_0043_BOOTSTRAP_BRANCH_REF"
    test -n "$FINCORE_0043_BOOTSTRAP_DISPATCH_TAG"
    test -n "$FINCORE_0043_BOOTSTRAP_TAG_RULESET_ID"
    test -n "$FINCORE_0043_BOOTSTRAP_TAG_RULESET_DIGEST"
    test -n "$FINCORE_0043_COLLISION_BRANCH_RULESET_ID"
    test -n "$FINCORE_0043_COLLISION_BRANCH_RULESET_DIGEST"
    test -n "$T0A_TOOLING_SHA"
    test -n "$T0B_CAPABILITY_SHA"
    test -n "$T0C_BASELINE_SHA"
    test -n "$FINCORE_0043_D0_TRUST_DIR"
    test -n "$FINCORE_0043_D0_ANCHOR_SHA256"
    test -n "$FINCORE_0043_D0_EXPECTED_BUNDLE_SHA256"
    test -n "$FINCORE_GITHUB_REPOSITORY"
    test -n "$FINCORE_0043_PR_NUMBER"
    test -n "$FINCORE_0043_PR_BASE"
    test -n "$EXPECTED_PR_BASE_SHA"
    test -n "$FINCORE_0043_EXPECTED_APP_ID"
    test -n "$FINCORE_0043_EXPECTED_WORKFLOW_ID"
    test -n "$FINCORE_0043_EXPECTED_WORKFLOW_PATH"
    test -n "$FINCORE_0043_EXPECTED_WORKFLOW_REF"
    test -n "$T8_EVIDENCE_SHA"
    test -n "$FINCORE_0043_D8A_EVIDENCE"
    test -n "$FINCORE_0043_D8A_ATTESTATION"
    test -n "$FINCORE_0043_D8A_ARTIFACT_ID"
    test -n "$FINCORE_0043_D8A_ARTIFACT_SHA256"
    test -n "$FINCORE_0043_REMOTE_AUTH_CAPABILITY"
    test -n "$FINCORE_0043_REMOTE_AUTH_CAPABILITY_SHA256"
    test -n "$FINCORE_0043_REMOTE_AUTH_JTI"
    test -n "$FINCORE_0043_BROKER_URI"
    test -n "$FINCORE_0043_GET_ONLY_PROXY_URI"
    test -n "$FINCORE_0043_REMOTE_ATTEST_WORKFLOW_ID"
    test -n "$FINCORE_0043_EXPECTED_RULE_DIGEST"
    test -n "$FINCORE_0043_EXPECTED_CHECK_RUN_ID"
    FINCORE_0043_REMOTE_EVIDENCE=$(mktemp -d /tmp/fincore-0043-remote-evidence.XXXXXX)
    FINCORE_0043_REMOTE_ACCEPTANCE=$(mktemp -d /tmp/fincore-0043-remote-acceptance.XXXXXX)
    test "$(git rev-parse HEAD)" = "$T8_EVIDENCE_SHA"
    test "$(git rev-parse --is-shallow-repository)" = false
    git cat-file -e "$FINCORE_0043_BOOTSTRAP_SHA^{commit}"
    git cat-file -e "$T0A_TOOLING_SHA^{commit}"
    git cat-file -e "$T0B_CAPABILITY_SHA^{commit}"
    git cat-file -e "$T0C_BASELINE_SHA^{commit}"
    git cat-file -e "$EXPECTED_PR_BASE_SHA^{commit}"
    git merge-base --is-ancestor "$FINCORE_0041_LANDED_SHA" "$FINCORE_0043_BOOTSTRAP_SHA"
    git merge-base --is-ancestor "$FINCORE_0043_BOOTSTRAP_SHA" "$T0A_TOOLING_SHA"
    git merge-base --is-ancestor "$T0A_TOOLING_SHA" "$T0B_CAPABILITY_SHA"
    git merge-base --is-ancestor "$T0B_CAPABILITY_SHA" "$T0C_BASELINE_SHA"
    git merge-base --is-ancestor "$T0C_BASELINE_SHA" "$T8_EVIDENCE_SHA"
    git merge-base --is-ancestor "$EXPECTED_PR_BASE_SHA" "$T8_EVIDENCE_SHA"
    git worktree add --detach "$FINCORE_0043_REMOTE_ACCEPTANCE/bootstrap" "$FINCORE_0043_BOOTSTRAP_SHA"
    /Users/yunjinqi/opt/anaconda3/bin/conda run -n base python -I "$FINCORE_0043_REMOTE_ACCEPTANCE/bootstrap/scripts/verify_0043_bootstrap_trust.py" --anchor "$FINCORE_0043_D0_TRUST_DIR/0043-d0-trust-anchor.json" --attestation "$FINCORE_0043_D0_TRUST_DIR/0043-d0-attestation.json" --bundle "$FINCORE_0043_D0_TRUST_DIR/0043-d0-expected-bundle.tar.zst" --expected-anchor-sha256 "$FINCORE_0043_D0_ANCHOR_SHA256" --expected-bundle-sha256 "$FINCORE_0043_D0_EXPECTED_BUNDLE_SHA256" --expected-repository "$FINCORE_GITHUB_REPOSITORY" --expected-workflow .github/workflows/0043-baseline.yml --expected-workflow-sha "$FINCORE_0043_BOOTSTRAP_SHA" --expected-tooling-sha "$T0A_TOOLING_SHA" --expected-capability-sha "$T0B_CAPABILITY_SHA" --expected-baseline-sha "$T0C_BASELINE_SHA" --require-service-attestation-match --output "$FINCORE_0043_REMOTE_ACCEPTANCE/trusted-shas.json"
    git worktree add --detach "$FINCORE_0043_REMOTE_ACCEPTANCE/t0a" "$T0A_TOOLING_SHA"
    test -f "$FINCORE_0043_D8A_EVIDENCE"
    test -f "$FINCORE_0043_D8A_ATTESTATION"
    /Users/yunjinqi/opt/anaconda3/bin/conda run -n base python -I "$FINCORE_0043_REMOTE_ACCEPTANCE/t0a/scripts/run_0043_final_gates.py" --verify-attested-final-evidence --candidate-repo "$PWD" --trust-anchor "$FINCORE_0043_D0_TRUST_DIR/0043-d0-trust-anchor.json" --trust-attestation "$FINCORE_0043_D0_TRUST_DIR/0043-d0-attestation.json" --trusted-expected-bundle "$FINCORE_0043_D0_TRUST_DIR/0043-d0-expected-bundle.tar.zst" --expected-anchor-sha256 "$FINCORE_0043_D0_ANCHOR_SHA256" --expected-bundle-sha256 "$FINCORE_0043_D0_EXPECTED_BUNDLE_SHA256" --technical-evidence "$FINCORE_0043_D8A_EVIDENCE" --technical-attestation "$FINCORE_0043_D8A_ATTESTATION" --expected-head "$T8_EVIDENCE_SHA" --expected-base-sha "$EXPECTED_PR_BASE_SHA" --expected-app-id "$FINCORE_0043_EXPECTED_APP_ID" --expected-workflow-id "$FINCORE_0043_EXPECTED_WORKFLOW_ID" --expected-workflow-path "$FINCORE_0043_EXPECTED_WORKFLOW_PATH" --expected-workflow-ref "$FINCORE_0043_EXPECTED_WORKFLOW_REF" --require-service-attestation-match --deny-repo-local-expected
    /Users/yunjinqi/opt/anaconda3/bin/conda run -n base python -I "$FINCORE_0043_REMOTE_ACCEPTANCE/t0a/scripts/check_remote_enforcement.py" --repository "$FINCORE_GITHUB_REPOSITORY" --pr "$FINCORE_0043_PR_NUMBER" --expected-base-ref "$FINCORE_0043_PR_BASE" --expected-base-sha "$EXPECTED_PR_BASE_SHA" --expected-head "$T8_EVIDENCE_SHA" --expected-app-id "$FINCORE_0043_EXPECTED_APP_ID" --expected-workflow-id "$FINCORE_0043_EXPECTED_WORKFLOW_ID" --expected-workflow-path "$FINCORE_0043_EXPECTED_WORKFLOW_PATH" --expected-workflow-ref "$FINCORE_0043_EXPECTED_WORKFLOW_REF" --required-context 0043-final-seal --prefer-required-workflow --allow-required-status-with-app-binding --deny-duplicate-context --technical-evidence "$FINCORE_0043_D8A_EVIDENCE" --technical-attestation "$FINCORE_0043_D8A_ATTESTATION" --require-active-rule --require-explicit-bypass-actors-field --deny-bypass --deny-pending-evidence --require-success --read-only --github-read-proxy "$FINCORE_0043_GET_ONLY_PROXY_URI" --output "$FINCORE_0043_REMOTE_EVIDENCE/client-preflight.json"
    /Users/yunjinqi/opt/anaconda3/bin/conda run -n base python -I "$FINCORE_0043_REMOTE_ACCEPTANCE/t0a/scripts/request_0043_remote_attestation.py" --mode hosted --broker-uri "$FINCORE_0043_BROKER_URI" --authorization-capability "$FINCORE_0043_REMOTE_AUTH_CAPABILITY" --expected-capability-sha256 "$FINCORE_0043_REMOTE_AUTH_CAPABILITY_SHA256" --expected-jti "$FINCORE_0043_REMOTE_AUTH_JTI" --repository "$FINCORE_GITHUB_REPOSITORY" --workflow .github/workflows/0043-remote-attestation.yml --workflow-dispatch-tag "$FINCORE_0043_BOOTSTRAP_DISPATCH_TAG" --expected-tag-ruleset-id "$FINCORE_0043_BOOTSTRAP_TAG_RULESET_ID" --expected-tag-ruleset-digest "$FINCORE_0043_BOOTSTRAP_TAG_RULESET_DIGEST" --expected-collision-branch-ruleset-id "$FINCORE_0043_COLLISION_BRANCH_RULESET_ID" --expected-collision-branch-ruleset-digest "$FINCORE_0043_COLLISION_BRANCH_RULESET_DIGEST" --require-collision-branch-creation-restricted --require-immutable-tag --deny-same-name-branch --expected-workflow-id "$FINCORE_0043_REMOTE_ATTEST_WORKFLOW_ID" --expected-run-head "$FINCORE_0043_BOOTSTRAP_SHA" --pr "$FINCORE_0043_PR_NUMBER" --expected-base-ref "$FINCORE_0043_PR_BASE" --expected-base-sha "$EXPECTED_PR_BASE_SHA" --expected-head "$T8_EVIDENCE_SHA" --technical-artifact-id "$FINCORE_0043_D8A_ARTIFACT_ID" --technical-artifact-sha256 "$FINCORE_0043_D8A_ARTIFACT_SHA256" --require-atomic-consumption --capture-and-attest --wait --output-snapshot "$FINCORE_0043_REMOTE_EVIDENCE/signed-snapshot.json" --output-consumption-receipt "$FINCORE_0043_REMOTE_EVIDENCE/authorization-consumption-receipt.json" --output-attestation "$FINCORE_0043_REMOTE_EVIDENCE/0043-remote-enforcement-attestation.json"
    /Users/yunjinqi/opt/anaconda3/bin/conda run -n base python -I "$FINCORE_0043_REMOTE_ACCEPTANCE/t0a/scripts/check_remote_enforcement.py" --repository "$FINCORE_GITHUB_REPOSITORY" --pr "$FINCORE_0043_PR_NUMBER" --expected-base-ref "$FINCORE_0043_PR_BASE" --expected-base-sha "$EXPECTED_PR_BASE_SHA" --expected-head "$T8_EVIDENCE_SHA" --expected-app-id "$FINCORE_0043_EXPECTED_APP_ID" --expected-workflow-id "$FINCORE_0043_EXPECTED_WORKFLOW_ID" --expected-workflow-path "$FINCORE_0043_EXPECTED_WORKFLOW_PATH" --expected-workflow-ref "$FINCORE_0043_EXPECTED_WORKFLOW_REF" --required-context 0043-final-seal --prefer-required-workflow --allow-required-status-with-app-binding --deny-duplicate-context --technical-evidence "$FINCORE_0043_D8A_EVIDENCE" --technical-attestation "$FINCORE_0043_D8A_ATTESTATION" --previous-snapshot "$FINCORE_0043_REMOTE_EVIDENCE/signed-snapshot.json" --remote-attestation "$FINCORE_0043_REMOTE_EVIDENCE/0043-remote-enforcement-attestation.json" --authorization-capability "$FINCORE_0043_REMOTE_AUTH_CAPABILITY" --atomic-consumption-receipt "$FINCORE_0043_REMOTE_EVIDENCE/authorization-consumption-receipt.json" --expected-jti "$FINCORE_0043_REMOTE_AUTH_JTI" --expected-attestation-workflow-id "$FINCORE_0043_REMOTE_ATTEST_WORKFLOW_ID" --expected-attestation-dispatch-tag "$FINCORE_0043_BOOTSTRAP_DISPATCH_TAG" --expected-tag-ruleset-digest "$FINCORE_0043_BOOTSTRAP_TAG_RULESET_DIGEST" --expected-attestation-run-head "$FINCORE_0043_BOOTSTRAP_SHA" --require-atomic-consumption-receipt --require-explicit-bypass-actors-field --require-same-etag --require-same-head-base-rule-check --require-active-rule --deny-bypass --deny-pending-evidence --require-success --read-only --github-read-proxy "$FINCORE_0043_GET_ONLY_PROXY_URI" --output "$FINCORE_0043_REMOTE_EVIDENCE/0043-remote-enforcement.json"

merge授权执行前不允许复用上述 `FINCORE_0043_REMOTE_ACCEPTANCE` 临时目录。以下 requester 只向 protected broker 提交新的 merge-recheck capability；`0043-merge-recheck.yml` 在独立新 runner fresh hydrate完整信任链并生成短TTL、`purpose=inspection_only` 的attested receipt。本块逐项验签它，并用另一枚test-only receipt实测ledger并发/重放；它既不消费真实merge receipt，也不执行merge：

    set -euo pipefail
    test -n "$D1_TRUSTED_REQUESTER_PATH"
    test -n "$D1_TRUSTED_REQUESTER_SHA256"
    test -n "$FINCORE_0043_MERGE_RECHECK_CAPABILITY"
    test -n "$FINCORE_0043_MERGE_RECHECK_CAPABILITY_SHA256"
    test -n "$FINCORE_0043_MERGE_REQUEST_JTI"
    test -n "$FINCORE_0043_MERGE_ACTOR_ID"
    test -n "$FINCORE_0043_MERGE_RECEIPT_JTI"
    test -n "$FINCORE_0043_CAS_TEST_RECEIPT_JTI"
    test -n "$FINCORE_0043_MERGE_RECHECK_WORKFLOW_ID"
    test -n "$FINCORE_0043_D0_ANCHOR_ARTIFACT_ID"
    test -n "$FINCORE_0043_D8B_ARTIFACT_ID"
    test -n "$FINCORE_0043_D8B_ARTIFACT_SHA256"
    test -n "$D1_MERGE_RECEIPT_VERIFIER_PATH"
    test -n "$D1_MERGE_RECEIPT_VERIFIER_SHA256"
    test "$(shasum -a 256 "$D1_TRUSTED_REQUESTER_PATH" | cut -d ' ' -f 1)" = "$D1_TRUSTED_REQUESTER_SHA256"
    test "$(shasum -a 256 "$D1_MERGE_RECEIPT_VERIFIER_PATH" | cut -d ' ' -f 1)" = "$D1_MERGE_RECEIPT_VERIFIER_SHA256"
    "$D1_TRUSTED_REQUESTER_PATH" request-merge-recheck --broker-uri "$FINCORE_0043_BROKER_URI" --merge-capability "$FINCORE_0043_MERGE_RECHECK_CAPABILITY" --expected-capability-sha256 "$FINCORE_0043_MERGE_RECHECK_CAPABILITY_SHA256" --expected-request-jti "$FINCORE_0043_MERGE_REQUEST_JTI" --expected-receipt-jti "$FINCORE_0043_MERGE_RECEIPT_JTI" --expected-actor-id "$FINCORE_0043_MERGE_ACTOR_ID" --workflow .github/workflows/0043-merge-recheck.yml --workflow-dispatch-tag "$FINCORE_0043_BOOTSTRAP_DISPATCH_TAG" --expected-tag-ruleset-digest "$FINCORE_0043_BOOTSTRAP_TAG_RULESET_DIGEST" --expected-collision-branch-ruleset-id "$FINCORE_0043_COLLISION_BRANCH_RULESET_ID" --expected-collision-branch-ruleset-digest "$FINCORE_0043_COLLISION_BRANCH_RULESET_DIGEST" --require-collision-branch-creation-restricted --require-immutable-tag --repository "$FINCORE_GITHUB_REPOSITORY" --pr "$FINCORE_0043_PR_NUMBER" --expected-head "$T8_EVIDENCE_SHA" --expected-base-sha "$EXPECTED_PR_BASE_SHA" --expected-rule-digest "$FINCORE_0043_EXPECTED_RULE_DIGEST" --expected-check-run-id "$FINCORE_0043_EXPECTED_CHECK_RUN_ID" --d0-anchor-artifact "$FINCORE_0043_D0_ANCHOR_ARTIFACT_ID" --d8a-artifact "$FINCORE_0043_D8A_ARTIFACT_ID" --d8b-artifact "$FINCORE_0043_D8B_ARTIFACT_ID" --receipt-purpose inspection_only --require-fresh-full-history --require-detached-t0a --require-atomic-consumption --require-service-attestation-match --max-receipt-ttl-seconds 300 --wait --output-receipt "$FINCORE_0043_REMOTE_EVIDENCE/0043-remote-merge-recheck.json" --output-attestation "$FINCORE_0043_REMOTE_EVIDENCE/0043-remote-merge-recheck-attestation.json"
    "$D1_MERGE_RECEIPT_VERIFIER_PATH" verify --merge-capability "$FINCORE_0043_MERGE_RECHECK_CAPABILITY" --expected-capability-sha256 "$FINCORE_0043_MERGE_RECHECK_CAPABILITY_SHA256" --receipt "$FINCORE_0043_REMOTE_EVIDENCE/0043-remote-merge-recheck.json" --attestation "$FINCORE_0043_REMOTE_EVIDENCE/0043-remote-merge-recheck-attestation.json" --expected-workflow-id "$FINCORE_0043_MERGE_RECHECK_WORKFLOW_ID" --expected-workflow-path .github/workflows/0043-merge-recheck.yml --expected-dispatch-tag "$FINCORE_0043_BOOTSTRAP_DISPATCH_TAG" --expected-run-head "$FINCORE_0043_BOOTSTRAP_SHA" --expected-d0-digest "$FINCORE_0043_D0_ANCHOR_SHA256" --expected-d8a-digest "$FINCORE_0043_D8A_ARTIFACT_SHA256" --expected-d8b-digest "$FINCORE_0043_D8B_ARTIFACT_SHA256" --expected-request-jti "$FINCORE_0043_MERGE_REQUEST_JTI" --expected-receipt-jti "$FINCORE_0043_MERGE_RECEIPT_JTI" --expected-actor-id "$FINCORE_0043_MERGE_ACTOR_ID" --expected-repository "$FINCORE_GITHUB_REPOSITORY" --expected-pr "$FINCORE_0043_PR_NUMBER" --expected-head "$T8_EVIDENCE_SHA" --expected-base "$EXPECTED_PR_BASE_SHA" --expected-rule-digest "$FINCORE_0043_EXPECTED_RULE_DIGEST" --expected-check-run-id "$FINCORE_0043_EXPECTED_CHECK_RUN_ID" --expected-purpose inspection_only --assert-not-merge-authorizing --max-ttl-seconds 300 --deny-revoked --output "$FINCORE_0043_REMOTE_EVIDENCE/merge-receipt-verdict.json"
    "$D1_TRUSTED_REQUESTER_PATH" exercise-receipt-cas --broker-uri "$FINCORE_0043_BROKER_URI" --test-receipt-jti "$FINCORE_0043_CAS_TEST_RECEIPT_JTI" --repository "$FINCORE_GITHUB_REPOSITORY" --test-only --no-merge --parallel 2 --expect-single-winner --expect-replay-rejected --output "$FINCORE_0043_REMOTE_EVIDENCE/merge-receipt-cas-test.json"

**Acceptance:**

- D0两层信任、D8a artifact attestation与trusted T0C→T8 ancestry全部通过；live PR head/base恰为 T8_EVIDENCE_SHA/EXPECTED_PR_BASE_SHA。
- successful `0043-final-seal` 来自唯一 expected App/workflow/check-run，且其D8a artifact digest匹配；同一head上0重复同名context。
- effective Ruleset优先强制expected required workflow；若只能强制status check，则context与App/integration ID同时绑定。无actor/admin bypass，0 `pending-evidence`，read-only verifier未产生远端mutation。
- remote capture-and-attest workflow实际从D0绑定且0-bypass不可变的bootstrap dispatch tag运行，tag object与ruleset digest匹配，run.head_sha精确匹配bootstrap SHA；Actions artifact由workflow内隔离sign job的pinned upload step产生，不使用不存在的客户端artifact-create API。
- 真实生成并验签的live snapshot envelope绑定D0、D8a、PR/head/base/rule/check-run/app/workflow ID/path/ref/run、broker actor_id/jti/expiry与atomic consumption receipt；并发相同jti恰好一个signer成功。Ruleset响应显式包含完整`bypass_actors`，GET-only proxy没有非GET调用。
- merge-time inspection recheck来自独立protected入口的新runner，重新hydrate并验签bootstrap/D0/D8a/D8b/T0A；receipt_jti由已验签capability预绑定，hash-pinned external verifier把workflow/tag/run/SHA/digests/signer/TTL、request_jti/receipt_jti、actor/repository/PR/head/base/rule/check逐项绑定到本次授权。专用test receipt证明CAS单winner与重放拒绝；inspection receipt保持非merge-authorizing且未被消费。真正merge不属于D8b；当前User-owned仓库因不满足merge queue资格而保持BLOCKED。未来组织迁移必须撤销旧信任、重做hosted D-1、重签D0并在新repository身份下重跑/重签D8a和D8b；之后才可走`merge_group` final seal及durable `committing/indeterminate` reconciler，不能把expected-head API或外部ledger描述成跨系统原子事务。

**Rollback:** 若曾获授权修改远端规则且配置错误，先阻止 merge，再按同一治理授权恢复前一份 signed live snapshot中的规则；D8a evidence可保留，但总体0043状态回到 BLOCKED。

**Suggested commit:** none；D8b 只产生仓库外 signed live evidence，远端规则 mutation 需要单独授权和审计记录。

## 6. 文件 ownership 与并行边界

| Owner | 独占路径 | 不得同时修改 |
| --- | --- | --- |
| Architecture/Acceptance | tests/parity、docs/quality/0043-*、baseline scripts | domain production files |
| Runtime | fincore/runtime、旧 api/core/results 迁移 | domain kernel |
| Metrics/Performance | fincore/metrics、fincore/performance | portfolio/factor |
| Portfolio | fincore/portfolio、旧 tearsheets/Pyfolio 迁移 | report shared renderer |
| Factor | fincore/factor_analysis、旧 alphalens 迁移 | portfolio |
| Domain Integrations | risk/attribution/simulation/optimization/data/extensions | runtime shared files |
| Reporting | report/viz/artifact/codec | domain kernel |
| Packaging/Quality | root __init__、pyproject、scripts、CI/docs | 0041_LANDED_SHA 确认前不可碰 0041 files |

T2、T4、T5 可在 T1 后并行。T1 仅获得 Metrics 对 `metrics/alpha_beta.py` TYPE_CHECKING import 的窄 handoff，用于在删除 `_types.py` 前 retarget；不得改公式。T1/D3 完成后，Runtime 将 fincore/runtime/validation.py 与 runtime/data.py 的相关 consolidation ownership 明确交给 T5；为完成 extensions/validation cutover，Runtime、Reporting、Packaging 还分别窄 handoff T5 Files 中点名的 core/context.py、core/engine.py、viz/base.py 与对应 tests，未列路径仍归原 owner。T5 同时是 fincore/contracts/validation.py、time_series.py、portfolio.py 的唯一 owner，T3/T4 对旧 contracts 只读。T3 明确等待 T2 canonical metrics；T2 完成后把 positions/transactions/round_trips/perf_attrib/perf_stats 五个旧 metrics modules 交给 T3，Runtime 只授予 T3 修改 _registry 对应五条临时映射的窄权限。D1 ledger 明确把 `utils/data_utils.py` 给 T2，把 `utils/date_utils.py`/`utils/deprecate.py` 的 retire-or-move 决策和删除权限给 T6；T6 等待 T2–T5 并独占 report/tearsheets/utils bridge。T7 是单线程 atomic cutover。每个领域将 OperationSpec 放在自己的 operations.py，只有 Runtime owner 在最终 cutover 聚合 catalog，避免多人编辑中央 registry。

每个 handoff 必须包含：

- base/head SHA
- D1 execution mode、D0 trust-anchor/bundle digest 与外部 artifact URI
- owned paths 和实际 diff
- capability/scenario 变化
- 执行命令、exit code 和日志路径
- source/wheel output digest
- PLOC/logical/delegate before/after
- 删除与保留清单
- known failures
- rollback commit

只 stage owned-path allowlist，禁止 git add .。

## 7. 删除与回滚规则

允许删除旧 source 的前置条件：

1. 对应 legacy surface 已有 disposition。
2. required capability 已有新 operation。
3. numerical/shape/plot/report scenario 已迁入 parity。
4. source 与 wheel scenario 都通过。
5. 生产消费者 rg 为 0。
6. T2–T6 的领域迁移 commit 可独立回滚；T7 的 source/test/governance 删除是单一 atomic cutover，只允许整笔 revert，不能拆分落地。
7. 每个 tranche 的完整 not integration_online suite 为绿；任何因 breaking design 失效的旧测试必须在同一 tranche 完成 disposition、语义迁移和 retarget/retire，不允许把红测试推迟到 T7/T8。

立即停止当前 tranche 的条件：

- capability_id 或 scenario_id missing。
- numerical tolerance、label、timezone、mutation isolation 未批准变化。
- 图表数据、报告章节、artifact ownership 或离线性丢失。
- optional dependency 变成 eager import。
- 为通过测试而修改 immutable golden。
- 通过恢复 shim、alias、strict profile 或双实现补洞。
- production PLOC cleanup 后未净负且没有同 milestone 偿还计划。

回滚策略：

- 实施期按 domain tranche revert，不保留半迁移 façade。
- 发布前失败直接丢弃 0.5 candidate。
- 发布后回滚恢复上一版 0.4 artifact；0.5 版本号不可复用。
- 数据/provider/report artifact 不做破坏性迁移；如未来需要 schema migration，另开迭代。

## 8. Definition of Done

0043 只有同时满足下列条件才完成：

1. D-1 hosted/self-hosted mode、T0A frozen acceptance toolchain 与三平台 attestation 均可追溯，candidate 未修改任何验收 runner/checker/schema。
2. 仓库外 D0 trust anchor/attestation/expected bundle 验签通过并绑定 T0A/T0B/T0C；所有 expected bytes 来自 trusted bundle，candidate 只提供 actual。
3. 0.5 新 API、runtime 和 ADR 已签署；没有 fincore.v2 或 compatibility shim；Python metadata 精确为 `>=3.11,<3.15` 且 3.11–3.14 classifiers/matrix 完整。
4. D0 required capability_id 与 scenario_id 一个不少。
5. 每个 leaf capability_id 恰好一个 implementation_fingerprint；它可映射一到多个共享该实现的 operation，每个 operation_id 恰好一个 canonical callable，Catalog 可解析全部 operations。
6. metrics、portfolio、performance、factor、risk、attribution、simulation、optimization、report、viz、runtime、data、extensions 全域 parity 通过。
7. Empyrical、Pyfolio、Alphalens、flat API、strict profile、legacy registry、dynamic dispatch 和旧 extras 全部删除。
8. tests/compat 的语义 case 已迁移，旧 manifests/oracles/meta-tests 已退役；测试删除不计减码。
9. production physical/logical LOC 同时满足相对 D0 -12% 与当前 43,480/35,270 参考上限，delegate/registry duplicate 至少 -60%。
10. source + 同一 wheel 的全部 profile、minimum/latest dependency lane、offline provider 和 optional dependency gate 全绿。
11. branch coverage >=60%，numerical/property/performance/static/docs/notices全绿；candidate caller graph匹配T0A contract，12个matrix+3个browser cell均有独立受信attestation，D8a final evidence已实际生成GitHub/self-hosted attestation并独立验签。
12. D8b signed live snapshot继承完整D0/D8a信任链，证明真实 PR head/base、唯一workflow/app/check-run与有效Ruleset/branch protection一致、无重复同名context、无bypass、0 pending evidence；`0043-final-seal` 是实际 merge block，签名前后及merge前ETag/rule/head/base均未漂移。

任何一项不满足，状态保持 BLOCKED；不得用“旧 API 不兼容是预期”掩盖真实 capability 缺失。

## 9. 执行交接

推荐在新的 clean worktree 中执行。开始前先复核：

1. 0041 governance/CI 是否已落到命名 commit/ref、是否已记录 0041_LANDED_SHA，且当前 clean HEAD 是否为其后代。
2. D-1 是否冻结 hosted/self-hosted mode；hosted 的 0043_BOOTSTRAP_SHA/tooling refs，或 self-hosted 的 runner-contract/signer/attestations 是否可用。
3. 实际分支角色、PR base ref/精确 SHA，以及 base drift 后必须重建而非复用旧 candidate 的约束。
4. .joyincode 规则文件是否恢复。
5. D0 baseline 是否来自各自声明的 clean T0A/T0B source，且仓库外 trust anchor/bundle 已验签、provenance 未混用。
6. 0.5.0.dev0 与 breaking policy 是否写入 ADR。

两种执行方式：

1. Subagent-Driven：在本会话按 Task -1、T0–T10 分派独立 owner，每个 tranche 后做代码审查和 parity review。
2. Parallel Session：在 clean worktree 开新任务，使用 superpowers:executing-plans 按 gate 批量执行并在 D1、D3、D7、D8a、D8b 停点验收。

未经新的实现授权，不启动 Task -1、T0 或任何实现/CI 变更；未经对应远端授权，不推送 tooling ref、不创建 PR、不修改 required checks；未经发布授权，不发布包或创建 tag。
