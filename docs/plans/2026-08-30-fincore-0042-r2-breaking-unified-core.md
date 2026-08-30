# Fincore 0042-R2 Breaking Unified Core Implementation Plan

> **For Codex:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 在不保留 Empyrical、Pyfolio、Alphalens 旧 API、旧导入路径、旧类模型和兼容壳的前提下，把现有全部有效分析能力迁入一套统一、低耦合、可复用且可度量的 Fincore 0.5 内核；在功能能力零缺失的基础上，生产 Python 代码至少减少 12%，并建立可持续优化性能与扩展领域能力的基础。

**Architecture:** 采用“一个领域内核、一个 Operation Catalog、每项能力一个 canonical 计算路径、一个结果与产物模型”。领域函数就是唯一计算与领域验证实现；直接调用返回自然值，`runtime.run` 解析后调用同一个函数并增加编排、缓存、provenance 与 Result 包装，不复制公式或领域验证。旧 façade、profile、registry、动态 wrapper、重复工作流和兼容测试只作为迁移期 oracle，最终从源码、wheel、文档、extras 和 Catalog 中全部删除。

**Tech Stack:** Python 3.11–3.14（最终支持窗口由 D0 依赖矩阵冻结）、NumPy、pandas、SciPy、statsmodels、Matplotlib、Plotly/Bokeh、pytest、Hypothesis、Ruff、mypy、setuptools/build、MkDocs。

---

## 0. 迭代身份、决策与边界

### 0.1 迭代身份

| 项目 | 决策 |
| --- | --- |
| 迭代编号 | `0042-R2` |
| 文档日期 | 2026-08-30 |
| 当前状态 | `D-ID` / `D-BREAK` 已按用户明确的 breaking-policy direction 本地通过；Task 0 可启动；`D0`、`D-TECH` 与 release 均未通过 |
| 目标版本 | `0.5.0.dev0`；这是 breaking release，不复用 `0.4.x` 的兼容承诺 |
| 参考计划 | `docs/plans/2026-08-24-fincore-structural-consolidation.md` |
| 历史 0042 | `docs/plans/2026-08-20-fincore-unified-analytics-platform.md`，已按“保留兼容 façade”路线进入历史 |
| 当前审计 HEAD | `master@2bcb65773f01dd836b5fb4d928741ff1b072179e`；只作设计参考，不是正式 D0 |

仓库已经存在并落地过一个兼容优先的 0042；其集成提交 `d319ad5ff7c47d16012be9ecfbd3b89079bbbf9f` 是当前 HEAD 的祖先，ADR-0042 也已被接受，因此不能静默重写历史。新计划使用 `0042-R2` 标识同一战略目标的第二版路线；Task -1 新增 `ADR-0042-R2`，并明确其对 `0.5+` supersede 旧 ADR，而不是修改旧 ADR 的历史结论。

`2026-08-24-fincore-structural-consolidation.md` 是本计划的设计输入，不再作为第二份并行执行计划。D-ID 通过后，只为该参考计划增加一行“Superseded by 0042-R2”状态指针，不重写其正文或历史证据；后续执行、验收和 handoff 只引用本文件。

旧的 0042 验收记录保持原样。**0042-R2 的成功不会追溯性地把旧 0042 验收改为 PASS；旧记录永久保持其当时的 BLOCKED 结论。**

### 0.2 本计划采用的破坏性决策

用户已明确不保留旧 API 兼容壳；该明确方向构成 `D-BREAK` 的本地决定。以下条目是该方向在 0042-R2 中的具体化；namespace、extras、错误与版本细节作为 Task 0/D0 的冻结输入记录，不把它们误称为已完成的 D0 或技术验收。本计划不保留下列兼容对象：

1. `fincore.empyrical`、`fincore.pyfolio`、`fincore.alphalens`。
2. 根包 flat metric API 及其动态 `__getattr__`、`__all__` 和 lazy alias。
3. Empyrical/Pyfolio 类、类方法、MRO、descriptor、state binding 和 monkeypatch 形状。
4. 旧函数路径、签名、参数顺序、默认值、`__module__` 和精确异常文案。
5. strict/enhanced profile、`PublicBinding`、`adapter_ref`、projection 和多表面 binding。
6. 旧 `METRIC_REGISTRY`、workflow registries、动态 dispatch 和 module class interception。
7. `pyfolio`、`alphalens`、`alphalens-pyfolio` 等旧 extra/profile 名。
8. 仅用于复刻上游历史怪癖、但不产生独立金融分析 observable 的行为。

最终候选中旧 import 成功是失败，不是兼容收益。不得建立 `fincore.v2`、隐藏 `sys.modules` alias、弃用 wrapper、过渡 wheel 或第三方同名 shim。

### 0.3 “功能完整”的新定义

本迭代保留的是分析能力和可观察语义，不是旧 API 的表达方式。每个 required `capability_id` 必须具备：

1. 一个清晰的领域 owner 和一个 canonical callable implementation fingerprint。
2. 至少一个新领域入口；每个 public required leaf capability/workflow 都必须有一个 `operation_id`，仅 private helper 可以不注册。
3. 已审核的公式、单位、方向、年化约定、有效输入域和随机数策略。
4. 数值 golden/oracle，包含明确 `rtol`、`atol` 或统计性质门。
5. pandas 值、shape、index、columns、name、顺序、时区和关键 dtype 契约。
6. 输入 mutation、缓存隔离、provider 注入、线程/会话隔离契约。
7. 图表的数据系列、标签、legend、轴、表格和资源 ownership 契约。
8. 报告章节、指标、单位、表格、HTML/PDF/XLSX、provenance 和 artifact 生命周期契约。
9. optional dependency 的 lazy import、缺失错误和未使用时不加载重依赖的契约。
10. source tree 与同一个候选 wheel 的等价行为证明。

旧 surface 只能被归类为：

- `required`：必须迁移为新能力与场景。
- `alias_only`：没有独立语义，必须证明与另一个 required capability 等价。
- `legacy_quirk`：只有兼容形状，没有独立分析 observable；需 product owner 与独立 reviewer 联合批准后删除。

不得用候选 Catalog 自己证明能力完整。能力全集必须来自旧公开定义、registries、manifests、文档、测试、examples、benchmarks、extras 和 wheel 内容的并集。

“功能完整”也不等于冻结已知错误。数值 expected 的权威顺序为：公开数学定义/标准或论文、独立参考实现、固定上游 oracle、性质/invariant、最后才是当前 Fincore 输出。若当前实现与独立 oracle 冲突，ledger 保留同一个 capability，但将 scenario 标为 `correction_required`；修复必须使用独立 `fix:` commit、变更说明和新旧差异证据，不得把错误输出写成 golden。现有 numerical oracle register 覆盖不足的能力必须在 D0 补齐，不能用“旧测试通过”替代金融正确性。

### 0.4 非目标

- 本轮不新增新的金融模型、指标或报告章节。
- 不为追求目录整齐而绿地重写已经有可靠 oracle 的数值算法。
- 不以压行、改短变量名、删除测试/文档、生成 Python 或把逻辑搬进 JSON 达成减码。
- 不把远端 Ruleset、GitHub App、发布凭据或 PyPI 发布作为本地技术重构的前置条件。
- 不自动 merge、push、打 tag、发布或修改远端设置。
- 不自动删除许可证和 provenance 记录；即使旧模块删除，派生实现仍需最终合规复核。

## 1. 当前事实与问题诊断

### 1.1 当前规模与重叠

当前只读盘点得到：

| 指标 | 设计时观测值 | 用途 |
| --- | ---: | --- |
| `fincore/**/*.py` 文件 | 153 | D0 前参考值 |
| production physical LOC | 49,412 | D0 前参考值 |
| 函数/方法 | 约 1,536 | 重复与职责盘点 |
| 类 | 132 | OO façade 盘点 |
| Catalog definitions | 176 | 当前投影规模 |
| Catalog bindings | 257 | 多表面重复规模 |
| public snapshot | 13 surfaces / 246 entries | 其中 244 项 `kind=unknown`，不能作 R2 证明 |

旧兼容/分派集群约 6,328 PLOC：

| 集群 | 当前 PLOC | R2 处理 |
| --- | ---: | --- |
| `empyrical.py` + `_empyrical_legacy.py` | 854 | 迁移唯一算法后删除 |
| `pyfolio.py` + `_pyfolio_impl.py` | 1,320 | 能力迁入 portfolio/report 后删除 |
| `alphalens/` | 3,032 | 非重复能力迁入 factor_analysis 后删除 |
| `_registry.py` + `_dispatch.py` | 1,122 | Catalog 成为权威后删除 |

这些只是 gross reduction 候选，不能直接整批删除。`report/compute.py`、tearsheets、context、utils 和部分 metrics 仍反向依赖旧 façade；必须先翻转内部依赖，再执行原子删除。

### 1.2 当前结构性根因

1. 根包同时暴露 flat API、Empyrical 类、Pyfolio 类、Alphalens 子包和增强领域，形成多个产品模型。
2. `_registry.py` 仍是事实上的真源，`OperationCatalog` 只是从旧 registry 投影；`capabilities.py` 又维护第三份清单。
3. 同一实现经 root/module/class/context/profile 重复绑定，增加维护量而没有增加分析能力。
4. metrics 通过动态 module surface 安装 validation/projection，真实调用链难以静态追踪。
5. Pyfolio 中大量方法只是对 metrics、tearsheets 和 report 的薄委托。
6. Alphalens 与 `factor_analysis` 有大量同名函数和重复 workflow。
7. report、results、factor tears 和 model codec 各自管理 artifact/result/provenance。
8. `api`、`core`、`contracts` 和领域代码之间存在双向依赖，内核会反向调用 façade。
9. 现有 public snapshot 会静默漏 surface，且大多数条目无法识别 callable kind。
10. 性能基准只覆盖部分 dispatch/DAG/factor workload，无法证明“平台更快”。

### 1.3 当前证据边界

- 旧 Empyrical/Pyfolio/Alphalens 聚焦语义测试实跑为 `1184 passed`；`tests/compat` 当前可收集约 1,386 个测试。它们是迁移语料，不是最终旧 API 保留门。
- 当前质量 snapshot 绑定旧 commit，branch coverage 45%，低于 60% 门，不能作为 R2 D0。
- 当前 root worktree 含用户已有的治理、CI 和文档改动；它们保持用户所有权并与 clean R2 worktree 隔离，不能封印为正式 baseline。
- `scripts/check_architecture_convergence.py` 与 `scripts/check_feature_parity.py` 当前不存在；Task 0 先以测试驱动新增，后续任务才能引用。
- `scripts/profile_workloads.py` 当前只覆盖 metrics/factor；Task 0 先扩展 rolling/transactions/risk/report 并冻结输出 digest。
- `check_performance.py` 当前 fixed budget 覆盖 dispatch 与 DAG，但缺 snapshot 门；R2 必须补齐。
- 历史 0042 审计已记录 risk、simulation、attribution 等领域的金融正确性疑点；R2 不得把这些当前输出当成自动可信的 parity expected。

因此，本文件中的当前数字都只是计划输入。正式百分比、性能比较和 coverage 必须来自 clean exact-SHA 的 fresh D0。

## 2. 目标架构

### 2.1 最终包拓扑

```text
fincore/
  __init__.py                 # 版本和领域 namespace；无 flat functions/classes
  runtime/
    specs.py                  # 唯一 OperationSpec
    catalog.py                # operation_id -> canonical callable
    builtins.py               # 唯一 composition root，lazy 聚合 domain.operations()
    data.py                   # immutable inputs/snapshots
    validation.py             # 通用边界验证
    engine.py                 # plan/run/batch
    session.py                # state/cache/isolation
    results.py                # Result/metadata/serialization
    artifacts.py              # artifact ownership/lifecycle
  metrics/                    # 纯收益、统计、比率、drawdown、rolling kernel
  performance/                # TWR/MWR/XIRR/cashflow/inference/disclosure
  portfolio/                  # positions/transactions/round trips/capacity
  factor_analysis/            # 唯一 factor prepare/model/analyze 实现
  risk/
  attribution/
  optimization/
  simulation/
  report/
    portfolio/
    factor/
    renderers/
  data/
  extensions/
  viz/
  exceptions.py
```

最终不存在：

```text
fincore/empyrical.py
fincore/_empyrical_legacy.py
fincore/pyfolio.py
fincore/_pyfolio_impl.py
fincore/alphalens/
fincore/_registry.py
fincore/_dispatch.py
fincore/_compat/
fincore/api/
fincore/core/
fincore/contracts/
fincore/results/
fincore/tearsheets/
fincore/capabilities.py
fincore/validation.py
fincore/_types.py
fincore/backends/
fincore/constants/
fincore/utils/
fincore/plugin/
fincore/hooks/
```

`api/core/contracts/results` 不是简单删除：通用 schema、snapshot、planner、result、artifact、codec 先迁入 `runtime`；领域合同与领域实现共址。只有 consumer count 为零、capability ledger 已映射且 source/wheel negative gate 可通过后才删除旧目录。

容易遗漏的 support 模块按下表处置，不能因为不在旧 `__all__` 或 Catalog 中就跳过：

| 当前模块 | 目标 owner | 目标位置 |
| --- | --- | --- |
| `backends/numpy_backend.py`、`_types.py` | Runtime | `runtime/backends.py`、`runtime/types.py` |
| `validation.py`、`contracts/validation.py` | Runtime + domains | 通用规则进 `runtime/validation.py`，金融规则进所属领域 |
| `constants/periods.py`、`interesting_periods.py` | Metrics/Portfolio | 频率与 timing 语义进入消费领域 |
| `constants/color.py`、`style.py` | Reporting | `viz` 或 `report` styles |
| `utils/math_utils.py`、`data_utils.py` | Metrics | 领域私有 numeric/data primitive |
| `utils/date_utils.py` | Metrics | financial date/frequency primitives；通用 timezone normalization 使用 runtime contract |
| `utils/common_utils.py` | Portfolio then Reporting | Task 3 先迁 portfolio helpers 并 handoff，Task 6 再迁 table/export/asset/legend；每个 helper 只有一个 owner |
| `utils/deprecate.py` | Cutover | 无独立 observable 时删除；若有诊断能力则由 runtime ledger 显式接收 |
| `data/contracts.py`、`data/snapshots.py` | Runtime + Data | 通用 snapshot 进 runtime，provider schema 留在 data |
| `plugin/`、`hooks/` | Extensions | `extensions/`，保留能力但重做 API |

### 2.2 唯一调用模型

新版本只提供三类入口：

1. **领域函数**：公开路径就是实现路径，如 `fincore.metrics.ratios.sharpe_ratio`。
2. **Runtime 执行**：`fincore.runtime.run(operation_id, inputs, config)` 和 `batch`，用于统一编排、缓存、provenance 和结果包装。
3. **领域工作流**：portfolio、factor、risk、report 下的显式 builder/analyze/render 函数。

直接领域函数返回自然 scalar/Series/DataFrame/model；`runtime.run` 解析后调用同一个公开函数，再包装为统一 `Result`。两种入口允许有不同的外层返回形状，但必须共享同一个领域 validation、计算实现和业务错误类别；runtime 只可增加 resolution/execution/resource 类诊断。不得为了 Catalog 生成第二套计算 wrapper。根包只导出版本、错误模型和领域 namespace；各领域 `__init__.py` 也不把同一个 callable 再导出一遍，公开 callable 只保留一个实现路径。

| 旧产品表面 | 新 owner | 迁移原则 |
| --- | --- | --- |
| Empyrical 指标与 rolling | `metrics` | 按公式职责分模块，只有一个函数路径 |
| Empyrical 现金流/绩效口径 | `performance` | TWR/MWR/XIRR 与披露共址 |
| Pyfolio positions/transactions/round trips | `portfolio` | 纯函数 + typed model，不创建替代 god class |
| Pyfolio tear/report | `report/portfolio` | 消费 portfolio/attribution 结果，compute model 与 renderer 分离 |
| Alphalens prepare/analyze | `factor_analysis` | 与现有增强实现合并，不保留双命名 |
| Alphalens tear/report | `report/factor` | 复用 factor compute model |
| root/class/context/profile bindings | 无 | 删除；编排需求统一进入 `runtime` |
| plugin/hooks | `extensions` | 保留扩展能力，重新定义窄接口 |

### 2.3 Catalog 真源模型

每个领域提供 `operations()`，返回不可变 `OperationSpec`：

```text
operation_id
capability_id
domain
callable
input_schema
output_schema
optional_extra
determinism_and_rng_policy
provenance
```

必须满足：

- Catalog 从领域实际 callable 聚合，不从旧 registry 投影。
- `runtime.catalog` 只定义通用 Catalog/索引，不 import 任何领域；`runtime.builtins` 是唯一 composition root，按固定清单 lazy import 各领域 `operations()` 并构建 builtin Catalog。
- 一个 `operation_id` 恰好解析到一个 callable。
- 一个 leaf `capability_id` 恰好一个 canonical implementation fingerprint，默认只对应一个 operation。
- 只有经 D-BREAK/D-DOMAIN 批准的参数化 semantic mode 才允许多个 operation IDs 共享该 fingerprint；不得用它制造别名路径或重复公开表面。
- `OperationSpec` 不包含 public path profile、string signature、adapter、projection 或 deprecated alias。
- Catalog 查询结果预构建为不可变索引，不在每次访问时重建全量映射。
- 除 `runtime.builtins` 这个受审计的 composition root 外，runtime 不 import 任何领域、report、extensions、旧 façade 或具体可选 renderer；composition root 也只能 import `operations()` metadata，不能调用领域私有实现。

扩展能力也不能形成第二真源：

- builtin Catalog 永久不可变，extension 不得覆盖 builtin `operation_id`。
- `Catalog.with_extensions(extension_snapshot)` 返回新的不可变 Catalog snapshot，不原地修改全局 singleton。
- extension operation ID 使用具名 namespace，并通过与 builtin 相同的 schema、collision、fingerprint、optional-import 和 provenance gate。
- hook/backend 若不是 operation，也必须登记在同一个不可变 `ExtensionSnapshot`；禁止保留进程级 mutable registry。
- `AnalysisSession` 在创建时固定 catalog/extension snapshot digest；缓存键与 Result provenance 都包含该 digest，运行中 registration 不得改变既有 session。

### 2.4 依赖方向

```text
domain kernels  <- runtime specs/data/validation
      ^                    |
      |                    v
domain workflows <- runtime engine/session/result/artifact
      ^                    |
      |                    v
report models --------> renderers
      ^
      |
data providers / extensions
```

硬约束：

- runtime 可以依赖通用 schema，不依赖领域公式。
- 领域 kernel 不依赖 runtime engine、report、extensions 或旧 API。
- report 只消费 canonical domain result/snapshot，不反向调用 Empyrical/Pyfolio。
- factor、portfolio、risk 可以组合 metrics/performance 公共 primitive，但禁止复制实现。
- `metrics.risk` 仅容纳无拟合状态的描述性序列风险度量；EVT/GARCH/VaR-ES forecast、calibration、backtest 和 walk-forward 归 `risk`。
- `portfolio` 只负责 positions/transactions/round trips 和 attribution input/contribution normalization；Brinson、factor/style exposure 与归因算法统一归 `attribution`。
- optional backend/renderer/provider 仅在调用对应能力时导入。
- core-only 环境必须能够构建完整 builtin Catalog；`operations.py` 与 callable 定义模块保持 dependency-neutral，可选包只在函数体或 backend factory 内加载。
- import graph 中非法 edge 和跨层 SCC/cycle 均为零。

### 2.5 统一结果与产物

- 一个 `Result[T]`：value、operation/capability identity、input digest、diagnostics、timing、provenance。
- 一个 `ArtifactBundle`：统一图、表、HTML、PDF、XLSX、interactive assets 的 ownership 与幂等 `close()`。
- 一个版本化 serializer：仅序列化稳定 schema，不 pickle session/private callable。
- 报告实行 compute-once/render-many；renderer 不重新计算金融指标。
- caller 提供 axes/resource 时不夺取 ownership；由 Fincore 创建时由 bundle 负责释放。
- source/wheel semantic digest 只包含金融值、容器/schema、稳定 diagnostics、operation/capability identity 和确定性 provenance；timing、run ID、临时/绝对路径、平台随机元数据不进入等价 digest，但必须由独立 schema/range/no-checkout-leakage gate 验证，不能简单丢弃。

### 2.6 统一错误模型

新公共错误只冻结类别和结构化诊断，不冻结旧异常名字或文案：

```text
invalid_input
insufficient_data
unsupported
optional_dependency_missing
numerical_failure
convergence_failure
provider_failure
execution_failure
resource_failure
```

每个错误包含 `category`、`operation_id`、`capability_id`、`details` 和可操作 remediation。领域 callable 从与其共址的不可变 operation metadata 注入静态 ID，因此直接调用与 `runtime.run` 产生相同业务 error identity；领域错误不得反向 import constants/metrics。

## 3. 能力、质量、性能与减码契约

### 3.1 能力 family

| family | 必须盘点和迁移的能力 |
| --- | --- |
| metrics | returns/compounding、ratios、alpha-beta/alignment、drawdown、rolling、risk、stats、consecutive、timing、bayesian |
| performance | TWR、MWR、XIRR、cashflow/fee/FX、inference、disclosure |
| portfolio | positions、transactions、round trips、perf attribution、perf stats、capacity |
| factor | prepare/forward/calendar、quantile/weights/returns/IC/turnover、PIT/multihorizon、cost/capacity、inference、events、tear workflows |
| risk | EVT、GARCH family、VaR/ES forecast、backtest、calibration、walk-forward、report |
| attribution | Brinson、Fama-French、style、provider injection/cache |
| simulation | Monte Carlo、GBM、bootstrap、scenario、path sampling |
| optimization | frontier、objectives、risk parity、constraints、feasibility |
| report | portfolio/factor builders、offline HTML、PDF、XLSX、provenance |
| viz | Matplotlib、HTML、Plotly、Bokeh、axes/resource ownership |
| runtime | snapshot/session/cache、DAG/rolling、result codec/artifact lifecycle |
| data | provider contracts、offline fake client、cache/snapshot |
| extensions | registration、discovery、isolation、hooks |

每个旧测试 nodeid 必须登记为 `migrate`、`replace` 或 `retire`。`retire` 必须指向 `alias_only/legacy_quirk` 决策；不得通过删除旧 tests 让缺失能力消失。

### 3.2 量化验收门

| 指标 | 硬门 | 目标 |
| --- | ---: | ---: |
| required capability/scenario missing | 0 | 0 |
| 未裁决 observable difference | 0 | 0 |
| leaf capability implementation fingerprint | 恰好 1 | 恰好 1 |
| operation canonical callable | 恰好 1 | 恰好 1 |
| production physical LOC | `<= floor(D0 × 0.88)` | `<= floor(D0 × 0.85)` |
| production logical LOC | `<= floor(D0 × 0.88)` | `<= floor(D0 × 0.85)` |
| normalized duplicate/delegate bodies | 相对 D0 至少减少 60% | 至少减少 80% |
| legacy façade/profile/registry/dispatch 文件 | 0 | 0 |
| legacy imports/extras/bindings/maintained executable docs refs | 0 | 0 |
| illegal import edge / cross-layer cycle | 0 / 0 | 0 / 0 |
| branch coverage | `>= max(D0, 60%)` | 同左 |
| changed measurable lines coverage | `>=95%` | 100% |
| canonical critical modules branch coverage | `>=90%` | `>=95%` |
| benchmark regression | median `<=10%`、p95 `<=15%`、RSS `<=10%` | 0 回退 |

LOC 口径由 Task 0 冻结：`fincore/**/*.py` splitlines 为 physical LOC；tokenize 后含非空白、非 comment-only token 的行是 logical LOC，docstring 计入。排除 tests/docs/assets/generated/vendor。不得把 Python 逻辑改成数据文件规避。

normalized AST duplicate 口径同样在 Task 0 冻结：去除位置、docstring、type comment，归一化局部变量和字面量，但保留调用目标、控制流和运算结构。脚本、schema、include/exclude manifest 和 SHA 都写入 D0。

### 3.3 性能兑现门

所有 benchmark 必须校验输出 digest 后计时，使用同平台、同 Python、同依赖、同数据与同线程设置，至少 2 次 warmup + 5 次 measured repeats。

绝对预算：

- Catalog resolution + runtime invocation overhead p95 `<=500µs`。
- DAG 规划 p95 `<=1ms`。
- snapshot 构建 p95 `<=10ms`。

相对回退门之外，还必须兑现至少一项真实提升：

- 至少 3 个预登记热点的 median wall time 改善 `>=20%`；或
- 至少 2 个热点改善 `>=20%`，且其中 1 个 peak RSS 改善 `>=30%`。

预登记热点从 D0 profiler 选择，至少覆盖 rolling metric、portfolio transactions/round trips、factor preparation/IC、risk forecast、report compute/render。不得在看到 candidate 结果后更换热点。

D0 必须为每个 old/new workload 冻结相同输入准备、输出 digest 和计时边界。旧 strict dispatch 只用于建立等价 capability scenario，不与新 direct-domain call 做非同构 overhead 对比；Catalog/runtime overhead 使用上面的独立绝对预算，端到端性能使用相同用户场景比较。

### 3.4 旧表面删除门

最终 source 和同一个 wheel 必须同时满足：

1. `import fincore.empyrical`、`import fincore.pyfolio`、`import fincore.alphalens` 失败。
2. wheel 中对应文件和目录不存在。
3. public API snapshot 的旧 surface/profile 项为 0。
4. wheel `METADATA` 的旧 `Provides-Extra` 项为 0；按 PEP 685 先把 hyphen/underscore/dot 归一化再做负断言，防止旧 extra 以拼写变体残留。
5. maintained docs、examples、type declarations、Catalog、entry points 中可执行旧 import/API 引用为 0；迁移指南、历史计划/验收和许可证/provenance allowlist 可保留纯文本旧名字，但不得包含可执行兼容示例。
6. `fincore` 根包不存在旧 callable/class alias。
7. 旧 compat tests/manifests 已由新 nodeid/oracle 接替，并保留只读 disposition ledger。

建议由 `tests/contracts/test_removed_legacy_surfaces.py` 对 checkout 与候选 wheel 运行同一组 negative assertions。

## 4. 决策门、里程碑和停止条件

| Gate | 问题 | 通过证据 |
| --- | --- | --- |
| D-ID Lineage Freeze | 新计划是否与历史 0042/0043 草案明确区分？ | ADR-0042-R2、R2 plan blob/commit、supersede 决策、用户明确 direction；`PASSED (local decision)` |
| D-BREAK Breaking Approval | 不保留旧 API 壳、版本、namespace、extras 是否获批准？ | 用户明确 breaking-policy direction 已记录于 ADR/plan/readiness；`PASSED (local decision)` |
| D-BASE Evidence Reset | clean exact SHA、能力、质量、架构、性能是否冻结？ | fresh D0 bundle 与 digest；尚未通过 |
| D-RUNTIME Runtime Foundation | 通用 Catalog/engine/session/result 是否可独立工作且不依赖旧 façade？ | runtime contracts + runtime family source/wheel parity |
| D-DOMAIN Domain Authority | 所有 family 是否能力零缺失、唯一实现，并已进入 builtin Catalog？ | per-family parity + complete Catalog report；此时冻结旧 registries |
| D-CUTOVER Atomic Cutover | 旧表面是否全部消失，新 API 是否稳定？ | negative imports + new API snapshot + wheel inspection |
| D-TECH Technical Seal | 功能、质量、性能、LOC、架构、package 是否同候选全绿？ | exact-SHA technical acceptance manifest |
| D-RELEASE Release Decision | 是否具备发布、合规和远端治理条件？ | 独立发布审查；不反向阻塞 D-TECH 结论 |

```text
D-ID -> D-BREAK -> D-BASE -> D-RUNTIME
                              |
                              +-> metrics/performance
                              +-> portfolio
                              +-> factor
                              +-> risk/attribution/simulation/optimization/data/extensions
                                           |
                                           v
                                  report/result/artifact
                                           |
                                           v
                                   performance cleanup
                                           |
                                           v
                                     atomic cutover
                                           |
                                           v
                                      D-TECH seal
                                           |
                                           v
                                  optional D-RELEASE review
```

出现以下任一情况立即停止，不得默默继续：

- D0 从 dirty checkout 采集，或 baseline commit/tree/toolchain 无法复现。
- 任一 legacy surface、生产模块或旧测试 nodeid 没有 disposition。
- 能力差异没有 owner/独立 reviewer 裁决。
- 新实现需要永久 shim 才能通过验收。
- 删除旧路径后仍有内部反向依赖。
- 性能 comparator 在无 baseline、pending 或平台不匹配时返回成功。
- candidate wheel 不是所有 installed/profile 检查消费的同一字节构件。

## 5. 实施任务

下列 task 中直接执行 candidate checkout 下 `scripts/...` 的命令只用于开发期快速诊断。任何 tranche handoff、D-DOMAIN、D-CUTOVER 或 D-TECH 正式结论，都必须改由 detached `D0_TOOLING_SHA` 的 `run_0042_r2_acceptance.py` 对 candidate/source/wheel 执行；candidate checker 输出不能签署 gate。

### Task -1: 冻结 0042-R2 身份、breaking policy 与准入

**Owner:** Architecture + Product + Acceptance

**Files:**

- Create: `docs/architecture/adr/0042-r2-breaking-unified-core.md`
- Create: `docs/quality/0042-r2-development-readiness.md`
- Freeze with the local decision: `docs/plans/2026-08-30-fincore-0042-r2-breaking-unified-core.md` blob SHA + containing commit in readiness evidence
- Modify only in a separately scoped status task: `docs/迭代计划/README.md`
- Modify status pointer only in a separately scoped status task: `docs/plans/2026-08-24-fincore-structural-consolidation.md`
- Preserve unchanged: `docs/architecture/adr/0042-unified-operation-model.md`
- Preserve unchanged: `docs/quality/2026-08-21-unified-platform-acceptance.md`

**Steps:**

1. ADR 写明 `Supersedes ADR-0042 for Fincore 0.5+`，旧 ADR 对 0.4 历史仍有效。
2. 记录用户明确的“保留功能能力、不保留旧 API surface”本地决定，并将 D-ID/D-BREAK 标为 `PASSED (local decision)`。
3. 为 Task 0/D0 列出待冻结的 namespace、root export、operation ID、error category、extras 和 version policy；不得把该清单误称为 D0、D-TECH 或 release 证据。
4. 冻结旧 extra 到新能力型 extra 的迁移表，例如 visualization、report-pdf、report-xlsx、bayesian、data provider。
5. 明确实施不授权 merge/push/tag/publish/远端设置修改。
6. 用 readiness preflight 验证 caller-supplied clean `dev` worktree、full expected HEAD、plan SHA 和 recorded baseline ancestry；隔离当前 user-owned dirty worktree，不接管或修改它。
7. 本次 documentation-only commit 不改 README 或参考计划状态指针；如需更新，必须由后续明确范围的 status task 执行，且不是 D-ID/D-BREAK 或 Task 0 的前置证据。

**Acceptance:**

- 历史 0042 文档和 BLOCKED acceptance 没有被改写。
- ADR、readiness 和 plan 使用同一 `0042-R2`、目标版本和 breaking 语义，并将 D-ID/D-BREAK 明确为本地决定通过。
- Task 0 start identity 由可复现的 caller-root/HEAD/plan-SHA/baseline-ancestry preflight 验证；D0、D-TECH 和 release 不因此通过。
- dirty user-owned 文件不进入 R2 commit。

**Rollback:** 若用户撤回本地 breaking-policy 决定，回滚本任务新建的 R2 文档；不回滚或修改历史 0042。

**Suggested commit:** `docs: approve 0042-r2 breaking unified core`

### Task 0: 建立能力账本、测量工具与 fresh D0

**Owner:** Acceptance + Quality；Task 0B 由各 Domain Test Owner 领取各自 coverage-gap 文件

**Depends on:** Task -1，以及已记录的 `D-ID` / `D-BREAK` local decision。

**Files:**

- Create: `tests/parity/fixtures/capability-ledger-0042-r2.json`
- Create: `tests/parity/fixtures/legacy-surface-inventory-0042-r2.json`
- Create: `tests/parity/fixtures/module-disposition-0042-r2.json`
- Create: `tests/parity/fixtures/test-node-disposition-0042-r2.json`
- Create: `tests/parity/fixtures/repository-surface-disposition-0042-r2.json`
- Create: `tests/parity/fixtures/planned-api-0.5.0.json`
- Create: `tests/parity/fixtures/0042-r2-gate-manifest.json`
- Create: `tests/parity/fixtures/0042-r2-matrix-evidence.schema.json`
- Create: `tests/parity/fixtures/0042-r2-architecture-threshold-policy.json`
- Create: `tests/parity/goldens/0042-r2/`
- Create after clean capture: `docs/quality/0042-r2-capability-baseline.json`
- Create after clean capture: `docs/quality/0042-r2-architecture-baseline.json`
- Create after clean capture: `docs/quality/0042-r2-performance-baseline.json`
- Create after clean capture: `docs/quality/0042-r2-quality-baseline.json`
- Create after clean capture: `docs/quality/0042-r2-quality-baseline.md`
- Create: `scripts/capture_capability_baseline.py`
- Create: `scripts/check_0042_r2_repository_surface_disposition.py`
- Create: `scripts/check_feature_parity.py`
- Create: `scripts/check_architecture_convergence.py`
- Create: `scripts/run_0042_r2_acceptance.py`
- Create: `requirements-0042-r2-acceptance.txt`
- Create: `tests/parity/test_ledger.py`
- Create: `tests/quality/test_capture_capability_baseline.py`
- Create: `tests/quality/test_check_feature_parity.py`
- Create: `tests/quality/test_architecture_convergence.py`
- Create: `tests/quality/test_0042_r2_profiling_contract.py`
- Create: `tests/quality/test_run_0042_r2_acceptance.py`
- Create: `tests/quality/test_0042_r2_matrix_evidence.py`
- Create: `tests/benchmarks/test_0042_r2_workloads.py`
- Create as needed: `tests/coverage_gaps/0042_r2/**`
- Modify: `scripts/profile_workloads.py`
- Modify: `scripts/check_performance.py`
- Modify: `scripts/snapshot_public_api.py`
- Modify: `scripts/test_installed_wheel.py`
- Modify before tooling freeze: `scripts/check_release_consistency.py`
- Modify: `tests/packaging/test_release_consistency.py`
- Create: `tests/quality/test_repository_surface_disposition.py`

**Steps:**

1. 先写失败测试，要求 inventory 取所有公开定义、registries、manifests、docs、examples、benchmarks、extras 与 wheel contents 的并集。
2. 为每个 surface 登记 capability、场景、owner、target operation、source nodeid、wheel nodeid、oracle/golden 和 disposition。
3. 为每个 `fincore/**/*.py` 登记 keep/move/delete、目标模块、consumer count 和 capability IDs，0 unmapped。
4. 为 active workflows、packaging/release scripts、maintained docs/templates、examples、type stubs、compat generators/checkers 建立 repository-surface disposition；每项逐路径绑定 raw Git blob、kind/category tags、受控 owner、lifecycle、completion gate、target contract/capability 与 rule ID，不能从 shell token 启发式推断决策。historical/provenance candidate 必须双向映射到单独 allowlist 并记录原始 digest；只允许受限文本后缀的纯文本记录标为 `text_only`，HTML、rendered 或 binary artifact 必须以非文本 provenance 条目保存，不能借 allowlist 作为可执行兼容示例。该 scoped mapping 的成功仍标记 `not_for_d0`，只能在后续与其他完整输入共同封入 D0。
5. 冻结全部 257 个现有 Catalog binding（Empyrical class 100、Empyrical module 49、metrics 50、flat API 20、context 18、performance 9、Pyfolio module 11），另行盘点 Pyfolio 类的 69 个 methods（其中 67 个 non-private）、Alphalens 61 个 function specs + 7 个 workflows，以及所有未入 Catalog 的增强领域能力；这些是 legacy surfaces，不得直接等同为 257 个独立 capabilities，alias/quirk 必须逐项 disposition。
6. 每个数值 scenario 登记 expected authority、来源版本/digest、tolerance 和 `preserve/correction_required`；候选或当前输出不能成为唯一 oracle。
7. 把旧 compatibility tests 中真实 numerical/container/error/plot/report 断言迁成独立 golden 或 invariant；纯签名/MRO/alias 断言仅进入 disposition。
8. 实现可复现 physical/logical LOC、normalized AST duplication、import graph、cycle、optional-import leakage 和 implementation fingerprint 测量；该通用 architecture checker 只提供可度量架构事实，不能替代第 12 项 Catalog/DAG/snapshot budget，也不能自行宣称 legacy-zero。
9. 修复 public snapshot：识别 callable kind、signature/default/kw-only，禁止静默跳过空 surface，并支持 source/wheel 比较。
10. 将 release consistency 改为版本化 contract：D0 可验证当前包，0.5 candidate contract 明确不要求 Alphalens/Pyfolio/Empyrical 文件；contract expected 来自冻结 bundle，不由 candidate 版本字符串选择。
11. 扩展 profiler 到 metrics、rolling、transactions、factor、risk、report；每个 workload 校验输出 schema/digest。
12. 补齐 Catalog resolution + runtime invocation、DAG、snapshot 三项 fixed budget；comparator 在无 baseline、pending 或平台不匹配时 fail closed。
13. 将 installed-wheel profiles 从旧项目名改成能力名，并增加 visualization、report-xlsx 和逐 data-provider offline profile；先更新 profile contract tests，再改脚本。
14. 实现独立 acceptance runner：只接受 candidate root、唯一 wheel、gate 和仓库外 D0 bundle；expected、schema、tolerance、tool argv 均来自 runner 所在的冻结 tooling SHA，candidate 只能提供 actual。
15. 冻结 required gate manifest：`tests`（全部非在线 functional，含 slow/serial/offline integration）、`static`、`package`、`quality`、`parity`、`architecture`、`performance`、`report`、`installed`、`matrix-cell`、`matrix-aggregate`、`final`、`evidence-child`；`final` 缺任一技术 gate evidence 都 fail closed，`evidence-child` 单独验证 verdict 文档的 parent/allowlist。
16. 冻结 matrix cell schema，并在 D0 bundle 中记录精确、有序的 `python_support_window`（目标为 3.11–3.14，实际集合以 D0 依赖矩阵为准）：candidate commit/tree、同一 wheel SHA256、D0 tooling/bundle digest、OS/runner image、Python full version、dependency lane/profile、固定 argv digest、test/output digest、时间和 verdict。`matrix-aggregate` 只接受 Linux/macOS/Windows × 该 D0-frozen Python 支持窗口的全部 cell；bundle 缺少该字段、字段为空或 cell 不完整均 fail closed。
17. 先提交并冻结 `D0_TOOLING_SHA` 与 source-tracked architecture threshold policy；policy 只定义由 D0 原始测量派生的 candidate threshold rules，不能把 D0 自身当成已达到 final reduction 的候选。随后从该 SHA 的 detached worktree 执行工具测试，之后不允许 Tasks 1–9 修改 runner、checker、profiler、measurement schema、policy schema 或 threshold。
18. 若 fresh coverage 仍低于 60%，启动独立 Task 0B coverage-gap sub-tranche：Quality owner 分配 domain test owners，只新增真实 branch/error/boundary 测试，不改生产语义、不使用无断言执行；发现 defect 时先交回独立 `fix:` tranche，修复和测试都落到 clean named commit 后再重测。
19. Task 0B 达到 overall branch `>=60%`、critical modules `>=90%` 后，再从 clean exact-SHA worktree 执行 Task 0C D0 capture；证据先写到仓库外目录，再作为单独、可审查的 baseline commit materialize 到上述 `docs/quality/0042-r2-*` 文件。记录 `D0_TOOLING_SHA`、baseline source SHA/tree、acceptance lock digest 和外部 bundle digest。
20. D0 bundle 必须包含 architecture baseline source provisioning manifest：baseline source commit/tree、architecture threshold policy 的 path/Git blob/SHA256，以及可验证地 materialize 为 clean detached checkout 的 source archive/object provenance。后续 architecture 比较必须将该 checkout 作为 `--baseline-source-root`；不得以 candidate checkout、`D0_TOOLING_ROOT` 或空临时目录替代。

**先验证工具测试：**

```bash
/Users/yunjinqi/opt/anaconda3/bin/conda run -n base python -m pytest -o addopts='' -p no:cacheprovider -p no:rerunfailures tests/parity/test_ledger.py tests/quality/test_capture_capability_baseline.py tests/quality/test_check_feature_parity.py tests/quality/test_architecture_convergence.py tests/quality/test_0042_r2_profiling_contract.py tests/quality/test_run_0042_r2_acceptance.py tests/quality/test_0042_r2_matrix_evidence.py tests/quality/test_repository_surface_disposition.py tests/packaging/test_release_consistency.py tests/benchmarks/test_0042_r2_workloads.py -q --tb=short --maxfail=0
/Users/yunjinqi/opt/anaconda3/bin/conda run -n base python scripts/check_0042_r2_repository_surface_disposition.py --facts tests/parity/fixtures/repository-surface-facts-discovery-0042-r2.json --disposition tests/parity/fixtures/repository-surface-disposition-0042-r2.json
```

**再从 clean exact-SHA 采集：**

```bash
set -euo pipefail
FINCORE_0042R2_D0_DIR=$(mktemp -d /tmp/fincore-0042-r2-d0.XXXXXX)
FINCORE_0042R2_MPL_DIR=$(mktemp -d /tmp/fincore-0042-r2-mpl.XXXXXX)
PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg MPLCONFIGDIR="$FINCORE_0042R2_MPL_DIR" /Users/yunjinqi/opt/anaconda3/bin/conda run -n base python scripts/capture_capability_baseline.py --inventory tests/parity/fixtures/legacy-surface-inventory-0042-r2.json --module-disposition tests/parity/fixtures/module-disposition-0042-r2.json --test-disposition tests/parity/fixtures/test-node-disposition-0042-r2.json --ledger tests/parity/fixtures/capability-ledger-0042-r2.json --fixture-dir tests/parity/goldens/0042-r2 --output "$FINCORE_0042R2_D0_DIR/capability-baseline.json" --deny-network
/Users/yunjinqi/opt/anaconda3/bin/conda run -n base python scripts/check_architecture_convergence.py --package fincore --capture "$FINCORE_0042R2_D0_DIR/architecture-baseline.json" --seal-baseline --threshold-policy tests/parity/fixtures/0042-r2-architecture-threshold-policy.json
/Users/yunjinqi/opt/anaconda3/bin/conda run -n base python scripts/profile_workloads.py --sizes small medium large --kinds metrics rolling transactions factor risk report --warmups 2 --repeats 5 --require-output-digest --output "$FINCORE_0042R2_D0_DIR/performance-baseline.json"
/Users/yunjinqi/opt/anaconda3/bin/conda run -n base python scripts/collect_quality_baseline.py --json "$FINCORE_0042R2_D0_DIR/current-baseline.json" --markdown "$FINCORE_0042R2_D0_DIR/current-baseline.md"
/Users/yunjinqi/opt/anaconda3/bin/conda run -n base python scripts/check_quality_snapshot.py --snapshot "$FINCORE_0042R2_D0_DIR/current-baseline.json"
```

上述 committed fixture/golden 输入必须在 clean exact-SHA 内存在，并将每个输入及其 include/exclude manifest 的 digest 写入 D0 bundle；`FINCORE_0042R2_D0_DIR` 只承载本次 clean capture 的输出，不得把空临时目录当作账本、disposition 或 golden 的来源。tooling SHA、baseline source SHA、clean status、平台和依赖 provenance 仍须按本任务的 D0 要求一并冻结。

上述 `check_feature_parity.py`、`check_architecture_convergence.py` 和扩展后的 profiler CLI 只有在本任务实现并通过测试后才可执行；它们不是当前仓库已存在的能力。

正式 tranche/final gate 不从 candidate checkout 执行这些脚本，而使用 detached `D0_TOOLING_SHA`：

```bash
test -n "$FINCORE_0042R2_D0_TOOLING_ROOT"
test -n "$FINCORE_0042R2_D0_BUNDLE"
/Users/yunjinqi/opt/anaconda3/bin/conda run -n base python -I "$FINCORE_0042R2_D0_TOOLING_ROOT/scripts/run_0042_r2_acceptance.py" --candidate-root "$PWD" --expected-bundle "$FINCORE_0042R2_D0_BUNDLE" --gate tranche --families runtime
```

runner 必须先验证自己的 blob/commit、acceptance lock 和 D0 bundle digest，再启动隔离 candidate subprocess；不能从 candidate import checker、expected fixture 或阈值。

**Acceptance:**

- legacy surface、生产模块、旧 test nodeid 映射率均为 100%。
- active workflow/script/maintained-doc/template disposition 为 100%；historical/provenance allowlist 的原始 digest 已冻结。
- required capability 的适用 happy/boundary/error/optional/provider/documented scenario 覆盖率为 100%。
- 孤儿能力 0、未裁决差异 0、无 owner 项 0；所有 `correction_required` 均有独立 oracle 和具名修复 owner。
- D0 记录 commit、tree、clean 状态、Python/依赖/平台、脚本 SHA、architecture threshold policy blob/digest、include/exclude manifest 和证据 digest；architecture baseline source provisioning manifest 能重建一个 clean exact baseline-source checkout。
- `D0_TOOLING_SHA` 与 baseline source SHA 分离记录；tooling SHA 是后续正式 gate 的唯一 checker/profiler/runner 权威。
- branch coverage 至少 60%；若 clean base 尚未达到，不得用旧 snapshot 代替，先补测试再冻结 D0。
- 性能 baseline 非 pending，所有 workload 输出 digest 有效。

**Rollback:** 整笔回滚 tooling/baseline commit；不得只删失败能力、改 tolerance 或重写历史 D0。

**Suggested commits:**

- `test: add 0042-r2 capability and architecture measurement`
- `test: freeze clean 0042-r2 capability quality and performance baseline`

### Task 1: 建立 canonical runtime、Catalog 与结果模型

**Owner:** Runtime

**Depends on:** Task 0 / D-BASE。

**Files:**

- Create: `fincore/runtime/__init__.py`
- Create: `fincore/runtime/specs.py`
- Create: `fincore/runtime/catalog.py`
- Create: `fincore/runtime/builtins.py`
- Create: `fincore/runtime/data.py`
- Create: `fincore/runtime/validation.py`
- Create: `fincore/runtime/engine.py`
- Create: `fincore/runtime/session.py`
- Create: `fincore/runtime/results.py`
- Create: `fincore/runtime/artifacts.py`
- Create: `fincore/runtime/backends.py`
- Create: `fincore/runtime/types.py`
- Modify: `fincore/exceptions.py`
- Retarget then delete after consumer count zero: `fincore/validation.py`, `fincore/backends/`, `fincore/_types.py`
- Create: `tests/runtime/test_catalog.py`
- Create: `tests/runtime/test_data.py`
- Create: `tests/runtime/test_engine.py`
- Create: `tests/runtime/test_session.py`
- Create: `tests/runtime/test_results.py`
- Create: `tests/runtime/test_artifacts.py`
- Retarget temporarily: `fincore/api/**`, `fincore/core/**`, `fincore/contracts/**`, `fincore/results/**`

**Steps:**

1. 写失败测试，证明 `OperationSpec` 不含 profile/public binding/adapter/projection/string signature。
2. 实现不可变 OperationSpec 和预索引 Catalog：重复 `operation_id` 失败；同一 leaf capability 出现两个不同 implementation fingerprints 失败；多个 operation IDs 共享同一 fingerprint 仅在已批准参数化 mode 下允许。
3. 实现 composition-root contract：Task 1 只用 runtime 自身的测试 operation 证明 `runtime.builtins` 聚合机制；Tasks 2–6 逐域接入，D-DOMAIN 才形成完整 builtin Catalog。不得在 D-RUNTIME 提前宣称全域权威。
4. 聚合使用显式 `domain.operations()` 固定清单；不扫描 `sys.modules`，不生成 wrapper，不 monkeypatch module class。
5. 把 AnalysisSnapshot 迁为 immutable runtime input，保留 copy-on-ingest、digest、timezone 和 mutation isolation。
6. 实现 `plan/run/batch`；直接调用 canonical callable，并统一 Result、diagnostics、timing、provenance。
7. 把 AnalysisContext 中通用 state/cache 迁入 AnalysisSession，不复制 Empyrical/Pyfolio 方法面。
8. 合并 Result、ArtifactBundle 和 serializer，资源 close 幂等，ownership 显式。
9. 迁移 NumPy backend 和公共 type/schema；逐一 retarget consumers。
10. 旧 api/core/contracts/results 仅作为内部迁移桥，不创建用户可见 re-export，并在 Task 8 删除。

**Verify:**

```bash
/Users/yunjinqi/opt/anaconda3/bin/conda run -n base python -m pytest -o addopts='' -p no:cacheprovider tests/runtime tests/contracts/test_artifact_lifecycle.py tests/contracts/test_result_protocol.py -q --tb=short --maxfail=0
FINCORE_0042R2_RUNTIME_DIST=$(mktemp -d /tmp/fincore-0042-r2-runtime.XXXXXX)
/Users/yunjinqi/opt/anaconda3/bin/conda run -n base python -m build --wheel --outdir "$FINCORE_0042R2_RUNTIME_DIST"
/Users/yunjinqi/opt/anaconda3/bin/conda run -n base python scripts/check_feature_parity.py --baseline docs/quality/0042-r2-capability-baseline.json --ledger tests/parity/fixtures/capability-ledger-0042-r2.json --families runtime --dist "$FINCORE_0042R2_RUNTIME_DIST"
```

**Acceptance:**

- 测试 operation 恰好一个 callable；每个 runtime capability 恰好一个 implementation fingerprint；不声称其他 family 已迁入。
- 除 `runtime.builtins` 的测试 composition 清单外，runtime 不 import domain；任何 runtime 模块都不 import `_registry`、`_dispatch`、Empyrical、Pyfolio、Alphalens、report implementation 或 optional renderer。
- source/wheel 的 runtime normalized result digest 一致。
- snapshot、cache、session、result、artifact 和 error category 契约全绿。

**Rollback:** 回滚完整 runtime tranche；不得把新 0.5 operation 写回旧 registry。

**Suggested commit:** `refactor: introduce canonical fincore runtime`

### Task 2: 统一 metrics/performance，迁移 Empyrical 能力

**Owner:** Metrics + Performance

**Depends on:** Task 1 / D-RUNTIME。

**Files:**

- Modify: `fincore/metrics/**`
- Modify: `fincore/performance/**`
- Exclude / hand off to Task 5B: `fincore/metrics/perf_attrib.py`
- Move then delete after consumers migrate: `fincore/constants/periods.py`, `fincore/constants/interesting_periods.py`, `fincore/utils/math_utils.py`, `fincore/utils/data_utils.py`, `fincore/utils/date_utils.py`
- Create: `fincore/metrics/operations.py`
- Create: `fincore/performance/operations.py`
- Create: `fincore/metrics/frequencies.py`
- Create: `fincore/metrics/_numeric.py`
- Create: `fincore/metrics/_rolling.py`
- Create: `tests/parity/test_metrics.py`
- Create: `tests/parity/test_performance.py`
- Retarget: `tests/test_metrics/**`, `tests/test_empyrical/**`, relevant `tests/compat/empyrical/**`
- Retain as oracle until Task 8: `fincore/empyrical.py`, `fincore/_empyrical_legacy.py`

**Steps:**

1. 逐项把 Empyrical-only 组合公式迁入现有 metrics/performance kernel；alias 映射到同一 callable。`metrics/perf_attrib.py` 只冻结场景并交给 Task 5B，不由 Metrics owner 改写。
2. 把旧 tests 的 numerical、alignment、rolling、NaN、index、dtype、timezone、mutation 场景改为只调用新领域 API。
3. 移除 metrics 模块中的动态 surface installer；公开函数直接执行 kernel。
4. 把隐式 validation、alignment、annualization 和 output normalization 变成明确 boundary primitive。
5. 合并 math/data utils，私有 primitive 只保留一个实现；禁止新建 catch-all utils。
6. 将 period/annualization/financial date helpers 迁入 metrics frequency/date primitives；涉及通用 timezone normalization 时只调用 Task 1 runtime contract，不复制实现。完成 consumer handoff 后删除旧 math/data/date/period 模块。
7. 为每个 leaf capability 注册 OperationSpec，不建立 root/class/profile 多 binding。
8. 运行 parity 后生成 Empyrical deletion-readiness；旧模块此阶段只作 oracle，不能进入任何新 API 或发布文档。

**Verify:**

```bash
PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg /Users/yunjinqi/opt/anaconda3/bin/conda run -n base python -m pytest -o addopts='' -p no:cacheprovider tests/parity/test_metrics.py tests/parity/test_performance.py tests/test_metrics tests/test_empyrical tests/numerical tests/property -q --tb=short --maxfail=0
/Users/yunjinqi/opt/anaconda3/bin/conda run -n base python scripts/check_feature_parity.py --baseline docs/quality/0042-r2-capability-baseline.json --ledger tests/parity/fixtures/capability-ledger-0042-r2.json --families metrics performance
```

**Acceptance:** metrics/performance capability missing 0；数值、标签、时区、mutation 差异 0；新代码对 Empyrical/legacy registry 的 import 为 0。

**Rollback:** 回滚整个 domain tranche；禁止恢复 wrapper 作为发布方案。

**Suggested commits:**

- `refactor: make metric kernels canonical operations`
- `refactor: migrate empyrical analytical capabilities`

### Task 3: 建立 portfolio 域，迁移 Pyfolio 工作流

**Owner:** Portfolio

**Depends on:** Task 2。

**Files:**

- Create: `fincore/portfolio/__init__.py`
- Create: `fincore/portfolio/models.py`
- Create: `fincore/portfolio/positions.py`
- Create: `fincore/portfolio/transactions.py`
- Create: `fincore/portfolio/round_trips.py`
- Create: `fincore/portfolio/contributions.py`
- Create: `fincore/portfolio/capacity.py`
- Create: `fincore/portfolio/operations.py`
- Modify then hand off remainder to Task 6: `fincore/utils/common_utils.py`
- Create: `tests/parity/test_portfolio.py`
- Retarget: `tests/test_pyfolio/**`, relevant `tests/compat/pyfolio/**`
- Retain as oracle until Task 8: `fincore/pyfolio.py`, `fincore/_pyfolio_impl.py`

**Steps:**

1. 将 positions、transactions、round trips、capacity、perf stats 和 attribution input/contribution normalization 分成纯函数和显式模型；Brinson/factor/style/perf-attrib 算法交给 Task 5B。
2. 把 Pyfolio 11 个 workflow 拆成领域 workflow builder；不新建一个替代 Pyfolio 的 god class。
3. session/cache 使用 runtime，而不是继承 Empyrical 或持有 façade state。
4. 原类方法测试改为 capability scenario，保留金融输出和资源 observable，不保留 MRO/descriptor/signature。
5. 报告构建所需中间模型先定义为 portfolio result，渲染留给 Task 6。
6. 所有 portfolio operation 直接引用 canonical metrics/performance callable。
7. 从 `common_utils.py` 只迁走 portfolio-owned positions/transactions/table-input helpers；按 helper disposition 将剩余 table/export/asset/legend 函数原样 handoff 给 Task 6，不并行改同一文件。

**Verify:**

```bash
PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg /Users/yunjinqi/opt/anaconda3/bin/conda run -n base python -m pytest -o addopts='' -p no:cacheprovider tests/parity/test_portfolio.py tests/test_pyfolio -q --tb=short --maxfail=0
/Users/yunjinqi/opt/anaconda3/bin/conda run -n base python scripts/check_feature_parity.py --baseline docs/quality/0042-r2-capability-baseline.json --ledger tests/parity/fixtures/capability-ledger-0042-r2.json --families portfolio
```

**Acceptance:** 11 个 workflow 和所有 portfolio leaf capability 均有新 operation/scenario；portfolio 不继承或 import Empyrical/Pyfolio；重复算法 implementation 为 0。

**Rollback:** 回滚 portfolio tranche；旧 Pyfolio 仍只作为迁移 oracle，不增加新能力。

**Suggested commit:** `refactor: replace pyfolio facade with portfolio workflows`

### Task 4: 合并 Alphalens 与 factor_analysis

**Owner:** Factor Analysis

**Depends on:** Task 1；可与 Task 3 并行，但共享文件需由 owner 协调。

**Files:**

- Modify: `fincore/factor_analysis/**`
- Create: `fincore/factor_analysis/operations.py`
- Create or consolidate: `fincore/factor_analysis/workflows.py`
- Create: `tests/parity/test_factor_analysis.py`
- Retarget: `tests/test_factor_analysis/**`, `tests/compat/alphalens/**`
- Retain as oracle until Task 8: `fincore/alphalens/**`

**Steps:**

1. 建立 49 个同名能力的一对一实现映射；验证哪个版本是 canonical，不以文件新旧决定。
2. 将 Alphalens-only 能力迁入 prepare/model/analyze 分层；领域层只产生可渲染 model，不持有 renderer。
3. 将 7 个 tear workflow 的 compute 部分迁为显式 workflow/model，并把 renderer/tearsheet observable 清单交给 Task 6。
4. 统一 forward returns、calendar、quantile、weights、IC、turnover、events、PIT、多期、成本/容量与统计推断 schema。
5. 对 index、group、timezone、loss/max_loss、zero-aware、by-date/by-group 写数值/结构 golden；plotting resource golden 在 Task 6 由 Reporting owner 落地。
6. operation metadata 与函数共址，删除 factor workflow registry 的第二真源。

**Verify:**

```bash
PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg /Users/yunjinqi/opt/anaconda3/bin/conda run -n base python -m pytest -o addopts='' -p no:cacheprovider tests/parity/test_factor_analysis.py tests/compat/alphalens tests/test_factor_analysis -q --tb=short --maxfail=0
/Users/yunjinqi/opt/anaconda3/bin/conda run -n base python scripts/check_feature_parity.py --baseline docs/quality/0042-r2-capability-baseline.json --ledger tests/parity/fixtures/capability-ledger-0042-r2.json --families factor
```

**Acceptance:** factor required capability/scenario missing 0；同名能力只剩一个实现；新 factor code 对 `fincore.alphalens` import 为 0；7 个 workflow 只依赖 canonical factor model。

**Rollback:** 回滚 factor tranche；不得通过转发到 Alphalens 让新测试变绿。

**Suggested commit:** `refactor: unify factor analysis capabilities`

### Task 5: 收敛 risk、attribution、simulation、optimization、data 与 extensions

**Owner:** Domain Owners + Runtime

**Depends on:** 分为三个可独立提交的 sub-tranche：5A-core（simulation/optimization/data/extensions）依赖 Task 1，可与 Tasks 2/4 并行；5A-risk 等待 Task 2 的 descriptive-risk handoff；5B-attribution 依赖 Tasks 2 和 3，消费 canonical metrics 与 portfolio inputs。

**Files:**

- Modify: `fincore/risk/**`
- Modify: `fincore/attribution/**`
- Modify: `fincore/simulation/**`
- Modify: `fincore/optimization/**`
- Modify: `fincore/data/**`
- Create: `fincore/extensions/**`
- Create: `fincore/extensions/snapshot.py`
- Create: each domain `operations.py`
- Create: `tests/parity/test_risk.py`
- Create: `tests/parity/test_attribution.py`
- Create: `tests/parity/test_simulation.py`
- Create: `tests/parity/test_optimization.py`
- Create: `tests/parity/test_data_extensions.py`
- Create: `tests/extensions/test_snapshot.py`

**Steps:**

1. 5A-core 把 simulation/optimization/data/extensions 的 validation、rng、provider/cache、result 和 optional import 接到统一 primitive。
2. 对已有独立 oracle 的算法只做依赖翻转和重复 helper 合并，不在结构迭代中顺手改金融定义。
3. 任一发现的数值 bug 先新增独立 oracle、记录为 capability correction，再在单独 commit 修复。
4. 5A-core 将 plugin/hooks 迁入 `ExtensionSnapshot`，保留 registration/discovery/isolation 能力，不保留旧注册表 API 形状或进程级 mutable singleton；验证 namespace collision、builtin override rejection、snapshot digest、session pinning 和并发隔离。
5. provider 统一 offline fake-client contract；在线集成只在独立 integration lane 运行。
6. 5A-risk 接收 Task 2 的显式 handoff：`trading_value_at_risk`、`gpd_risk_estimates*` 和其他参数化/GPD/EVT 能力迁入 `risk`；`metrics.risk` 最终只保留描述性历史样本度量。
7. 5A-risk 的 forecast/calibration/backtest/walk-forward 产生 typed domain result；`risk/report.py` 不渲染，Task 6 负责 report/export。
8. 5B-attribution 将 `metrics/perf_attrib.py`、Pyfolio perf-attrib surface 与现有 attribution 实现合并；portfolio 只提供标准化 holdings/returns/exposure inputs，Brinson/factor/style/perf attribution 只在 `attribution` 有一个实现。
9. Task 5 不拥有共享 constants/common utils 清理；严格按 module disposition 接收自己领域的私有 helper，其他 support 文件由 Tasks 2/3/6 的唯一 owner 迁移。

**Verify:**

```bash
/Users/yunjinqi/opt/anaconda3/bin/conda run -n base python -m pytest -o addopts='' -p no:cacheprovider tests/parity/test_risk.py tests/parity/test_attribution.py tests/parity/test_simulation.py tests/parity/test_optimization.py tests/parity/test_data_extensions.py tests/numerical tests/property -q --tb=short --maxfail=0
/Users/yunjinqi/opt/anaconda3/bin/conda run -n base python scripts/check_feature_parity.py --baseline docs/quality/0042-r2-capability-baseline.json --ledger tests/parity/fixtures/capability-ledger-0042-r2.json --families risk attribution simulation optimization data extensions
```

**Acceptance:** 各 family missing 0；duplicate helper 0；domain kernel 不依赖 report/旧 registry；core import 不加载禁止的 optional roots。

**Rollback:** 按 domain commit 独立回滚；跨域 runtime schema 变更必须由所有消费者同时回滚。

**Suggested commits:** 每个 domain 一个 `refactor:` commit，数值修正使用独立 `fix:` commit。

### Task 6: 统一 report、renderer、result 与 artifact

**Owner:** Reporting；Runtime Owner 独占 `runtime/results.py`、`runtime/artifacts.py`、`runtime/builtins.py` 的最终集成

**Depends on:** Tasks 2–5。

**Files:**

- Reorganize: `fincore/report/**`
- Create: `fincore/report/portfolio/**`
- Create: `fincore/report/factor/**`
- Create: `fincore/report/renderers/**`
- Modify: `fincore/viz/**`
- Move then delete after consumers migrate: remaining `fincore/utils/common_utils.py`, `fincore/constants/color.py`, `fincore/constants/style.py`
- Modify by Runtime Owner only: `fincore/runtime/results.py`
- Modify by Runtime Owner only: `fincore/runtime/artifacts.py`
- Modify by Runtime Owner only: `fincore/runtime/builtins.py`
- Create: `tests/parity/test_report_models.py`
- Create: `tests/parity/test_report_renderers.py`
- Create: `tests/parity/test_artifact_lifecycle.py`
- Create: `tests/runtime/test_builtin_catalog_optional_isolation.py`
- Create: `tests/parity/goldens/0042-r2/reports/`

**Steps:**

1. 先把 `report/compute.py` 对 Empyrical/Pyfolio 的调用改为 canonical metrics/portfolio/factor operations。
2. 统一 portfolio/factor/risk 报告的 compute model；renderer 不重复计算。
3. 合并 ReportArtifacts、factor artifacts 和 results ArtifactBundle。
4. 对表格、章节、单位、series/legend、offline assets、PDF/XLSX 内容建立 normalized semantic golden。
5. 分离 Matplotlib、HTML/PDF/XLSX、Plotly/Bokeh renderer；optional import 保持 lazy。
6. 使用真实 Chromium/PDF lane 验证交互、资源和分页；Agg lane 验证数据与 ownership，不以像素完全相同作为唯一门。
7. 删除旧 tearsheet 内部依赖；旧入口仍留到 Task 8 原子切换。
8. 所有 domain owners 只提交各自 `operations()`；lane 汇合后由 Runtime Owner 单独更新 `runtime.builtins` 固定清单、运行完整 Catalog coverage/duplicate gate，再冻结 D-DOMAIN。并行 domain task 不得争写 composition root。
9. 接收 Task 3 的 `common_utils.py` remainder，迁移 table/export/asset/legend 与 color/style 到 report/viz 私有模块；consumer count 为零后删除旧 support 文件。
10. 在 source 和 core-only wheel 环境阻断 PDF/Plotly/Bokeh/provider 等可选包后构建完整 builtin Catalog，断言禁止模块不进入 `sys.modules`；只有调用对应 operation 才允许返回 `optional_dependency_missing`。

**Verify:**

```bash
PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg /Users/yunjinqi/opt/anaconda3/bin/conda run -n base python -m pytest -o addopts='' -p no:cacheprovider tests/parity/test_report_models.py tests/parity/test_report_renderers.py tests/parity/test_artifact_lifecycle.py tests/runtime/test_builtin_catalog_optional_isolation.py tests/test_report -q --tb=short --maxfail=0
/Users/yunjinqi/opt/anaconda3/bin/conda run -n base python scripts/check_feature_parity.py --baseline docs/quality/0042-r2-capability-baseline.json --ledger tests/parity/fixtures/capability-ledger-0042-r2.json --families report viz
```

**Acceptance:** report/viz scenario missing 0；renderer 重新计算金融指标次数 0；artifact double-close/leak 0；新 report 对旧 façade/tearsheets import 为 0；所有领域 `operations()` 已由 `runtime.builtins` 完整聚合，Catalog coverage 100%、重复 operation/fingerprint 0；core-only source/wheel Catalog 构建成功且 optional import leakage 0；D-DOMAIN 通过后旧 registries 冻结为只读 oracle。

**Rollback:** 回滚完整 reporting tranche；不得保留两套 ArtifactBundle。

**Suggested commit:** `refactor: unify report models renderers and artifacts`

### Task 7: 优化预登记热点并偿还临时代码量

**Owner:** Performance + Domain Owners

**Depends on:** Tasks 2–6；尚未执行旧表面删除。

**Files:**

- Modify based on profiler: canonical runtime/domain modules only
- Create: `docs/quality/0042-r2-performance-report.md`

**Steps:**

1. 只优化 D0 预登记热点，不在看到结果后更换 workload。
2. 优先去除 Catalog 每次重建、重复 alignment/snapshot、重复 rolling moments、factor 重复 groupby、report 重复 compute。
3. 缓存键必须包含全部语义输入、版本和 config；写 mutation/session/concurrency 测试。
4. 使用向量化或共享中间量时，先证明输出 digest 和数值 tolerance 不变。
5. 每次性能 commit 同时记录 LOC、复杂度、wall/p95/RSS 和 benchmark variance。
6. 清理临时 adapter/helper，使每个 domain cleanup commit 净减 production LOC。
7. Task 0 冻结的 profiler、checker、workload、schema 和 threshold 不得修改；若发现工具缺陷，停止 Task 7，走 D0 amendment 并重采 baseline，不能在 candidate 中修 checker 后继续比较。
8. `0042-r2-performance-report.md` 只 materialize 冻结 runner 产生的证据摘要，记录 runner/tooling SHA、baseline/candidate digest 和原始外部证据位置。

**Verify:**

```bash
FINCORE_0042R2_PERF_DIR=$(mktemp -d /tmp/fincore-0042-r2-perf.XXXXXX)
test -n "$FINCORE_0042R2_D0_TOOLING_ROOT"
test -n "$FINCORE_0042R2_D0_BUNDLE"
/Users/yunjinqi/opt/anaconda3/bin/conda run -n base python -I "$FINCORE_0042R2_D0_TOOLING_ROOT/scripts/run_0042_r2_acceptance.py" --candidate-root "$PWD" --expected-bundle "$FINCORE_0042R2_D0_BUNDLE" --gate performance --families metrics rolling transactions factor risk report --warmups 2 --repeats 5 --output-dir "$FINCORE_0042R2_PERF_DIR"
```

**Acceptance:** 所有 workload 不越回退门；Catalog resolution + runtime invocation、DAG、snapshot 绝对预算通过；预登记热点达到第 3.3 节真实提升门；输出 digest 无变化。

**Rollback:** 逐个性能 commit 回滚；不放宽 tolerance、删 workload 或缩小数据集。

**Suggested commits:** 每个独立热点一个 `perf:` commit，随后一个 `refactor: remove duplicated execution paths`。

### Task 8: 原子完成 breaking cutover 与 package 重塑

**Owner:** One Cutover Owner（Architecture）；Packaging 与 Domain Owners 只签署 deletion-readiness 和 review，不并发写入

**Depends on:** Tasks 1–7 全部 parity/deletion-readiness 通过。

**Files:**

- Modify: `fincore/__init__.py`
- Modify: `pyproject.toml`
- Modify: `CHANGELOG.md`
- Modify: `.github/workflows/ci.yml`
- Modify: `.github/workflows/publish.yml`
- Modify: `.github/pull_request_template.md`, `.github/ISSUE_TEMPLATE/**`
- Modify: `tests/quality/test_workflow_integrity.py`
- Create: `tests/quality/test_0042_r2_matrix_workflow.py`
- Modify maintained user docs only: `README.md`, `docs/API_STABILITY.md`, `docs/MIGRATION.md`, `docs/architecture/public-api-map.md`, `mkdocs_docs/**`, `examples/**`，以及 repository-surface disposition 中所有 active maintained rows
- Create: `docs/migration/0.5-breaking-migration.md`
- Preserve as historical/provenance allowlist: prior `docs/plans/**`, prior `docs/quality/**`, `docs/迭代计划/**`, license/notice records；仅 Task -1 已批准的 status/index pointer 例外
- Create: `tests/contracts/fixtures/public-api-0.5.0.json`
- Create: `tests/contracts/test_removed_legacy_surfaces.py`
- Delete: `fincore/empyrical.py`
- Delete: `fincore/_empyrical_legacy.py`
- Delete: `fincore/pyfolio.py`
- Delete: `fincore/_pyfolio_impl.py`
- Delete: `fincore/alphalens/`
- Delete: `fincore/_registry.py`
- Delete: `fincore/_dispatch.py`
- Delete: `fincore/_compat/`
- Delete: `fincore/capabilities.py`
- Delete after consumer count zero: `fincore/api/`, `fincore/core/`, `fincore/contracts/`, `fincore/results/`, `fincore/tearsheets/`, old catch-all utils/constants
- Delete after consumer count zero: `fincore/validation.py`, `fincore/_types.py`, `fincore/backends/`, `fincore/plugin/`, `fincore/hooks/`
- Delete after repository-surface disposition is complete: `scripts/check_alphalens_upstream_test_migration.py`, `scripts/generate_compat_manifest.py`，以及仅服务旧 compatibility surfaces 的 generator/checker
- Retire after disposition complete: old compatibility tests/manifests/snapshots

**Steps:**

1. 冻结 cutover SHA、capability parity report、consumer graph 和 deletion allowlist。
2. 根包只保留 version、errors 和 domain namespaces；移除 lazy flat API。
3. 更新版本与 classifiers；支持窗口、minimum/latest constraints 和 extras 必须与真实 wheel matrix 一致。
4. 将旧 extras 改为能力型 extras；旧名字不提供 alias。
5. 在一个 atomic cutover commit 中删除全部旧 façade、registries、dispatch、profiles、compat adapters 和已迁移目录。
6. 迁移指南提供“旧能力/路径 → 新领域 API”静态映射，但不得包含可执行 shim。
7. 生成 planned API 的实际 snapshot 并比较；任何新增/缺失都需回到 D-BREAK 审核。
8. source 与 wheel 运行 old import negative、legacy reference gate、METADATA/contents inspection。
9. 只有 test-node disposition 100% 且新 source/wheel nodeid 已验证，才删除旧 compat assets。
10. 更新 release consistency、CHANGELOG、active workflows、issue/PR templates 和 maintained docs；CI/publish 不再运行 compat-alphalens、alphalens/alphalens-pyfolio profiles 或要求旧 runtime modules。历史 ADR/acceptance/provenance 文件保持 digest 不变。
11. 配置不依赖 Ruleset 的技术 matrix：一个 build job 产出唯一 wheel，Linux/macOS/Windows × D0-frozen `python_support_window`（目标 3.11–3.14）cells 下载并核对同一 SHA256，以冻结 runner 的 `matrix-cell` gate 输出 schema 化证据；远端不可用时允许使用预登记 self-hosted matrix，但 schema 和 cell 集相同，缺 cell 时 D-TECH 保持 BLOCKED。
12. cutover 后重跑完整非在线测试；红色时回滚整个 commit，不部分恢复 façade。

**Verify:**

```bash
PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg /Users/yunjinqi/opt/anaconda3/bin/conda run -n base python -m pytest -o addopts='' -p no:cacheprovider tests/contracts/test_removed_legacy_surfaces.py tests/parity tests/numerical tests/property -q --tb=short --maxfail=0
/Users/yunjinqi/opt/anaconda3/bin/conda run -n base python scripts/snapshot_public_api.py --check tests/contracts/fixtures/public-api-0.5.0.json
test -n "$FINCORE_0042R2_D0_TOOLING_ROOT"
test -n "$FINCORE_0042R2_D0_BASELINE_SOURCE_ROOT"
test -n "$FINCORE_0042R2_D0_BUNDLE"
test -z "$(git -C "$FINCORE_0042R2_D0_BASELINE_SOURCE_ROOT" status --porcelain=v1 --untracked-files=all)"
/Users/yunjinqi/opt/anaconda3/bin/conda run -n base python -I "$FINCORE_0042R2_D0_TOOLING_ROOT/scripts/check_architecture_convergence.py" --source-root "$PWD" --package fincore --baseline "$FINCORE_0042R2_D0_BUNDLE/architecture-baseline.json" --baseline-source-root "$FINCORE_0042R2_D0_BASELINE_SOURCE_ROOT" --require-no-cycles
FINCORE_0042R2_CUTOVER_DIST=$(mktemp -d /tmp/fincore-0042-r2-cutover.XXXXXX)
/Users/yunjinqi/opt/anaconda3/bin/conda run -n base python -m build --outdir "$FINCORE_0042R2_CUTOVER_DIST"
/Users/yunjinqi/opt/anaconda3/bin/conda run -n base python -m twine check "$FINCORE_0042R2_CUTOVER_DIST"/*
/Users/yunjinqi/opt/anaconda3/bin/conda run -n base python scripts/test_installed_wheel.py --dist "$FINCORE_0042R2_CUTOVER_DIST" --profiles core factor-analysis visualization report-pdf report-xlsx bayesian all --data-providers all
/Users/yunjinqi/opt/anaconda3/bin/conda run -n base python -I "$FINCORE_0042R2_D0_TOOLING_ROOT/scripts/run_0042_r2_acceptance.py" --candidate-root "$PWD" --candidate-dist "$FINCORE_0042R2_CUTOVER_DIST" --expected-bundle "$FINCORE_0042R2_D0_BUNDLE" --gate cutover --require-source-wheel-equal --require-legacy-zero --data-providers all
```

通用 architecture checker 不接收 `--require-legacy-zero`：该 flag 只由冻结 acceptance runner 的专用 legacy-removal contract 解释。runner 还必须先按 D0 bundle 的 provisioning manifest materialize 并验证 `FINCORE_0042R2_D0_BASELINE_SOURCE_ROOT` 的 commit/tree/policy blob；shell 的 clean 检查不是该 provenance 验证的替代。

**Acceptance:** 旧 source 文件、wheel 文件、imports、root aliases、profiles、extras、bindings、active workflow/script 和 maintained docs executable refs 均为 0；historical/provenance allowlist digest 不变；planned/actual API 一致；同一 wheel 的全部能力 profile 与 data-provider offline profile 通过。

**Rollback:** 原子回滚 cutover commit，回到“新内核完整、旧代码仅作 oracle”的不可发布状态；不得选择性恢复旧壳。

**Suggested commit:** `refactor!: complete fincore 0.5 breaking cutover and migration map`

迁移文档、package metadata、新 snapshot 与旧表面删除必须在同一个 cutover commit；不得把可执行代码和迁移说明拆成可单独落地的中间状态。

### Task 9: exact-SHA 全量技术验收

**Owner:** Independent Acceptance

**Depends on:** Task 8 / D-CUTOVER。

**Files:**

- Create only in evidence-only child after verdict: `docs/quality/0042-r2-acceptance.md`
- Create only in evidence-only child after verdict: `docs/quality/0042-r2-evidence-digests.json`
- Create outside source then materialize: acceptance manifest, wheel/sdist digests, capability/quality/performance/architecture reports
- Do not modify production code/tests/checkers: failed gate 返回对应 implementation owner；Acceptance 只记录 BLOCKED 证据

**Steps:**

1. 先冻结互斥 `BUILD_AUTHORITY=ci|self_hosted`。CI 模式由 Task 8 workflow 的唯一 build job 从 clean exact-SHA 构建；self-hosted 模式由预登记 acceptance builder 构建。两种模式都只构建一次 sdist/wheel，并记录 commit、tree、clean 状态、build frontend/version、constraints digest 和 artifact SHA256。
2. runner 从 D0 bundle 的 provisioning manifest materialize clean detached baseline-source checkout，验证 baseline commit/tree、architecture threshold policy Git blob/SHA256 和重建 measurement；该 checkout 仅通过 `--baseline-source-root` 提供给 D0-tooling architecture checker，绝不由 candidate 或 tooling checkout 代替。
3. 所有 Python/OS/minimum/latest/profile 和最终 acceptance 都下载/接收 build authority 产生的同一个 wheel 字节，不在任何 consumer job 内重建。
4. 对 source 与 wheel 执行完整 capability ledger；normalized output/golden 完全一致。`package` gate 还必须只读解包唯一 sdist，验证内容、metadata、maintained docs、license/provenance 和 legacy-zero 与 tested source contract 等价；不得从 sdist 重建第二个候选 wheel。
5. 执行 full non-online、numerical/property、coverage、Ruff、format、mypy、MkDocs、dependency matrix、architecture、LOC、duplicate、performance。
6. 执行真实 HTML/PDF/XLSX、Chromium、Plotly/Bokeh 和 artifact lifecycle。
7. 执行 core、factor-analysis、visualization、report-pdf、report-xlsx、bayesian、每个 data provider offline profile 与 all。
8. 扫描最终 wheel 的 LICENSE/NOTICE/provenance；删除旧代码不等于自动解除归属义务。
9. acceptance 文档明确区分 D-TECH 与 D-RELEASE；本地全绿不自动授权 merge/tag/publish。
10. matrix evidence 可来自现有 CI artifact 或预登记 self-hosted runners，但两者都必须执行冻结 `matrix-cell` contract；aggregate 只接受完整 3 OS × D0-frozen `python_support_window` cell 集和同一 wheel digest，不依赖 Ruleset/required-check 配置。
11. 在运行任何 gate 前冻结 `TECHNICAL_CANDIDATE_SHA`；所有 evidence 和构件绑定该 parent，不允许 acceptance 过程中修改 candidate tree。
12. final PASS 后才创建 evidence-only child commit，allowlist 仅为 `docs/quality/0042-r2-acceptance.md` 与外部证据 digest 索引。该 commit 必须记录 `tested_parent_sha=TECHNICAL_CANDIDATE_SHA`；D-TECH verdict 和 wheel 仍属于 tested parent，evidence child 不得被描述为重新测试过的 candidate，也不得重建 wheel。

**核心命令：**

build authority 只执行一次以下等价动作；CI mode 将其放在唯一 build job，self-hosted mode 在独立 builder 执行：

```bash
set -euo pipefail
test -z "$(git status --porcelain)"
FINCORE_0042R2_BUILD_OUT=$(mktemp -d /tmp/fincore-0042-r2-build.XXXXXX)
/Users/yunjinqi/opt/anaconda3/bin/conda run -n base python -m build --outdir "$FINCORE_0042R2_BUILD_OUT"
/Users/yunjinqi/opt/anaconda3/bin/conda run -n base python -m twine check "$FINCORE_0042R2_BUILD_OUT"/*
(cd "$FINCORE_0042R2_BUILD_OUT" && shasum -a 256 ./* > build-artifacts.sha256)
```

build authority 随后把整个 dist 目录（含 `build-artifacts.sha256`）作为一个不可变 artifact 交给 matrix cells 和 Independent Acceptance。self-hosted 同一执行域直接绑定 build 目录；CI consumer 必须先把唯一 artifact 下载到显式绝对目录。两种模式都先校验 manifest，再进入 consumer gate：

```bash
set -euo pipefail
test -n "$FINCORE_0042R2_BUILD_AUTHORITY"
case "$FINCORE_0042R2_BUILD_AUTHORITY" in
  self_hosted)
    test -n "$FINCORE_0042R2_BUILD_OUT"
    export FINCORE_0042R2_FINAL_DIST="$FINCORE_0042R2_BUILD_OUT"
    ;;
  ci)
    test -n "$FINCORE_0042R2_DOWNLOADED_DIST"
    test -d "$FINCORE_0042R2_DOWNLOADED_DIST"
    export FINCORE_0042R2_FINAL_DIST="$FINCORE_0042R2_DOWNLOADED_DIST"
    ;;
  *)
    exit 2
    ;;
esac
test -f "$FINCORE_0042R2_FINAL_DIST/build-artifacts.sha256"
(cd "$FINCORE_0042R2_FINAL_DIST" && shasum -a 256 -c build-artifacts.sha256)
```

CI workflow 必须把 artifact ID、build job/run identity、candidate SHA 和 manifest digest 写入 cell evidence；仅有同名目录或 artifact 名不构成 handoff。下面的 consumer 命令不得再次运行 build：

```bash
set -euo pipefail
test -n "$FINCORE_0042R2_D0_TOOLING_ROOT"
test -n "$FINCORE_0042R2_D0_BUNDLE"
test -n "$FINCORE_0042R2_MATRIX_EVIDENCE"
test -n "$FINCORE_0042R2_FINAL_DIST"
test -d "$FINCORE_0042R2_FINAL_DIST"
test -z "$(git status --porcelain)"
FINCORE_0042R2_CANDIDATE_ROOT=$(git rev-parse --show-toplevel)
FINCORE_0042R2_CANDIDATE_HEAD=$(git rev-parse HEAD)
PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg /Users/yunjinqi/opt/anaconda3/bin/conda run -n base python -m pytest -o addopts='' -p no:cacheprovider tests -q --tb=short --maxfail=0 -m "not integration_online" --ignore=tests/benchmarks
/Users/yunjinqi/opt/anaconda3/bin/conda run -n base python -m pytest -o addopts='' -p no:cacheprovider tests/numerical tests/property -q --tb=short -n 0 --maxfail=0
/Users/yunjinqi/opt/anaconda3/bin/conda run -n base python -m ruff check fincore tests scripts
/Users/yunjinqi/opt/anaconda3/bin/conda run -n base python -m ruff format --check fincore tests scripts
/Users/yunjinqi/opt/anaconda3/bin/conda run -n base python -m mypy fincore
/Users/yunjinqi/opt/anaconda3/bin/conda run -n base python -m mkdocs build --strict
# `build-artifacts.sha256` is a handoff manifest, not a distribution; check only the immutable wheel and sdist.
/Users/yunjinqi/opt/anaconda3/bin/conda run -n base python -m twine check "$FINCORE_0042R2_FINAL_DIST"/*.whl "$FINCORE_0042R2_FINAL_DIST"/*.tar.gz
FINCORE_0042R2_WHEEL_COUNT=$(find "$FINCORE_0042R2_FINAL_DIST" -maxdepth 1 -type f -name '*.whl' -print | wc -l | tr -d ' ')
test "$FINCORE_0042R2_WHEEL_COUNT" -eq 1
FINCORE_0042R2_WHEEL=$(find "$FINCORE_0042R2_FINAL_DIST" -maxdepth 1 -type f -name '*.whl' -print)
test -f "$FINCORE_0042R2_WHEEL"
FINCORE_0042R2_FINAL_EVIDENCE=$(mktemp -d /tmp/fincore-0042-r2-acceptance.XXXXXX)
/Users/yunjinqi/opt/anaconda3/bin/conda run -n base python -I "$FINCORE_0042R2_D0_TOOLING_ROOT/scripts/run_0042_r2_acceptance.py" --candidate-root "$FINCORE_0042R2_CANDIDATE_ROOT" --candidate-head "$FINCORE_0042R2_CANDIDATE_HEAD" --candidate-wheel "$FINCORE_0042R2_WHEEL" --expected-bundle "$FINCORE_0042R2_D0_BUNDLE" --gate tests --include-slow --include-serial --include-offline-integration --benchmarks-covered-by performance --output-dir "$FINCORE_0042R2_FINAL_EVIDENCE/tests"
/Users/yunjinqi/opt/anaconda3/bin/conda run -n base python -I "$FINCORE_0042R2_D0_TOOLING_ROOT/scripts/run_0042_r2_acceptance.py" --candidate-root "$FINCORE_0042R2_CANDIDATE_ROOT" --candidate-head "$FINCORE_0042R2_CANDIDATE_HEAD" --candidate-wheel "$FINCORE_0042R2_WHEEL" --expected-bundle "$FINCORE_0042R2_D0_BUNDLE" --gate static --output-dir "$FINCORE_0042R2_FINAL_EVIDENCE/static"
/Users/yunjinqi/opt/anaconda3/bin/conda run -n base python -I "$FINCORE_0042R2_D0_TOOLING_ROOT/scripts/run_0042_r2_acceptance.py" --candidate-root "$FINCORE_0042R2_CANDIDATE_ROOT" --candidate-head "$FINCORE_0042R2_CANDIDATE_HEAD" --candidate-wheel "$FINCORE_0042R2_WHEEL" --candidate-dist "$FINCORE_0042R2_FINAL_DIST" --expected-bundle "$FINCORE_0042R2_D0_BUNDLE" --gate package --require-one-sdist --require-sdist-source-equivalence --require-legacy-zero --output-dir "$FINCORE_0042R2_FINAL_EVIDENCE/package"
/Users/yunjinqi/opt/anaconda3/bin/conda run -n base python -I "$FINCORE_0042R2_D0_TOOLING_ROOT/scripts/run_0042_r2_acceptance.py" --candidate-root "$FINCORE_0042R2_CANDIDATE_ROOT" --candidate-head "$FINCORE_0042R2_CANDIDATE_HEAD" --candidate-wheel "$FINCORE_0042R2_WHEEL" --expected-bundle "$FINCORE_0042R2_D0_BUNDLE" --gate quality --require-fresh-coverage --require-changed-lines 0.95 --require-critical-branches 0.90 --output-dir "$FINCORE_0042R2_FINAL_EVIDENCE/quality"
/Users/yunjinqi/opt/anaconda3/bin/conda run -n base python -I "$FINCORE_0042R2_D0_TOOLING_ROOT/scripts/run_0042_r2_acceptance.py" --candidate-root "$FINCORE_0042R2_CANDIDATE_ROOT" --candidate-head "$FINCORE_0042R2_CANDIDATE_HEAD" --candidate-wheel "$FINCORE_0042R2_WHEEL" --expected-bundle "$FINCORE_0042R2_D0_BUNDLE" --gate parity --families all --require-source-wheel-equal --output-dir "$FINCORE_0042R2_FINAL_EVIDENCE/parity"
/Users/yunjinqi/opt/anaconda3/bin/conda run -n base python -I "$FINCORE_0042R2_D0_TOOLING_ROOT/scripts/run_0042_r2_acceptance.py" --candidate-root "$FINCORE_0042R2_CANDIDATE_ROOT" --candidate-head "$FINCORE_0042R2_CANDIDATE_HEAD" --candidate-wheel "$FINCORE_0042R2_WHEEL" --expected-bundle "$FINCORE_0042R2_D0_BUNDLE" --gate architecture --require-loc-reduction 0.12 --require-duplicate-reduction 0.60 --require-legacy-zero --require-no-cycles --output-dir "$FINCORE_0042R2_FINAL_EVIDENCE/architecture"
/Users/yunjinqi/opt/anaconda3/bin/conda run -n base python -I "$FINCORE_0042R2_D0_TOOLING_ROOT/scripts/run_0042_r2_acceptance.py" --candidate-root "$FINCORE_0042R2_CANDIDATE_ROOT" --candidate-head "$FINCORE_0042R2_CANDIDATE_HEAD" --candidate-wheel "$FINCORE_0042R2_WHEEL" --expected-bundle "$FINCORE_0042R2_D0_BUNDLE" --gate performance --families metrics rolling transactions factor risk report --warmups 2 --repeats 5 --output-dir "$FINCORE_0042R2_FINAL_EVIDENCE/performance"
/Users/yunjinqi/opt/anaconda3/bin/conda run -n base python -I "$FINCORE_0042R2_D0_TOOLING_ROOT/scripts/run_0042_r2_acceptance.py" --candidate-root "$FINCORE_0042R2_CANDIDATE_ROOT" --candidate-head "$FINCORE_0042R2_CANDIDATE_HEAD" --candidate-wheel "$FINCORE_0042R2_WHEEL" --expected-bundle "$FINCORE_0042R2_D0_BUNDLE" --gate report --real-browser chromium --real-html --real-pdf --real-xlsx --interactive-backends plotly bokeh --output-dir "$FINCORE_0042R2_FINAL_EVIDENCE/report"
/Users/yunjinqi/opt/anaconda3/bin/conda run -n base python -I "$FINCORE_0042R2_D0_TOOLING_ROOT/scripts/run_0042_r2_acceptance.py" --candidate-root "$FINCORE_0042R2_CANDIDATE_ROOT" --candidate-head "$FINCORE_0042R2_CANDIDATE_HEAD" --candidate-wheel "$FINCORE_0042R2_WHEEL" --expected-bundle "$FINCORE_0042R2_D0_BUNDLE" --gate installed --profiles core factor-analysis visualization report-pdf report-xlsx bayesian all --data-providers all --dependency-lanes minimum latest --output-dir "$FINCORE_0042R2_FINAL_EVIDENCE/installed"
/Users/yunjinqi/opt/anaconda3/bin/conda run -n base python -I "$FINCORE_0042R2_D0_TOOLING_ROOT/scripts/run_0042_r2_acceptance.py" --candidate-root "$FINCORE_0042R2_CANDIDATE_ROOT" --candidate-head "$FINCORE_0042R2_CANDIDATE_HEAD" --candidate-wheel "$FINCORE_0042R2_WHEEL" --expected-bundle "$FINCORE_0042R2_D0_BUNDLE" --gate matrix-aggregate --matrix-evidence-dir "$FINCORE_0042R2_MATRIX_EVIDENCE" --require-os linux macos windows --require-support-window-from-bundle --output-dir "$FINCORE_0042R2_FINAL_EVIDENCE/matrix"
/Users/yunjinqi/opt/anaconda3/bin/conda run -n base python -I "$FINCORE_0042R2_D0_TOOLING_ROOT/scripts/run_0042_r2_acceptance.py" --candidate-root "$FINCORE_0042R2_CANDIDATE_ROOT" --candidate-head "$FINCORE_0042R2_CANDIDATE_HEAD" --candidate-wheel "$FINCORE_0042R2_WHEEL" --expected-bundle "$FINCORE_0042R2_D0_BUNDLE" --gate final --evidence-dir "$FINCORE_0042R2_FINAL_EVIDENCE" --output-dir "$FINCORE_0042R2_FINAL_EVIDENCE/final"
```

第一条 pytest 命令因 `-o addopts=''` 不生成 coverage，只是快速诊断；它包含 slow、serial 和 integration_offline，只排除 integration_online 与由 performance gate 完整接管的 benchmark nodeids。fresh branch/changed-line/critical-module coverage 由冻结 runner 的 `quality` gate 重新采集；Ruff/format/mypy/MkDocs/twine 的直接命令也只是预检，正式证据分别来自 `static`/`package` gate。不同 OS/Python cell 必须下载并校验同一个 `FINCORE_0042R2_WHEEL` SHA256，再以同一 runner contract 运行 `matrix-cell`；任一 cell 重建 wheel 即失败。

在 final PASS 后，由 Acceptance owner 只提交两份 evidence 文档，再验证 child 关系：

```bash
FINCORE_0042R2_EVIDENCE_HEAD=$(git rev-parse HEAD)
/Users/yunjinqi/opt/anaconda3/bin/conda run -n base python -I "$FINCORE_0042R2_D0_TOOLING_ROOT/scripts/run_0042_r2_acceptance.py" --gate evidence-child --tested-parent "$FINCORE_0042R2_CANDIDATE_HEAD" --evidence-head "$FINCORE_0042R2_EVIDENCE_HEAD" --allow-path docs/quality/0042-r2-acceptance.md --allow-path docs/quality/0042-r2-evidence-digests.json --evidence-dir "$FINCORE_0042R2_FINAL_EVIDENCE/final"
```

**Acceptance:**

- capability/scenario missing 0，source/wheel semantic mismatch 0。
- old surface negative gate 全绿。
- physical/logical LOC 至少减少 12%，duplicate/delegate bodies 至少减少 60%。
- illegal import/cycle/optional leakage 为 0。
- branch coverage、changed-line coverage 和 critical-module coverage 达门。
- 全 workload 不越回退门，且达到真实性能提升门。
- 同一 wheel 的 profiles、minimum/latest、真实 report/browser 和 data offline lanes 全绿。
- 唯一 sdist 与 tested source 的内容/metadata/license/provenance/legacy-zero contract 等价，且未生成第二个候选 wheel。
- acceptance manifest 绑定 exact SHA 与 artifact digests，证据目录不可由 candidate 运行时覆盖。
- matrix aggregate 覆盖 D0-frozen `python_support_window` 的全部 OS/Python cells；缺 cell、wheel digest 不同或 schema/argv 漂移均为 BLOCKED。
- evidence-only child 的 parent 恰好是 tested candidate，diff 只含 acceptance 文档/证据索引；D-TECH 明确归属于 parent SHA。

**Rollback:** D-TECH 失败时保持 BLOCKED，回到对应 owner 的最后一个绿色 tranche；不修改 baseline、删失败测试或降低门槛。

**Suggested evidence-only child commit:** `docs: record 0042-r2 technical acceptance evidence for <tested-parent-sha>`

## 6. Ownership 与并行边界

| Owner | 独占写入范围 | 可依赖但不可随意改写 |
| --- | --- | --- |
| Acceptance | parity tooling、goldens、quality/acceptance docs | production domain code |
| Runtime | `fincore/runtime/**`、error algebra | 领域公式、report renderer |
| Metrics/Performance | `metrics/**`、`performance/**` | portfolio/factor/report |
| Portfolio | `portfolio/**` | metrics formulas、report renderer |
| Factor | `factor_analysis/**` | portfolio、risk |
| Domain owners | risk/attribution/simulation/optimization/data/extensions | runtime contracts |
| Reporting | report/viz/renderers | 金融 kernel |
| Packaging | root exports、pyproject、wheel profile、migration docs | capability oracle |

并行规则：

- Task 4 与 Task 5A-core 可在 Task 1 后与 Task 2 并行；Task 3 与 Task 5A-risk 必须等待 Task 2，Task 5B-attribution 必须等待 Tasks 2/3。任何 lane 修改 runtime schema 前先由 Runtime owner 审核。
- 每个 worker 必须知道其他人也在同一代码库工作，不得回滚或覆盖他人改动。
- 共享文件使用 owner-mediated handoff；禁止两个 tranche 同时修改 `fincore/__init__.py`、`pyproject.toml` 或 runtime schema。
- Task 6 等待所有上游领域稳定；Task 8 只有一个 Cutover Owner 写入，Packaging/Domain Owners 只提供签署与 review，不并行拆分删除。
- independent acceptance 不参与实现，也不使用 candidate 自报的 expected bytes。

## 7. 删除、回滚与证据规则

### 7.1 删除前四重条件

任何旧模块只有同时满足以下条件才能进入 Task 8 allowlist：

1. capability ledger 映射率 100%。
2. 新 API source/wheel 场景全部通过。
3. 内部 consumer count 为 0。
4. alias/quirk disposition 与独立 reviewer 已签署。

删除 material 文件后，计划明确报告删除了什么以及能否通过回滚 cutover commit 恢复。实施不得使用 destructive workspace 命令清理用户文件。

### 7.2 中间状态不可发布

Tasks 1–7 可以在开发分支中临时同时存在新内核和旧 oracle，但该状态：

- 不是兼容策略；
- 不更新用户文档以承诺双 API；
- 不构建 release candidate；
- 不允许旧 façade 被新代码调用；
- 必须在 Task 8 一次性删除。

### 7.3 证据不可自证

- baseline/golden 来自 clean D0，不来自 candidate Catalog。
- candidate 只能提供 actual，不能改 expected。
- 性能、架构、LOC、parity checker 的 schema 或阈值在 D0 后变更，必须作为独立 amendment 审核并重新采集可比 baseline。
- focused tests 是诊断证据；只有 exact-SHA full source/same-wheel 闭环可签 D-TECH。

## 8. Definition of Done

0042-R2 只有全部满足时才达到 D-TECH：

- [ ] 历史 0042/ADR/acceptance 未被改写，0042-R2 lineage 清晰。
- [ ] D0 来自 clean exact SHA，能力、模块、测试映射均 100%。
- [ ] 最终只有一个 domain kernel、一个 Catalog、每项能力一个 canonical 计算/领域验证路径、一个 result/artifact model；direct call 与 runtime wrapper 不复制实现。
- [ ] Empyrical、Pyfolio、Alphalens 旧 API、类、导入、extras、profiles、registries 和 compat shell 全部为 0。
- [ ] required capability/scenario missing 0，未裁决差异 0。
- [ ] 每个 leaf capability 恰好一个 implementation fingerprint。
- [ ] source 与同一个候选 wheel 的 normalized semantic output 一致。
- [ ] production physical/logical LOC 相对 D0 至少减少 12%。
- [ ] normalized duplicate/delegate body 相对 D0 至少减少 60%。
- [ ] illegal import edge、cycle、反向依赖和 optional leakage 为 0。
- [ ] coverage、Ruff、format、mypy、MkDocs、package、dependency matrix 全绿。
- [ ] 所有 workload 不回退，并达到预登记性能提升门。
- [ ] report/browser/PDF/XLSX/artifact lifecycle 使用同一 wheel 验证。
- [ ] 许可证、NOTICE、来源和最终 wheel 内容已复核。
- [ ] acceptance manifest 绑定 exact commit/tree/wheel/sdist/toolchain/baseline digest。
- [ ] D-TECH verdict 绑定 tested parent SHA；evidence-only child 只含两份允许的质量文档且通过 parent/allowlist gate。
- [ ] 没有自动 merge、push、tag、publish 或远端治理变更。

## 9. 执行交接

实施者开始前必须输出：

1. R2 pre-document baseline SHA、当前 expected head 和 ancestry 证明。
2. clean worktree 状态与 dirty 用户改动隔离方式。
3. D-ID/D-BREAK local-decision record，以及 D0 尚未通过的边界说明。
4. D0 capability/module/test inventory 摘要与证据目录。
5. 当次只领取一个 task/tranche 的 ownership 和写入 allowlist。
6. 预计 commit、验证命令、回滚点和需要独立 reviewer 的差异。

每个 task 完成后必须交接：

- exact commit/tree；
- 实际修改文件和 `git diff --stat`；
- focused 与 full test 结果，二者分开报告；
- capability/scenario/consumer/duplicate/LOC/performance 增量；
- source/wheel 证据与 artifact digest；
- 未完成项、blocked gate 和下一 owner；
- 明确说明是否发生删除，以及恢复方式。

推荐执行顺序：Task -1 → Task 0 → Task 1；随后并行三条 lane：A 为 Task 2 → Task 3 → Task 5B-attribution（Task 2 后并行插入 Task 5A-risk），B 为 Task 4，C 为 Task 5A-core；所有 sub-tranche 汇合后再执行 Task 6 → Task 7 → Task 8 → Task 9。任何阶段不得为了“先看到新目录”而跳过 D0、parity 或 consumer reversal。
