# 质量稳态迭代计划（CQ-2026-09）：零功能与逻辑变更

## 0. 元数据与边界

| 字段 | 值 |
| --- | --- |
| 状态 | `PROPOSED`；本轮已落地的低风险门禁修复见第 4 节 |
| 审计时间 | 2026-09-20（Asia/Shanghai） |
| 审计起点 | `master@2354bb5`；工作树已有用户拥有的治理/文档改动，不能作为干净发布基线 |
| 目标 | 让质量门禁可执行、测试工件不污染打包验收，并为后续可证明的复杂度和静态规则收敛建立顺序 |
| 不在范围内 | 数值算法、指标公式、公开 API、异常语义、依赖版本、性能目标、发布、远端、合并与删除用户工件 |

本计划不替代正在进行的 0041 治理工作，也不解除 0043 的前置阻断。任何生产源码重构必须先满足第 6 节的 D0 条件；不能以“代码行减少”替代兼容性或数值行为证明。

### 不变性契约

每个后续变更都必须同时保持以下观察结果不变：

1. 已冻结的公开模块、导出名、调用签名与版本策略；如契约本身需要改变，必须先有独立的兼容性决策，不能随重构顺带更新 fixture。
2. 对固定输入的数值结果、索引/时区语义、NaN/空输入处理、异常类型与消息契约。
3. 可选依赖的惰性导入、数据/报告工件格式与运行时性能基线（除非该任务专门且事先批准性能变更）。
4. 用户当前未提交的文件、忽略文件和远端状态；本计划不授权清理、暂存、提交、推送或删除。

## 1. 当前证据与审计结论

| ID | 发现 | 当前证据 | 状态 | 处置优先级 |
| --- | --- | --- | --- | --- |
| CQ-A01 | MyPy 的 pre-commit hook 指向已删除的 `fincore/core`、`constants`、`plugin` 等路径，钩子无法作为门禁运行。 | 原钩子退出码 2；`python -m mypy fincore` 对 128 个源码文件通过。 | 本轮修复 | P0 |
| CQ-A02 | 打包测试复制源码时没有忽略 `.pytest_tmp`；测试临时产物会进入复制源并导致 `copytree` 竞态。 | 全量离线运行中 17 个 packaging errors；根目录 `.pytest_tmp/` 已被 `.gitignore` 忽略。 | 本轮修复 | P0 |
| CQ-A03 | 当前质量快照不是当前提交的证据。 | `scripts/check_quality_snapshot.py` 返回 `source.commit does not match HEAD`。 | `BLOCKED`：需干净基线重采集 | P0 |
| CQ-A04 | 冻结的 `public-api-0.4.0.dev0.json` 与当前源码漂移；本地还残留已忽略的旧模块 `.pyc`，使 legacy-surface 探针可发现 12 个幽灵模块。 | `snapshot_public_api.py --check ...` 失败；本地探针受仅有字节码的旧目录污染，而从 `git archive HEAD` 构造的干净源副本通过。 | `BLOCKED`：仅兼容契约仍需决策 | P0 |
| CQ-A05 | 37 个函数的 McCabe 圈复杂度大于 10；最高为 `runtime/results.py::_decode`（37）、`factor_analysis/models.py::_fingerprint_bytes`（33）。 | `ruff --select C901` 扫描结果。 | 待分批重构 | P1 |
| CQ-A06 | Ruff 全局忽略 `F401` 与 `E402`，隔离扫描仍发现 5 个未使用导入和 6 个导入顺序问题。 | `ruff --isolated --select F401/E402 fincore`。 | 待逐项收敛 | P2 |
| CQ-A07 | 项目指令引用的 `.joyincode/rules/backend.md` 与 `frontend.md` 不存在。 | 两个路径读取均失败。 | `BLOCKED`：需负责人恢复或签署 waiver | P1 |

已通过但不代表发布就绪的静态检查：Ruff、Ruff format、Bandit、全包 MyPy。它们只说明当前可扫描源码满足这些局部规则；不覆盖质量快照新鲜度、公开 API 决策、干净构建输入或完整发布验收。

## 2. 任务编排

| 阶段 | 任务 | 依赖 | 可修改范围 | 负责人 | 完成判据 |
| --- | --- | --- | --- | --- | --- |
| Q0 | 建立干净 D0 输入 | 0041 用户改动完成或另建独立干净 worktree | 仅外部证据目录 | 质量负责人 | 记录 commit、工作树 cleanliness、解释器、依赖锁定和原始退出码；不得以当前脏工作树生成基线。 |
| Q1 | 修复本地类型门禁 | 无 | `.pre-commit-config.yaml` | 实现者 + 独立验收 | `pre-commit run mypy --all-files` 从仓库根 `fincore` 运行且通过。 |
| Q2 | 隔离打包测试工件 | Q1 无强依赖 | 两个 packaging 测试文件 | 实现者 + 独立验收 | 构建源复制忽略 `.pytest_tmp`，并有回归断言；打包测试在仓库外 `--basetemp` 下通过。 |
| Q3 | 固化契约输入 | Q0 | 契约/测试工具与 fixture（需单独批准） | 架构负责人 | 把“版本化源码”与本地字节码/缓存分离；为 `0.4` fixture 的保留、替换或退休形成单一书面决策。 |
| Q4 | 复杂度热点分片 | Q3 | 每次仅一个模块及其测试 | 模块所有者 | 先录制行为矩阵，再抽取无状态私有 helper；每个提交不新增 C901，目标函数复杂度下降且所有契约、数值、性能用例通过。 |
| Q5 | 缩小静态规则豁免 | Q3、Q4 | `pyproject.toml` 与已验证的单个源码文件 | 模块所有者 | 先为每一项分类为“删除、修正、必要的局部豁免”；禁止用新的全局 ignore 掩盖问题。 |
| Q6 | 重采集质量证据与 CI 门禁 | Q0、Q1-Q5 对应批次完成 | 证据文件/CI（需单独授权） | 质量负责人 | 新快照绑定候选 commit、完整测试/完整性为 0、branch coverage >= 60%，再启用 fail-closed 校验。 |

### Q4 的固定拆分次序

1. `fincore/runtime/results.py`：`_encode`/`_decode`（序列化分派）。先冻结每一种 payload 的字节/结构结果，再仅抽取类型分派表或私有 leaf helper。
2. `fincore/factor_analysis/models.py`：`_fingerprint_bytes`、`_wire_scalar`、`_unwire_scalar`。先覆盖标量、索引、数组、嵌套容器、异常和确定性指纹。
3. `fincore/optimization/{frontier,objectives,risk_parity}.py`。先冻结约束可行性、收敛失败、返回结构和数值容差。
4. 其余 28 个热点按单模块、单语义簇推进；不得为降低复杂度跨领域移动公开 API。

## 3. 验收用例与命令

所有命令使用 Anaconda `base` 环境；测试临时目录必须位于仓库外。`<TMP>` 代表用 `mktemp -d /tmp/fincore-...` 创建的专用目录。

| 用例 ID | 覆盖目标 | 命令/证据 | 通过标准 |
| --- | --- | --- | --- |
| AC-Q1-01 | MyPy hook 可执行 | `/Users/yunjinqi/opt/anaconda3/bin/conda run -n base env PYTHONDONTWRITEBYTECODE=1 python -m pre_commit run mypy --all-files` | exit 0，且输出检查 `fincore`。 |
| AC-Q2-01 | 包装源隔离 | `/Users/yunjinqi/opt/anaconda3/bin/conda run -n base env PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg python -m pytest -q tests/packaging/test_release_consistency.py tests/packaging/test_wheel_contents.py --basetemp <TMP> -p no:cacheprovider` | 所有选择的用例通过；stage source 不含 `.pytest_tmp`。 |
| AC-Q3-01 | 公开 API 决策 | 固定 fixture 后运行 `scripts/snapshot_public_api.py --check <approved-fixture>` | 只有已批准的契约才可通过；当前 `0.4` fixture 漂移不能被静默覆盖。 |
| AC-Q3-02 | legacy 输入隔离 | 在仅含版本化源码、无 `__pycache__` 的临时源副本运行 `tests/contracts/test_removed_legacy_surfaces.py` | legacy spec 与 legacy root export 均为空，或由批准的兼容策略明确列出。 |
| AC-Q4-01 | 复杂度不倒退 | `ruff check fincore --select C901 --config 'lint.mccabe.max-complexity=10'` | 目标函数数量下降；未触及模块无新增发现。 |
| AC-Q4-02 | 行为保持 | 对每个热点运行对应 `tests/numerical`、`tests/property`、`tests/parity`、`tests/performance` 节点 | 数值、异常、工件和性能阈值不倒退。 |
| AC-Q5-01 | Lint 豁免收敛 | `ruff check --isolated --select F401,E402 fincore` 与正常 Ruff 门禁 | 已处理项为 0 或转为带原因的最小 per-file 例外。 |
| AC-Q6-01 | 当前质量快照 | 在干净 worktree 中运行质量采集器，再运行 `scripts/check_quality_snapshot.py --snapshot docs/quality/current-baseline.json` | commit 匹配、完整性与运行退出码为 0、branch coverage >= 60%。 |
| AC-Q6-02 | 全量离线回归 | `/Users/yunjinqi/opt/anaconda3/bin/conda run -n base env PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg python -m pytest -q -m 'not integration_online' --basetemp <TMP> -p no:cacheprovider` | 全部非在线测试通过；在线与外部发布验证另标 `NOT_RUN`，不能计为通过。 |

## 4. 本轮已落地的最小修复

| 变更 | 理由 | 行为影响 | 独立验证 |
| --- | --- | --- | --- |
| `.pre-commit-config.yaml`：MyPy 入口改为 `fincore`，并设 `pass_filenames: false` | 修复已删除路径导致的失效 hook，并让检查根随当前包结构演进。 | 无运行时影响；仅本地质量门禁。 | AC-Q1-01 通过。 |
| `tests/packaging/test_release_consistency.py`、`tests/packaging/test_wheel_contents.py`：忽略 `.pytest_tmp`，并增加 staged-source 断言。 | 防止测试临时目录进入构建源复制而造成竞态/假失败。 | 无包或运行时代码影响；仅测试隔离。 | 20 个 packaging tests 通过；Ruff check/format 通过。 |

## 5. 当前验收状态

| 门 | 状态 | 说明 |
| --- | --- | --- |
| Lint / format / Bandit / 全包 MyPy | `PASS` | 当前源码的局部静态检查通过。 |
| MyPy pre-commit | `PASS` | 本轮修复后独立复跑通过。 |
| Packaging 测试（20 节点） | `PASS` | 使用仓库外 `--basetemp` 复跑。 |
| 全量离线测试 | `INCOMPLETE` | 初跑暴露 `.pytest_tmp` 污染和 legacy bytecode；Q2 已修复且干净源副本的 legacy 探针通过，但尚未在修复后重跑完整矩阵。 |
| Legacy-surface 契约 | `PASS`（干净源副本） | `git archive HEAD` 的临时源副本通过；当前工作树仍含已忽略的旧 `.pyc`，本地直接探针会发现 12 个 legacy specs，且本轮不删除用户工件。 |
| 0.4 公共 API fixture | `FAIL` | 当前源码与冻结 fixture 漂移，需独立兼容性决策。 |
| 当前质量快照 | `BLOCKED` | 快照 commit 不匹配 HEAD；当前工作树也不干净。 |
| 缺失项目规则 | `BLOCKED` | 需项目负责人恢复规则文件或签署 waiver。 |
| 发布/远端/合并 | `NOT_RUN` | 不在本次授权范围。 |

## 6. 决策门、回滚与交接

1. **D0（开始源码重构）**：仅在 Q0、CQ-A03、CQ-A04、CQ-A07 都有书面结论后开放。否则只能修复测试/工具可靠性，不能声称代码结构已经安全可改。
2. **D1（每个复杂度批次）**：提交前必须有 API/数值/异常/性能基线；复杂度下降不是单独的通过条件。
3. **D2（质量快照）**：只能从干净候选提交采集，不得由候选自身修改预期证据或覆盖历史 fixture。
4. **回滚**：每项任务独立提交；出现任一行为比较、性能、公开 API 或完整性失败时，仅 revert 该任务提交。不得 reset、force-push 或删除用户拥有的忽略文件。
5. **交接包**：每批交接 `git diff --check`、精确测试节点与退出码、工具版本、基线 commit、变更前后 C901 计数、API/数值比对和明确的 `PASS`/`FAIL`/`BLOCKED`/`NOT_RUN` 状态。
