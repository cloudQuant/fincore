# 分支治理（Branch Governance）

> 决策来源：`.github/governance/0041/decisions.md`（D0 = 候选 A，2026-08-23 签署）。
> 本文件与 [CONTRIBUTING.md](../../CONTRIBUTING.md)、[GOVERNANCE.md](../../GOVERNANCE.md)
> 交叉引用；四处描述必须一致。历史迭代快照（`docs/迭代计划/`）不受本文件约束。

## 分支模型（候选 A）

| 分支 | 角色 | 允许进入的变更 | 进入方式 | 离开/同步方式 | 禁止事项 |
|---|---|---|---|---|---|
| `dev` | **日常贡献入口**（默认分支，切换前见下方过渡说明） | 常规功能、普通修复、文档、测试、重构、社区贡献 | `feature/*` → `dev` 的 PR；无直接 push | 经 `dev → master` promotion PR 进入发布线 | 不发布、不直接接收发布线专属热修复的替代通道 |
| `master` | **稳定发布线** | 已在 `dev` 完成验证的 promotion 变更；`hotfix/*` 安全/发布阻断修复 | promotion PR 或 `hotfix/*` PR；无直接 push | 从本分支的已验证候选产生发布；修复须 1 个工作日内前移 `dev` | 不接受未经完整验证的常规功能直推 |
| `origin/gh-pages` | 已于 2026-10-22 删除的遗留部署分支 | 不再写入 | — | 文档站点仍由 `master` 的 docs workflow 部署 | 不作为源码、审查或当前文档产物分支 |

### 过渡说明（默认分支切换前）

GitHub 默认分支当前仍为 `master`（切换动作按决策记录留待 Ruleset 生效前的管理员阶段执行）。
因此 fork 后新建 PR 时 GitHub 可能默认以 `master` 为基线——**请手动把目标分支改为 `dev`**，
除非你正在提交 `hotfix/*` 或 promotion PR。

## 目标工作流

```text
普通贡献：
fork / feature/* ── PR + R0/R1 验证 ──> dev

发布候选：
dev ── promotion PR（完整验证 + 差异/回滚说明）──> master ── Release ──> PyPI（人工环境批准）

发布线热修复（安全/发布阻断）：
hotfix/<issue>-<slug> (from master) ── PR + R3 门禁 ──> master ── 前移 PR（≤1 个工作日）──> dev

发布线同步（保持集成线新鲜）：
master ── 同步 PR ──> dev
```

规则：

1. `dev → master` 的 promotion PR 是进入发布线的**唯一**常规通道；promotion PR 必须列出
   变更范围、已排除内容、完整验证结果、性能/兼容性差异与回滚点。
2. `master` 上的 hotfix 必须创建前移 PR 或书面记录"不前移"的理由与负责人；不得以
   "已修 master"关闭 `dev` 上的同类风险。
3. `dev` 上未晋升差异的最长存续期为 **14 个自然日**或下一个发布候选评审日（先到者）；
   届时必须 promotion、明确放弃，或先经 `master → dev` 同步 PR 解决分叉。
   禁止用 force push 重置受保护分支。
4. 发布只能消费 `master` 上与成功 CI 同 SHA 的候选工件；`dev` 永不直接发布。
5. 文档站点仍只从 `master` 部署。

## PR 目标分支决策表

| 情况 | 默认目标 | 最小证据 | 合并后动作 |
|---|---|---|---|
| 文档、示例、非行为性工具（R0） | `dev` | 受影响文档 strict build 或静态检查 | 进入下一个 promotion 候选集 |
| 局部功能/修复，未改变公开行为或数值算法（R1） | `dev` | 回归测试与 fast suite | 进入下一个 promotion 候选集 |
| 公开 API、严格兼容 façade、数值内核、冻结清单或跨模块集成（R2） | `dev` | R2 验证（compat/numerical/oracles/property + API snapshot）、必要 ADR、code owner 复审 | 随 promotion 进入发布线 |
| 破坏性公开 API（R2/R3） | `dev` | ADR、SemVer/弃用窗口、API snapshot 差异报告 | CHANGELOG breaking 区块；不因合并改变 readiness seal |
| 安全/供应链/发布配置（R3） | `hotfix/* → master`（经私密渠道先报告） | 最小复现、修复回归、R3 清单 | 1 个工作日内前移 `dev`；不在公开 issue 泄露细节 |
| 发布候选集成（promotion） | `dev → master` | 完整验证、差异/回滚说明 | 触发发布流程（人工批准） |

## 风险级别

路径只能给出下限；任何改变结果、公开行为、依赖供应链或发布权限的改动都必须升到
对应等级。不能因 diff 小、测试数量多或作者熟悉模块而降级。

| 等级 | 典型范围 | 最低评审 | 最低验证 |
|---|---|---|---|
| R0 文档/元数据 | 普通 docs、mkdocs_docs、示例、非策略性的 Issue Form/PR 文本 | 1 位维护者 | 受影响文档 strict build 或静态检查 |
| R1 局部实现 | 无公开行为变化的局部测试、非数值工具、单后端展示代码 | 1 位模块 owner | 定向回归 + fast suite |
| R2 行为/兼容/数值 | `fincore/empyrical.py`、`fincore/pyfolio.py`、`fincore/alphalens/`、`fincore/factor_analysis/`、`fincore/core/`、`fincore/_dispatch.py`、`fincore/_registry.py`、`fincore/contracts/`、任何改变计算结果的 `fincore/metrics/` 代码、`tests/compat/fixtures/` | code owner + 独立维护者（缺失时走 D3 例外） | 全量 CI；compat、numerical、oracles、property；当前 API snapshot；必要 ADR |
| R3 安全/治理/发布/供应链 | `.github/workflows/`、`.github/CODEOWNERS`、`.github/governance/`、`publish.yml`、`pyproject.toml`、`setup.py`、requirements/constraints、许可证/notice、私密安全修复 | 核心维护者明确批准；涉及 R2 时叠加其证据 | 目标分支完整 CI、打包/发布检查、人工变更审查和回滚记录 |

R0 不包括 Actions workflow、CODEOWNERS、Ruleset manifest、发布/安全策略——这些文件
即使只有一行改动也至少是 R3。`metrics/` 下只要改动数值公式、边界条件、输入清洗或
输出投影，即为 R2。

R2 的本地诊断命令（CI 全量结果仍是合并依据）：

```bash
MPLBACKEND=Agg python -m pytest -o addopts='' tests/compat tests/numerical tests/oracles tests/property -q --tb=short --maxfail=0
python scripts/snapshot_public_api.py --check tests/contracts/fixtures/public-api-0.4.0.dev0.json
python -m mypy fincore --ignore-missing-imports
```

## 原生规则与人工责任的边界

| 控制项 | 实现位置 | 不应作出的错误承诺 |
|---|---|---|
| 禁止 force push、删除、直接更新；要求 PR、1 个批准、最新提交复审、解决会话 | GitHub Ruleset（M4 阶段由管理员应用） | 不能把仓库内的 JSON manifest 当成已生效保护 |
| 路径 owner 复审 | CODEOWNERS + Ruleset 的 require code-owner review | 一个文件多个 owner 时任一 owner 批准即可；不天然等于"两名独立维护者" |
| R2 路径识别、风险标签建议、证据声明 | `PR Governance / policy` workflow（无路径过滤）；`CI / build` 独立执行实际门禁 | workflow 不能仅凭 PR 模板复选框证明数值正确 |
| R2 独立复审与单维护者例外 | GitHub review 事件 + policy check + issue/PR 审计记录 | 管理员 bypass 不能被伪装成正常审批 |

## 管理员例外与 bypass 记录

- 所有 bypass 必须最小化，并留下公开可审计的记录（关联 issue、理由、时间、后续修复 PR），
  不得包含漏洞细节或凭据。
- D3 单维护者例外：24 小时冷静期、逐项清单、`governance-exception` 标签、到期日 ≤30 天、
  明示"不是独立双人审查"。
- CI 永不持有管理员令牌或 Ruleset 写权限。

## 镜像与权威（决策 D1）

GitHub（`cloudQuant/fincore`）是唯一的 PR、审查与发布权威。README 中保留的
Gitee 地址（`gitee.com/yunjinqi/fincore`）**当前不可公开读取**（2026-08-23 探针 404），
仅作历史参考标注；在建立可验证的公开镜像、非机密同步机制与 SHA 监控之前，
不得将其描述为可用克隆来源。
