# 迭代 0041 决策记录（Decision Log）

> 计划日期：2026-08-23
> 决策签署日期：2026-08-23
> 决策人：@cloudQuant（仓库管理员 + 唯一已确认核心维护者，经交互式确认签署）
> 基线证据：`.github/governance/0041/baseline.json`（captured_at_utc 2026-08-23T07:55:27Z，基线提交 2bcb657）
> 状态规则：D0-D4、D6 必须为 `approved` 或 `rejected`，不得留空；D5 允许 `not-triggered`。

## 决策总表

| 编号 | 决策 | 结论 | 状态 | 证据/退出条件 |
|---|---|---|---|---|
| D0 | 分支语义 | **候选 A**：`dev` = 日常贡献入口（默认分支待切），`master` = 稳定发布线；常规变更一律 `feature/* → dev`，`dev → master` 走 promotion PR | approved | 本文件 + `docs/governance/branch-governance.md` + fork PR 演练目标正确 |
| D1 | 权威与镜像 | GitHub 为唯一 PR/审查权威；README 保留 Gitee 地址但**显式标注镜像暂不可用**（2026-08-23 探针 404），不得作为可用克隆引导 | approved | baseline B9 + README 标注文本；恢复镜像需可公开读取 + 同步责任 + SHA 监控 |
| D2 | Owner 矩阵 | CODEOWNERS 全部覆盖路径指向 **@cloudQuant**（唯一已确认有写权限的维护者）；其他维护者名单确定前禁止虚构账号；单维护者限制在 MAINTAINERS.md 透明标注 | approved | codeowners/errors 为空 + 真实 PR 产生预期 review request |
| D3 | 审批门槛与单维护者例外 | 原生 Ruleset 基线：两分支均要求 PR、≥1 位批准、CODEOWNERS 路径 owner 复审、禁 force push/删除、需解决会话。R2 的独立复审由 `PR Governance / policy` 检查 + 人工流程补充；第二维护者到位前，唯一维护者自身的 R2 PR 只能走**管理员例外**：24 小时冷静期、逐项清单、关联 issue、到期日 ≤30 天、`governance-exception` 标签，并明示"不是独立双人审查" | approved | 例外记录可审计；M4 演练覆盖至少一次合规例外 |
| D4 | Required checks | 两个分支均要求稳定检查 **`PR Governance / policy`** 与 **`CI / build`**；context 名称与 app id 在 M4 演练前一律为 `pending-evidence`，只能由真实 PR 的 check-runs API 输出提升为可部署值；路径过滤不得使 required check 永久 Pending | approved | manifest 中 context/app id 与 GitHub API 输出一致 |
| D5 | Merge Queue | **not-triggered**：仅当连续 4 周每天 ≥3 个待合并 PR 或反复基线冲突时单独立项；先让所有 required workflow 支持 `merge_group` 再启用队列 | not-triggered | 月度指标 + 独立 issue；不阻塞 M0-M4 |
| D6 | 私密安全入口 | 采用 **GitHub Private Vulnerability Reporting（PVR）** 为唯一私密入口；当前状态 disabled（baseline B7），管理员已承诺在 Settings → Code security 启用；SECURITY.md 在启用确认前必须标注"待管理员启用确认"。SLA 沿用现有 SECURITY.md：5 个工作日确认、15 个工作日初步评估 | approved（启用动作待完成） | `gh api repos/cloudQuant/fincore/private-vulnerability-reporting` 返回 `{"enabled":true}` 后移除待确认标注 |

## 候选 A 的连带义务（D0 附件）

选择候选 A 不是"反转箭头"，以下义务随 D0 一并生效：

1. **默认分支**：记录为"将在 Ruleset 生效前切至 `dev`"；实际 PATCH 操作由管理员在 M4 阶段带演练证据执行。切换前，CONTRIBUTING、PR 模板与 Issue Form 必须明确说明"fork PR 默认基线仍为 master，请手动选择 dev"的过渡例外。
2. **promotion 方向**：`dev → master` promotion PR 是唯一进入发布线的常规通道；`master → dev` 仅允许同步 PR（保持集成线新鲜）。
3. **dev 的 CI 入口**：`ci.yml` 与 `test-priority.yml` 的 `pull_request`/`push` 触发改为 `[master, dev]`，移除不存在的 `main`；`dev` 的 PR 与 push 获得同等可见验证。
4. **发布责任不变**：`publish.yml` 仍只消费 `master` 上与 Release 同 SHA 的已验证候选工件；`dev` 永不直接发布。
5. **文档部署不变**：`docs.yml` 仍只从 `master` 部署稳定文档站点。
6. **dev 过期规则**：`dev` 上未晋升差异最长存续 14 个自然日或下一个发布候选评审日（先到者），届时必须 promotion、放弃，或先经 `master → dev` 同步 PR 解决分叉；禁止 force push 重置受保护分支。

## 签署与修订规则

- 本文件只能由决策人修订；每次修订追加带日期的修订条目，不得静默改写已签署结论。
- 实施者（agent 或人）不得代替决策人新增 `approved` 结论。
- 任何决策从 `approved` 回退为 `rejected`/`blocked` 时，M1-M4 中依赖该决策的产物必须在同一提交内同步修正或标记失效。
