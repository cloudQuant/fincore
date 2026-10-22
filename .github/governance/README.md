# Governance configuration source of truth（治理配置源）

本目录是 fincore 分支保护的**期望状态**（expected state）唯一来源。

## 规则

1. **manifest ≠ 已生效设置。** `rulesets/*.json` 描述管理员*应当*应用的 GitHub
   Ruleset 基线；在管理员通过 UI/API 实际应用并通过演练验证之前，它们只是期望，
   不得对外宣称分支已受保护。
2. **required check 的 `context`/`app_id` 在 M4 演练前必须是字面量
   `"pending-evidence"`。** 只有真实 PR 的 check-runs API 输出才能将其替换为
   可部署值（见 `.github/governance/0041/decisions.md` 的 D4）。
3. **离线比较器**：`scripts/ci/verify_github_governance.py` 接受保存下来的
   GitHub API 快照 JSON（无网络、无 token），输出机器可读的 `PASS`/`DRIFT`
   报告。用法：

   ```bash
   # 管理员导出 live-state 后：
   python scripts/ci/verify_github_governance.py \
     --expected .github/governance/rulesets \
     --labels .github/governance/labels.json \
     --actual <path-to-live-state.json>
   ```

   live-state 文件格式见 `tests/quality/fixtures/github_governance/matching.json`
   （脱敏示例）。原始 API payload 与任何 token 都不得提交进仓库。
4. **决策溯源**：每个 manifest 的 `decision_source` 指向
   `.github/governance/0041/decisions.md`；修改策略必须先改决策记录，再改 manifest。
5. **标签**：`labels.json` 定义治理标签集；标签由维护者 triage 添加，不是
   Ruleset 原生能力——不要幻想 Ruleset 能按标签判断 PR。

## 文件

| 文件 | 内容 |
|---|---|
| `rulesets/master.json` | 发布线 `master` 的期望基线 |
| `rulesets/dev.json` | 贡献入口 `dev` 的期望基线 |
| `labels.json` | 治理标签定义（名称、颜色、说明） |
