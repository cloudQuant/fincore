# Maintainers

> 本文件是治理事实来源之一，与 [GOVERNANCE.md](GOVERNANCE.md)、
> [SECURITY.md](SECURITY.md)、[SUPPORT.md](SUPPORT.md)、
> [.github/CODEOWNERS](.github/CODEOWNERS) 和
> [分支治理](docs/governance/branch-governance.md) 交叉引用。
> 禁止虚构人员、团队或邮箱；没有真实负责人的角色必须显式标注空缺。

## 当前维护者

| GitHub 用户 | 写权限 | 角色 |
| --- | --- | --- |
| [@cloudQuant](https://github.com/cloudQuant) | :white_check_mark: | 核心维护者；暂代全部下述职责 |

## 职责边界

| 领域 | 当前负责人 | 说明 |
| --- | --- | --- |
| Release / 发布 | @cloudQuant | 只有发布线（`master`）上通过全部门禁的同 SHA 候选工件可进入发布流程；见 [branch-governance.md](docs/governance/branch-governance.md) |
| Governance / 治理配置 | @cloudQuant | Ruleset、CODEOWNERS、`.github/governance/` 的唯一整合负责人 |
| Security / 安全响应 | @cloudQuant | 私密漏洞入口见 [SECURITY.md](SECURITY.md)；5 个工作日确认、15 个工作日初步评估 |
| 数值正确性（oracle/property） | @cloudQuant（待补充领域专家） | 稳定算法的独立 oracle 与 property 测试责任 |

## 已知缺口（透明声明，不得伪装）

1. **第二维护者缺口（决策 D3）**：当前只有一名已确认维护者。R2 级（行为/兼容/数值）
   变更要求的"独立复审"在第二维护者到位前无法真实满足；唯一维护者自身的 R2 PR
   只能走管理员例外流程：24 小时冷静期、逐项清单、关联 issue、到期日不超过 30 天、
   打 `governance-exception` 标签，并明示其**不是独立双人审查**。
2. **CODEOWNERS 回退（决策 D2）**：所有覆盖路径暂指向 @cloudQuant 单人；
   新维护者名单确定并确认写权限后必须更新本文件与 CODEOWNERS。
3. **私密漏洞报告启用状态（决策 D6）**：GitHub Private Vulnerability Reporting
   截至基线日（2026-08-23）为 disabled，管理员已承诺启用；启用确认前
   [SECURITY.md](SECURITY.md) 保持"待启用"标注。

## 成为维护者

持续高质量贡献（含可复核的测试与文档证据）的贡献者可被提名。提名由现有维护者
发起，要求：至少 3 个已合并的非平凡 PR、理解[风险级别与门禁]
(docs/governance/branch-governance.md#风险级别)，并愿意承担上表某一领域的轮值。
