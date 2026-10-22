# Fincore 0043 开发准入审查

- 审查日期：2026-08-24
- 审查范围：0043 重构计划与实施前置条件
- 计划版本：`docs/plans/2026-08-24-fincore-structural-consolidation.md`
- 计划 SHA256：`4a8ee66010ace8f269a7303a27d1667f36229c497df51743531e5a38b7656b50`
- 当前仓库：`master@2bcb65773f01dd836b5fb4d928741ff1b072179e`
- 结论：**NOT READY — BLOCKED AT D-1/D0**

> 2026-10-22 说明：0041 治理实现已落到 `dev`；本文保留为落地前的历史准入快照，文中“0041 未形成 landed SHA”不再代表当前状态。

## 1. 结论

0043 的 canonical 实施计划已经完成，并通过两路独立只读复核，最终结果均为 `P0=0, P1=0`。计划明确采用 breaking `0.5.0.dev0`：不保留 Empyrical、Pyfolio、Alphalens 的旧 API、类、profile、extra 或 shim；功能完整性改由独立 capability ledger、语义 golden、source/wheel scenarios 与领域测试证明。

这不代表仓库现在可以开始业务重构。当前仍缺少 0041 的落地提交、项目规范文件、有效质量基线和 D-1 验收信任根，因此 0043 必须保持 `Proposed/BLOCKED`。本次没有修改生产代码、测试、CI、远端分支、Ruleset 或发布状态。

## 2. 已具备的计划条件

1. 0043 只有一份 canonical 计划，不复制第二份 numbered plan。
2. 目标拓扑、领域 ownership、任务依赖、删除顺序、回滚和 handoff 已明确。
3. “功能不减少”已与“旧 API 兼容”解耦：required `capability_id` 和 `scenario_id` 不得减少，但旧路径必须从新 wheel 消失。
4. 生产 Python 代码量采用相对 D0 的硬门：physical 与 logical LOC 均至少净减少 12%，delegate/registry duplicate 至少减少 60%；当前参考上限分别为 43,480 与 35,270。
5. 兼容测试中的数值、标签、图表和工作流语义先迁入 parity/domain tests，之后才允许删除旧 façade 与 compat 基础设施。
6. T7 是单一 atomic cutover；T8 只物化证据与文档；D8a/D8b 不自动 merge、tag 或 release。
7. hosted/self-hosted 两种验收模式互斥，expected bytes 必须来自仓库外已验签 D0 bundle，candidate 不能自证。
8. 当前 User-owned 仓库不具备本计划要求的 merge queue 路径，真实 merge 保持 BLOCKED；未来若迁移到组织仓库，必须作为新 trust epoch 重做 D-1、D0、D8a 和 D8b。

计划结构校验结果：1722 行、12 个 Task、6 个 `~~~` fence、0 trailing whitespace、0 bare `python/python3` 命令。两位复核者在最终 SHA 上均报告 `P0=0, P1=0`。

## 3. 当前阻断项

| 阻断项 | 当前证据 | 解锁条件 |
| --- | --- | --- |
| 0041 未形成可继承的 landed SHA | `docs/迭代计划/README.md` 将 0041 标为 active；工作树含大量 0041 治理/CI 的 modified 与 untracked 文件 | 0041 先按其自身流程落到命名 commit/ref，记录 `0041_LANDED_SHA`；0043 clean base 必须证明是其后代 |
| 后端与前端规范文件缺失 | `.joyincode/rules/backend.md`、`.joyincode/rules/frontend.md` 均不存在 | 恢复文件，或由有权负责人签署明确 waiver |
| 当前质量快照不可作为 D0 | checker 报告 snapshot commit 与 HEAD 不匹配，branch coverage 45.0% 低于 60% | 在 0041 clean 后代按 T0 流程重新采集、签署并冻结 D0 quality/architecture/performance 基线 |
| D-1 执行模式未选择 | 没有已签署的 `D1_EXECUTION_MODE` 或对应外部 runner/service records | 在 hosted 与 self-hosted 中选择且只选择一种，完成该模式的独立审批和预登记 |
| hosted 远端权限与服务未授权 | 没有针对 bootstrap branch、唯一非发布 dispatch tag、三份 Ruleset、App/broker/proxy/two ledgers 的具体授权和可验 attestation | 逐项取得授权并完成 Task -1；缺任一项则继续 BLOCKED |
| self-hosted 三平台证据未提供 | 没有签名 gate schema、acceptance-owner/signer、隔离 runner、Ubuntu/macOS/Windows attestation 与两类 ledger contract | 若选择 self-hosted，先在仓库外完成全部预登记和 live race 验证 |
| D0 外部信任锚不存在 | 没有 `0043-d0-trust-anchor.json`、attestation 与 expected bundle | 完成 T0a/T0b/T0c 的 clean、不可 amend 提交序列，并由独立 signer 生成外部不可变资产 |

## 4. 精确解锁顺序

1. 完成 0041，记录 `0041_LANDED_SHA`，保留现有未提交改动的 ownership，不用 0043 覆盖它们。
2. 恢复 `.joyincode/rules/backend.md` 与 `frontend.md`，或签署 waiver。
3. 从 `0041_LANDED_SHA` 的 clean 后代开始，选择 `hosted` 或 `self_hosted` D-1 模式。
4. 若为 hosted：分别授权 bootstrap tooling change、bootstrap branch、唯一非发布 dispatch tag、三份 0-bypass Ruleset，以及 App、broker、GET-only proxy、authorization ledger、merge-receipt ledger。
5. 若为 self-hosted：冻结 acceptance-owner key、signer、gate schema、runner contract、两类 ledger contract和三平台 attestation。
6. 完成 T0a tooling commit；在 clean T0a 上捕获并签署 capability ledger/goldens，形成 T0b；再在 clean T0b 后代采集 quality/architecture/performance，形成 T0c。禁止 amend 或用 candidate 文件作为 expected。
7. 生成并验签仓库外 D0 anchor、attestation 和 expected bundle；只有 D0 通过后才能启动 T1。
8. 真实 merge、tag、publish 均不在本次准入授权内；到达对应 gate 后仍需单独授权。

## 5. 本次只读证据

| 命令 | Exit | 结果 |
| --- | ---: | --- |
| `git rev-parse HEAD` | 0 | `2bcb65773f01dd836b5fb4d928741ff1b072179e` |
| `git branch --show-current` | 0 | `master` |
| `git status --short` | 0 | 0041 治理/CI 与本文档相关改动仍在工作树；不是 clean baseline |
| `test -f .joyincode/rules/backend.md` | 1 | 缺失 |
| `test -f .joyincode/rules/frontend.md` | 1 | 缺失 |
| `/Users/yunjinqi/opt/anaconda3/bin/conda run -n base python scripts/check_quality_snapshot.py --snapshot docs/quality/current-baseline.json` | 1 | commit mismatch；branch coverage `45.0% < 60.0%` |
| 计划结构/SHA 校验 | 0 | SHA256 与本报告一致；结构检查无异常 |

本次未运行完整测试套件、wheel matrix、浏览器/PDF、性能或发布门，也不据此声称实现已通过；这些属于 T0–T10 的后续证据。

## 6. 准入判定

- 计划质量：**PASS**（两路独立复核 `P0=0, P1=0`）。
- 业务重构实施：**BLOCKED**。
- 首个允许动作：先完成 0041 handoff 与规则恢复/waiver，然后执行 0043 Task -1；在此之前不得开始 T0 或修改生产代码。
