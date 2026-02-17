# 0033 持续优化与产品化方案

## 问题发现日期

2026-02-13

---

## 一、0032 后基线状态（实测）

### 已达成

1. `python -m compileall -q fincore` 通过。
2. `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q` 全量无失败（仅外部依赖相关跳过）。
3. `ruff check fincore tests` 全通过。

### 当前仍存在的优化缺口

1. `mypy fincore` 仍有 2 个错误，集中在优化模块：
   - `fincore/optimization/risk_parity.py:57`
   - `fincore/optimization/risk_parity.py:88`
2. 优化求解器鲁棒性不足：`risk_parity/frontier/objectives` 中多个路径直接使用 `res.x`，未对 `res.success` 做统一失败处理（`fincore/optimization/risk_parity.py:80`, `fincore/optimization/frontier.py:81`, `fincore/optimization/objectives.py:152`）。
3. 测试工程质量仍可提升：数据提供方测试仍有较多网络跳过（`tests/test_data/test_providers.py` 多处 `@pytest.mark.skip`）。
4. 文档与产品信息存在漂移：README 测试数仍写为 1299（`README.md:6`, `README.md:307`, `README.md:642`），与当前基线不一致。
5. 新增优化模块尚未在用户文档中系统化呈现（README 中缺少 optimization 使用章节）。

---

## 二、0033 目标

1. 让优化模块达到“可发布质量”（类型正确 + 失败可解释 + 行为稳定）。
2. 降低 CI 与本地验证噪音（减少网络依赖跳过、提升测试确定性）。
3. 补齐对外文档与安装体验，让新增能力真正可被用户发现并使用。

---

## 三、0033 任务分解

## Phase A（P0）优化模块质量闭环

1. 修复 `risk_parity` 类型错误，明确返回类型（建议引入 TypedDict / dataclass 作为返回协议）。
2. 为 `frontier/objectives/risk_parity` 增加统一求解结果校验：
   - `res.success == False` 时抛出带上下文的异常（目标函数、约束、message、status）。
   - 对 NaN/inf 权重做结果守卫。
3. 增加异常场景测试：
   - 协方差近奇异、不可行约束、极端风险预算。

## Phase B（P1）类型与 CI 门禁完善

1. CI `typecheck` 扩展覆盖 `fincore/optimization`（现仅 core/metrics/plugin/data）。
2. 在本地与 CI 增加“优化模块最小 smoke + 类型检查”作为回归门槛。
3. 逐步收紧 mypy 配置，减少 `ignore_errors` 范围（先从 optimization 开始）。

## Phase C（P1）数据测试去网络化

1. 将 `tests/test_data/test_providers.py` 拆分为：
   - 纯单元测试（mock provider/response，不依赖网络）；
   - 集成测试（保留网络与真实 API，单独标记 `integration`）。
2. 在默认 CI 中仅跑单元测试，集成测试按环境变量触发。
3. 目标：默认回归跳过数显著下降，失败信号更聚焦代码本身。

## Phase D（P1）文档与 API 对齐

1. 更新 README 中测试规模、测试命令说明与当前基线。
2. 增加 Optimization 章节（有效前沿、风险平价、约束优化）与示例。
3. 评估是否在 `fincore.__init__` 提供 optimization 快捷入口（或明确推荐 `from fincore.optimization import ...`）。

## Phase E（P2）报告产品化增强

1. 为报告能力补充可选依赖分组（建议新增 `report` extra，包含 Playwright/PyPDF2）。
2. 提供离线图表资源策略（本地静态资源优先，CDN 兜底），降低内网环境失败率。
3. 为 PDF 渲染失败增加可诊断日志（Chromium 未安装、渲染超时、资源加载失败）。

---

## 四、验收标准（量化）

1. 类型质量：`mypy fincore/optimization` 0 错误。
2. 可靠性：优化相关测试新增并全绿，覆盖求解失败路径。
3. 回归质量：默认测试流程中外部网络导致的跳过显著下降（目标 <= 3）。
4. 文档一致性：README 测试规模、能力列表与代码现状一致。
5. 可用性：报告 PDF 依赖安装路径清晰、离线环境可运行（或明确报错指导）。

---

## 五、建议执行节奏（1 周）

1. 第 1-2 天：Phase A（优化模块类型 + 求解失败处理 + 单测）。
2. 第 3 天：Phase B（CI typecheck 扩展）。
3. 第 4-5 天：Phase C（provider 测试去网络化）。
4. 第 6 天：Phase D（README/API 对齐）。
5. 第 7 天：Phase E（report 依赖与离线策略）。

---

## 六、风险与控制

1. 风险：求解器失败处理变严格后，可能暴露历史“静默容错”路径。
   - 控制：先加 warning + 兼容开关，再逐步切换为 hard fail。
2. 风险：去网络化测试后，与真实 API 行为可能出现偏差。
   - 控制：保留集成测试集，按计划定时执行。
3. 风险：README 与 API 调整影响旧教程。
   - 控制：补充迁移说明与最小可运行示例。
