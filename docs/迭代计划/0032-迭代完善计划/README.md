# 0032 迭代完善计划（稳定性 + 世界一流能力建设）

## 问题发现日期

2026-02-13

---

## 一、当前项目体检结论（基于实测）

### 1) P0 级阻断问题

1. `fincore/metrics/drawdown.py:408-447` 存在缩进错位、重复赋值和变量名错用（`start_date/end_date` 与 `start_idx/end_idx` 混用），导致模块无法导入。
2. `python -m compileall -q fincore` 失败（`drawdown.py` 语法错误）。
3. `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q` 在收集阶段即报错（`drawdown.py` 触发 `IndentationError`）。

### 2) 回归影响范围

1. 由于大量指标依赖 `drawdown`，同一根因引发大面积连锁失败（在排除部分目录后的一次全量回归中，失败列表达到 469 条）。
2. 受影响模块包括：`core/context`、`metrics/rolling`、`metrics/yearly`、`metrics/ratios`、`pyfolio` tearsheets 等。

### 3) 结构性 Bug / 质量风险

1. `fincore/plugin/__init__.py:56-59`：`register_metric` 把“函数执行结果”注册到表中，而不是注册函数本身；装饰后并不会立即注册。
2. `fincore/plugin/__init__.py:101-104`：`register_viz_backend` 注册的是实例且发生在调用时，不符合“注册后可发现”的常见语义。
3. `fincore/plugin/__init__.py:114-152`：`priority` 参数未生效，hook 执行顺序与文档不一致。
4. 插件系统存在双实现（`fincore/plugin/__init__.py` 与 `fincore/plugin/registry.py`），语义不一致，维护风险高。

### 4) 功能待完善（与“世界一流”目标差距）

1. `fincore/attribution/fama_french.py:318-356` 仍为占位数据获取。
2. `fincore/attribution/style.py:505-550` 仍为占位实现且返回随机数据，不可用于生产。
3. 数据层在类型契约和接口一致性上仍有缺口（`fincore/data/providers.py` 存在多处 override 与参数类型不一致信号）。
4. 测试发现机制不完整：`tests/test_attribution/__init__.py` 含测试代码但默认不会被 pytest 发现，存在“假通过”风险。

---

## 二、0032 目标定义

1. 先恢复“可运行、可测试、可发布”的工程基线。
2. 再修复插件系统语义，打通扩展生态能力。
3. 补齐关键真实数据能力（归因因子数据），从“演示可用”走向“生产可用”。

---

## 三、0032 迭代任务分解

## Phase A（P0，必须先做）

1. 修复 `drawdown.py` 语法与逻辑回归（含 `get_max_drawdown_period` / `max_drawdown_days`）。
2. 增加 `drawdown` 模块最小导入与关键路径单测，防止再次出现语法级回归。
3. 验收：
   - `python -m compileall -q fincore` 通过。
   - `pytest tests/test_core tests/test_empyrical tests/test_pyfolio -q` 不再因 `drawdown` 导入失败。

## Phase B（P1，扩展能力修复）

1. 统一插件实现（建议保留 `plugin/registry.py` 为单一事实来源）。
2. 修复三个装饰器行为：
   - `register_metric` 注册函数本身；
   - `register_viz_backend` 在装饰阶段注册类（或工厂）并保持语义一致；
   - `register_hook` 实现基于 `priority` 的有序执行。
3. 为插件系统补充单测（注册时机、返回类型、优先级排序、重复注册策略）。

## Phase C（P1，核心功能增强）

1. 将 Fama-French / Style 数据获取从占位实现升级为“可配置 provider + 缓存”。
2. 禁止默认返回随机占位数据；无数据时返回明确异常和可观测日志。
3. 为因子数据增加最小离线回放样例，保证 CI 可重复测试。

## Phase D（P1，质量门禁）

1. 将 `mypy`（核心包）纳入 CI，至少覆盖：`fincore/core`、`fincore/metrics`、`fincore/plugin`、`fincore/data`。
2. 将测试发现与目录约定标准化（避免测试代码放在不会被发现的 `__init__.py`）。
3. 补充“全模块导入 smoke test”，防止语法错误在运行时才暴露。

## Phase E（P2，迈向世界一流）

1. 启动优化模块 MVP（有效前沿 + 风险平价 + 约束求解接口），对齐 0027 规划。
2. 报告层模块化（拆分 `report.py`），提升可扩展性与可维护性。
3. 增加面向机构用户的能力清单：多组合对比、参数快照、结果可追溯（元数据 + 配置落盘）。

---

## 四、0032 验收标准（量化）

1. P0 验收：
   - 语法检查 0 错误；
   - 与 drawdown 相关的 core/empyrical/pyfolio 测试不再出现导入级错误。
2. 工程验收：
   - 插件系统行为与文档一致（新增测试全绿）；
   - 不再存在“随机占位数据默认返回”的生产路径。
3. 质量验收：
   - CI 增加 mypy 关卡并可稳定通过；
   - 测试发现覆盖规则明确并落地。

---

## 五、建议执行顺序（一个迭代内）

1. 第 1 天：Phase A（恢复可运行）。
2. 第 2-3 天：Phase B（插件语义修复 + 测试）。
3. 第 4-5 天：Phase C（归因数据能力上线）。
4. 第 6 天：Phase D（CI 质量门禁）。
5. 第 7 天：Phase E 设计评审与 MVP 启动。

---

## 六、迭代风险与控制

1. 风险：修复 `drawdown` 后会暴露第二层逻辑回归。
   - 控制：按模块分批回归测试，先 core/metrics，再 pyfolio。
2. 风险：插件系统“行为变更”影响已有用户代码。
   - 控制：提供兼容层与 deprecation 提示，保留一个过渡版本。
3. 风险：外部数据源不稳定导致 CI 波动。
   - 控制：引入本地 fixture + mock provider，外部网络测试改为可选集成测试。
