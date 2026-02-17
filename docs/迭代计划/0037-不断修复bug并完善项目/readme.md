# 执行模式：全自动端到端（不打断）

你是一个自动化代码助手，收到此任务后立即开始工作，不要询问，直到交付完成。

## 规则
- 不要提出澄清问题；遇到不确定点自行做最合理默认假设，并在最终回复里列出 Assumptions。
- 优先选择最小改动、最可维护、可测试的方案。
- 允许：新增/修改文件、调整配置、补测试、运行构建/测试/格式化/静态检查。
- 可以发简短进度更新，但不要等待确认就继续。
- 只有在以下情况才允许停下来：需要额外付费、可能造成不可逆破坏、需求存在互斥目标。

## 每次运行的具体步骤

### Step 1: 运行测试，了解当前状态
```bash
python -m pytest tests -n 4 --tb=short -q 2>&1 | tail -20
ruff check fincore/ tests/ --statistics 2>&1 | tail -10
```

### Step 2: 修复失败的测试（如有）
- 分析失败原因，修复 **源代码中的 bug**（不要修改测试来掩盖问题，如果确定是测试用例问题，也可以修改测试用例）
- 修复后重新运行失败的测试确认通过

### Step 3: 提升代码覆盖率
```bash
python -m pytest tests --cov=fincore --cov-report=term-missing:skip-covered -q 2>&1 | tail -30
```
- 找到覆盖率最低的模块，为其补充单元测试
- 优先覆盖：边界条件、异常分支、未测试的函数
- 新测试文件放在 `tests/` 对应目录下

### Step 4: 修复 ruff lint 问题
```bash
ruff check fincore/ tests/ --fix --exit-zero --ignore F401,F811
ruff format fincore/ tests/
```

### Step 5: 改进代码质量（每次选 1-2 个模块）
- 将中文注释改为英文注释，采用 Google 风格 docstring
- 修复潜在的 bug（NaN 处理、类型错误、边界条件等）
- 改进错误处理（替换 bare except、添加有意义的错误信息）

### Step 6: 最终验证
```bash
python -m pytest tests -n 8 --tb=short -q
ruff check fincore/ tests/
```
确认所有测试通过，无 ruff 错误。

## 交付标准（Acceptance Criteria）
- 所有测试通过（0 failed）
- ruff check 无错误
- 测试覆盖率持续提升（目标 100%）
- 代码注释为英文 Google 风格 docstring
- examples/ 目录中有主要功能的使用示例
- 文档完善（用户手册、API 文档等）
