# 测试质量监控系统

**目的**: 持续监控测试质量，及早发现回归

---

## 📊 监控指标

### 1. 测试覆盖率

**目标**:
- Core模块: >95%
- P0指标: >90%
- P1功能: >85%
- 整体: >75%

**监控频率**: 每次PR

**工具**:
```bash
# pytest-cov
pytest tests/ --cov=fincore --cov-fail-under=75

# 覆盖率报告
pytest --cov=fincore --cov-report=html --cov-report=json
```

---

### 2. 测试通过率

**目标**: 100%

**监控**:
- P0测试: 必须100%通过
- P1测试: 必须100%通过
- P2/P3: 允许<1%失败（但需修复）

**报告**:
```bash
# 运行并统计
pytest tests/ -v --tb=no | grep -E "passed|failed|error"
```

---

### 3. 性能回归

**目标**:
- P0指标: <10ms (1000点数据)
- 核心功能: <50ms (5000点数据)

**监控**:
```bash
# 基准测试
pytest tests/benchmarks/ --benchmark-only --benchmark-autosave

# 对比基准
pytest tests/benchmarks/ --benchmark-compare=.benchmarks/baseline.json
```

---

## 🔧 CI/CD集成

### GitHub Actions配置

```yaml
name: Test Quality Monitor

on: [push, pull_request]

jobs:
  test-quality:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install -e ".[dev]"

      # 1. P0测试 (必须通过)
      - name: Run P0 tests
        run: pytest -m p0 -v --tb=short
        continue-on-error: false

      # 2. 覆盖率检查
      - name: Coverage check
        run: |
          pytest tests/ --cov=fincore --cov-fail-under=75 \
            --cov-report=xml --cov-report=html

      # 3. 上传覆盖率报告
      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml

      # 4. 性能基准
      - name: Performance benchmarks
        run: |
          pytest tests/benchmarks/test_p0_metrics_performance.py \
            --benchmark-only --benchmark-autosave

      # 5. 边缘情况测试
      - name: Edge case tests
        run: pytest tests/test_edge_cases.py -v

      # 6. 集成测试
      - name: Integration tests
        run: pytest tests/integration/ -v

  # 夜间完整测试
  nightly-full-test:
    runs-on: ubuntu-latest
    if: github.event_name == 'schedule'
    steps:
      - name: Run full test suite
        run: pytest tests/ -v --cov=fincore --cov-report=html

      - name: Generate report
        run: |
          echo "## Test Quality Report" >> $GITHUB_STEP_SUMMARY
          echo "" >> $GITHUB_STEP_SUMMARY
          pytest tests/ --collect-only -q >> $GITHUB_STEP_SUMMARY
```

---

## 📈 质量仪表板

### 指标追踪

| 指标 | 当前值 | 目标值 | 状态 |
|------|--------|--------|------|
| **Core覆盖率** | 99% | >95% | ✅ 达标 |
| **P0测试通过率** | 100% | 100% | ✅ 达标 |
| **整体覆盖率** | ~75% | >75% | ✅ 达标 |
| **测试数量** | 1635 | >1600 | ✅ 达标 |
| **边缘情况覆盖** | 85个 | >70个 | ✅ 达标 |

---

## 🚨 警报规则

### 严重 (阻塞合并)

- ❌ P0测试失败
- ❌ 覆盖率低于70%
- ❌ 性能回归>20%

### 警告 (需要审查)

- ⚠️ P1测试失败
- ⚠️ 覆盖率下降>5%
- ⚠️ 新增代码覆盖率<60%

### 提示 (改进建议)

- 💡 新增模块无测试
- 💡 边缘情况覆盖不足
- 💡 测试执行时间>10分钟

---

## 📋 检查清单

### PR提交前

- [ ] 所有P0测试通过
- [ ] 覆盖率未降低
- [ ] 新代码有测试
- [ ] 性能未回归

### 每周检查

- [ ] 运行完整测试套件
- [ ] 审查覆盖率趋势
- [ ] 检查慢测试
- [ ] 更新基准数据

### 每月检查

- [ ] 生成质量报告
- [ ] 评估测试效果
- [ ] 优化测试性能
- [ ] 更新文档

---

## 📊 报告模板

### 周度报告

```markdown
# 测试质量周报 - Week X

## 覆盖率
- Core: XX%
- Metrics: XX%
- 整体: XX%

## 测试统计
- 总测试数: XXX
- 通过: XXX
- 失败: XXX
- 跳过: XXX

## 性能
- P0平均耗时: XXms
- 慢测试: X个

## 问题
1. ...
2. ...

## 改进计划
1. ...
2. ...
```

---

## 🔍 工具脚本

### 生成质量报告

```bash
#!/bin/bash
# scripts/generate_quality_report.sh

echo "## Test Quality Report - $(date)" > test-quality-report.md
echo "" >> test-quality-report.md

# 覆盖率
echo "### Coverage" >> test-quality-report.md
pytest tests/ --cov=fincore --cov-report=term >> test-quality-report.md
echo "" >> test-quality-report.md

# 测试统计
echo "### Test Statistics" >> test-quality-report.md
pytest tests/ --collect-only -q >> test-quality-report.md
echo "" >> test-quality-report.md

# 性能
echo "### Performance" >> test-quality-report.md
pytest tests/benchmarks/ --benchmark-only --benchmark-sort=mean >> test-quality-report.md

echo "Report generated: test-quality-report.md"
```

---

## 📞 告警通知

### Slack集成

```python
# scripts/notify_test_failure.py

import requests
import os

def notify_slack(message):
    webhook_url = os.environ['SLACK_WEBHOOK_URL']
    payload = {'text': message}
    requests.post(webhook_url, json=payload)

# 使用
if test_failed:
    notify_slack(f"❌ P0 Test Failed: {test_name}")
```

---

**创建日期**: 2026-03-09
**维护者**: 开发团队
**更新频率**: 每月
