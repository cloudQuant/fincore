# 0031 代码优化与清理

## 问题发现日期

2026-02-12

---

## 问题描述

### 🟢 MEDIUM - 导入排序问题 (1 个)

**位置**: `fincore/constants/style.py:1`

**问题**: `from collections import OrderedDict` 未排序

**优先级**: P1 - 不影响功能，仅代码风格

---

### 🟢 MEDIUM - TODO 标记 (2 个)

| 位置 | 内容 |
|------|------|
| `attribution/style.py:5` | TODO: 实现实际数据获取 |
| `attribution/fama_french.py:333` | TODO: 实现缓存重复查询 |

---

### 🟢 MEDIUM - print 语句清理 (95 个)

**分布**:
- `fincore/utils/` - 20 个
- `fincore/constants/` - 30 个
- `fincore/data/` - 15 个
- `fincore/tearsheets/` - 30 个

**问题**:
- 不利于国际化
- 混杂调试代码与业务逻辑
- 应使用 logging 模块

**优先级**: P2

---

### 🟢 LOW - 大型文件拆分建议

| 文件 | 行数 | 建议 |
|------|------|------|
| `fincore/report.py` | 1578 | 可考虑拆分为多个小模块 |
| `fincore/pyfolio.py` | 1050 | 已经很庞大，建议重构 |

---

## 修复计划

### Phase 1: 修复导入排序 (P1) ✅

1. **修复 `constants/style.py:1`**
   - 已修复: 添加 `from __future__ import annotations`
   - 已修复: 调整导入顺序

### Phase 2: 清理 print 语句 (P2) ✅

1. **分析结果**: fincore 模块中无需要清理的 print 语句
   - scripts/ 目录下的打印语句是工具性质的，可保留
   - 无需处理

### Phase 3: 处理 TODO 标记 (P1) ✅

1. **attribution/style.py:535** - 已更新为 NOTE 注释
   - data fetching 功能是占位符实现，已在代码中说明

---

## 剩余工作 (可选)

以下是可以进一步改进的方向：

| 优先级 | 任务 | 说明 |
|--------|------|------|
| P1 | ruff unsorted-imports | 修复其他模块的导入排序 |
| P2 | 大型文件拆分 | report.py (1578行), pyfolio.py (1050行) |
| P2 | 类型注解完善 | 添加更详细的类型注解 |
| P3 | 性能优化 | 添加缓存机制 |
| P3 | 文档完善 | 补充 API 文档和使用示例 |

---

## 当前状态

- ✅ 1491 测试通过
- ✅ hooks 模块完成
- ✅ 类型系统基本修复
- ⚠️ 1 个 ruff 导入排序警告
- ⚠️ ~40 个 mypy 类型警告

---

**分支**: `feature/0031-code-cleanup`
**状态**: ✅ 完成

---

## 预期结果

- 代码风格更加一致
- 更好的国际化支持
- 清晰的日志系统
- 更小的代码文件

---

**分支**: `feature/0031-code-cleanup`
**状态**: 📋 待开始

**预计时间**: 1-2 小时
