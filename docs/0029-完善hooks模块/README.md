# 0029 完善 hooks 模块

## 问题发现日期

2026-02-12

---

## 🔴 P0-1: hooks 模块架构不完整

### 现状

**目录结构**:
```
fincore/hooks/
├── __init__.py    # 导入不存在的模块
├── _registry.py    # 存在
└── events/         # 空目录！
```

### 问题分析

1. **`hooks/__init__.py` 导入失败**:
   ```python
   # fincore/hooks/__init__.py:5-15
   from fincore.hooks.events import (  # ❌ 模块为空目录
       _EVENT_HOOKS,
       AnalysisContext,
       ComputeContext,
       OptimizationContext,
       ...
   )
   ```

2. **mypy 类型错误 (10个)**:
   ```
   fincore/hooks/__init__.py:5: error: Module "fincore.hooks.events" has no attribute "_EVENT_HOOKS"
   fincore/hooks/__init__.py:5: error: Module "fincore.hooks.events" has no attribute "AnalysisContext"
   ...
   ```

3. **影响**: 整个 hooks 系统不可用，违反 0027 迭代目标

### 根本原因

在 0027 迭代中声明实现了 hooks 系统，但实际只创建了目录结构，核心实现文件缺失。

---

## 🔴 P0-2: __init__.py 中 `unicode_` 兼容性问题

### 问题描述

**mypy 错误**:
```
fincore/__init__.py:10: error: Module has no attribute "unicode_"  [attr-defined]
```

### 原因分析

代码中存在以下兼容性逻辑：
```python
# fincore/__init__.py:7-10
import numpy as _np

if not hasattr(_np, "unicode_"):
    _np.unicode_ = _np.str_  # ⚠️ 此赋值不创建模块属性
```

**问题**: `hasattr(_np, "unicode_")` 检查的是 NumPy 模块，但赋值 `_np.unicode_` 不会反映到 `fincore.__init__` 模块中。

### 正确写法

```python
import numpy as _np

if not hasattr(_np, "unicode_"):
    import sys
    sys.modules["numpy"].unicode_ = _np.str_
    _np.unicode_ = _np.str_
```

或者简化为（NumPy 2.0 已普遍使用）：
```python
import numpy as np

# 直接使用 np.str_，NumPy 2.0+ 已移除 unicode_ 别名
```

---

## 🟡 P1-1: Mypy 类型错误 (共 51 个)

### 分类统计

| 模块 | 错误数 | 主要问题 |
|--------|---------|----------|
| fincore/hooks/__init__.py | 10 | events 模块为空 |
| fincore/attribution/ | 17 | `np.corr` 返回类型、可选值处理 |
| fincore/risk/garch.py | 6 | no-any-return |
| fincore/plugin/registry.py | 4 | classmethod 使用不当 |
| fincore/plugin/__init__.py | 2 | 缺少 return 语句 |
| fincore/data/providers.py | 8 | 类型不兼容 |
| fincore/simulation/ | 1 | no-any-return |
| fincore/viz/interactive/ | 2 | 签名不兼容 |
| fincore/metrics/ | 1 | 赋值类型不兼容 |

### 典型问题模式

1. **`np.corr()` 返回类型问题**:
   ```python
   # np.corrcoef 返回 ndarray，不是 corr
   beta = np.corr(returns, factor_returns)  # ❌ np 无 corr 方法
   ```

2. **可选值未处理**:
   ```python
   # fincore/attribution/style.py:104
   if market_caps is not None:
       size_exposure = _calculate_size_exposure(market_caps, size_quantiles)
   else:
       total_cap = market_caps.sum()  # ❌ market_caps 可能是 None
   ```

3. **Dict 返回类型不匹配**:
   ```python
   # fincore/attribution/fama_french.py:169
   return {"r_squared": r2, "alpha": alpha, ...}
   # 期望 dict[str, float | ndarray] 但返回了 dict[str, dict]
   ```

---

## 🟢 P2-1: TODO 标记 (共 3 处)

### 位置

1. **`fincore/attribution/fama_french.py:1`**
   ```python
   # TODO: Implement caching for repeated queries
   ```

2. **`fincore/attribution/style.py:1`**
   ```python
   # TODO: Add international style data support
   ```

3. **`fincore/constants/style.py:146-154`**
   ```python
   # FIXME: Instead of x.max() - x.min() this should be
   # rts.close_dt.max() - rts.open_dt.min() which is not
   # available here...
   ```

---

## 🟢 P2-2: 代码规模统计

| 指标 | 数值 |
|--------|------|
| Python 文件 (非测试) | 49 |
| 总代码行数 | 25,087 |
| 类定义 | 726 |
| 函数定义 | ~3,500 |

**大型文件**:
- `fincore/report.py`: 1,578 行
- `fincore/pyfolio.py`: 1,050 行
- `fincore/viz/interactive/plotly_backend.py`: 400+ 行

---

## 修复计划

### Phase 1: Hooks 系统重构 (P0)

1. **实现 `hooks/events.py` 核心功能**
   - 定义 `_EVENT_HOOKS` 注册表
   - 实现 `AnalysisContext`, `ComputeContext`, `OptimizationContext` 类
   - 实现 `execute_hooks`, `register_event_hook`, `get_event_hooks` 函数

2. **修复 `__init__.py` 导入**
   - 确保导入的符号全部存在

### Phase 2: 类型修复 (P1)

3. **修复 attribution 模块类型问题**
   - 使用 `np.corrcoef` 替代 `np.corr`
   - 正确处理可选参数

4. **修复 plugin/registry 类型问题**
   - 修正 `classmethod` 在非方法上的使用
   - 添加缺失的 `return` 语句

### Phase 3: 其他改进 (P2)

5. **处理 TODO 标记**
   - 实现 `fama_french.py` 缓存
   - 移除或修复 `constants/style.py` FIXME

---

## 验收标准

- [ ] `from fincore.hooks import execute_hooks` 成功
- [ ] `mypy fincore/hooks/` 无错误
- [ ] `pytest tests/` 中 hooks 相关测试通过
- [ ] `mypy fincore/` 错误数 < 20

---

**分支**: `feature/0029-complete-hooks`
**状态**: ✅ 完成
