# 0030 类型系统改进

## 问题发现日期

2026-02-12

---

## 问题描述

### P0-1: attribution 模块类型错误 (17 个)

**主要问题**:

1. **`np.corr()` 不存在** - 应使用 `np.corrcoef()`
   - 位置: `fama_french.py:138`, `style.py:435`

2. **可选值未处理** - `None` 值调用方法前未检查
   - 位置: `style.py:104` - `market_caps.sum()` 在 None 检查前调用

3. **Dict 返回类型不匹配** - 返回嵌套 dict 而非基础类型
   - 位置: `fama_french.py:169` - 返回 `dict[str, dict]` 而非 `dict[str, float | ndarray]`

### P1-1: plugin/registry 类型问题 (4 个)

**主要问题**:

1. **`@classmethod` 装饰器使用不当** - 在普通函数上使用
   - 位置: `registry.py:107`

2. **返回类型不兼容** - 装饰器返回复杂类型而期望 `type`
   - 位置: `registry.py:119`

3. **缺少 return 语句** - 某些代码路径没有返回值

### P2-1: 其他模块类型错误 (11 个)

| 模块 | 问题数 | 说明 |
|--------|---------|------|
| fincore/data | 8 | override 类型不兼容 |
| fincore/simulation | 1 | no-any-return |
| fincore/viz/interactive | 2 | 签名不兼容 |
| fincore/core | 1 | union-attr |
| fincore/report | 2 | no-any-return |
| fincore/utils | 1 | 条件函数签名不一致 |
| fincore/metrics | 2 | 赋值类型不兼容 |

---

## 修复计划

### Phase 1: attribution 类型修复 (P0)

1. **修复 `np.corr` → `np.corrcoef`**
   - 搜索所有使用 `np.corr` 的地方
   - 替换为 `np.corrcoef` 或 `np.corrcoef(x, y=None)`

2. **修复可选值处理**
   - 在 `style.py:104` 添加 `market_caps is not None` 检查
   - 确保安全调用 `.sum()`, `.mean()` 等方法

3. **修复返回类型**
   - `fama_french.py:169` - 确保 `beta` 参数类型正确

### Phase 2: plugin/registry 类型修复 (P1)

1. **移除 `@classmethod` 装饰器**
   - `create_instance` 是普通函数，不应该用 `@classmethod`

2. **修复返回类型**
   - 调整 `register_viz_backend` 返回类型匹配

3. **添加缺失的 return 语句**
   - 确保 `register_viz_backend` 的 wrapper 正确返回

### Phase 3: 其他模块类型修复 (P1)

1. **data/providers.py** - 修复 `fetch` 方法签名类型
2. **simulation/base.py** - 添加返回类型注解
3. **viz/interactive/** - 修复方法签名兼容性
4. **core/context.py** - 修复 union-attr 问题
5. **report.py** - 添加返回类型
6. **utils/** - 统一函数签名

---

## 验收标准

- [ ] `mypy fincore/attribution/` 无错误
- [ ] `mypy fincore/plugin/` 无错误
- [ ] `mypy fincore/` 错误数 < 30
- [ ] `pytest tests/` 全部通过
- [ ] `ruff check fincore/` 无警告

---

## 预期结果

修复后:
- mypy 错误: 51 → < 10
- 代码类型安全性提升
- 更好的 IDE 自动补全支持

---

**分支**: `feature/0030-type-improvements`
**状态**: 📋 待开始

**预计时间**: 2-3 小时
