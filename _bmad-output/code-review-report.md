# fincore 代码评审报告

**评审日期:** 2026-03-10  
**评审类型:** BMAD 对抗式代码评审 (Adversarial Code Review)  
**范围:** fincore 应用源代码 (`fincore/`)，排除 `_bmad/`、`.cursor/`、`.claude/`、测试文件  

---

## 概览

| 严重程度 | 数量 |
|----------|------|
| 🔴 HIGH   | 1    |
| 🟡 MEDIUM | 6    |
| 🟢 LOW    | 5    |

---

## 🔴 CRITICAL / HIGH 问题

### 1. Path Traversal — `to_html(path=...)` 路径遍历风险

**文件:** `fincore/core/context.py:330-334`  
**严重程度:** HIGH  
**类别:** Security

**描述:** `to_html(path=...)` 直接使用传入的 `path` 进行 `os.makedirs()` 和 `open()`。用户可控制的路径（如 `../../../etc/passwd` 或 `path/to/../../sensitive`）可能导致写入到预期以外的目录。

**建议修复:**

```python
def to_html(self, path: str | None = None) -> str:
    ...
    if path:
        path = os.path.abspath(os.path.normpath(path))
        # 可选：校验路径在允许的根目录下
        pardir = os.path.dirname(path)
        if pardir:
            os.makedirs(pardir, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)
```

可选：限制写入到指定根目录，若路径逃逸则 `raise`。

---

## 🟡 MEDIUM 问题

### 2. Path Traversal — HTML 报告生成

**文件:** `fincore/report/render_html.py:243`  
**严重程度:** MEDIUM  
**类别:** Security

**描述:** `generate_html` 接受 `output` 参数并直接写入，未校验路径，可能覆盖任意文件。

**建议修复:**

- 写入前对 `output` 做规范化：`os.path.abspath(os.path.normpath(output))`
- 可选：校验解析后的路径在允许的根目录下，否则抛出异常

---

### 3. ZeroDivisionError — `detect_intraday`

**文件:** `fincore/utils/common_utils.py:554-556`  
**严重程度:** MEDIUM  
**类别:** Error handling

**描述:** 当 `transactions` 为空或无可交易标的时，`txn_count` 为 0，`daily_pos.count(axis=1).sum() / txn_count` 会触发 `ZeroDivisionError`。

**建议修复:**

```python
txn_count = daily_txn.groupby(level=0).symbol.nunique().sum()
if txn_count == 0:
    return False  # 无交易 → 非日内策略
return daily_pos.count(axis=1).sum() / txn_count < threshold
```

---

### 4. 不必要的 Series 拷贝 — `_compute_sortino`

**文件:** `fincore/core/engine.py:143-144`  
**严重程度:** MEDIUM  
**类别:** Performance

**描述:** `downside = ret.copy()` 会创建完整拷贝，然后原地修改。对长序列会带来不必要的内存与计算开销。

**建议修复:**

```python
downside = np.minimum(ret.values, 0.0)
rolling_downside_std = pd.Series(downside, index=ret.index).rolling(w, min_periods=w).std(ddof=1)
```

---

### 5. AnalysisContext 创建失败被静默忽略

**文件:** `fincore/empyrical.py:204-215`  
**严重程度:** MEDIUM  
**类别:** Error handling

**描述:** 若 `AnalysisContext` 创建失败，异常被捕获并只记录日志，`self._ctx` 保持为 `None`。后续依赖 `self._ctx` 的调用会报出难以理解的错误。

**建议修复:**

- 方案 A：重新抛出异常，让调用方看到原始错误
- 方案 B：在使用 `_ctx` 前显式检查并给出清晰错误：

```python
if self._ctx is None:
    raise RuntimeError(
        "AnalysisContext could not be created. Check that returns are valid."
    )
```

---

### 6. KeyError — BrinsonAttribution `_apply_sector_mapping`

**文件:** `fincore/attribution/brinson.py:370-371`  
**严重程度:** MEDIUM  
**类别:** Error handling

**描述:** `sector_df = df[assets]` 假定 `sector_mapping` 中所有 `assets` 都存在于 `df.columns`。缺失资产会直接抛出 `KeyError`，且无明确说明。

**建议修复:**

```python
for sector, assets in self.sector_mapping.items():
    missing = set(assets) - set(df.columns)
    if missing:
        raise ValueError(
            f"Sector '{sector}' references assets not in DataFrame: {missing}"
        )
    sector_df = df[assets]
```

---

### 7. RollingEngine 未校验 `window`

**文件:** `fincore/core/engine.py:59-62`  
**严重程度:** MEDIUM  
**类别:** Error handling / Validation

**描述:** 未校验 `window > 0`，非法或极大窗口可能造成意外行为。

**建议修复:**

```python
def __init__(self, ...):
    ...
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    self._window = window
```

---

## 🟢 LOW 问题

### 8. `get_top_drawdowns` 循环中错误变量

**文件:** `fincore/metrics/drawdown.py:336`  
**严重程度:** LOW  
**类别:** Edge case / Logic

**描述:** 条件使用 `len(returns) == 0`，但 `returns` 在循环中不变。更合理的是在 `underwater` 被清空后终止。

**建议修复:**

```python
if len(underwater) == 0:
    break
```

可移除或替换原有的 `len(returns) == 0` 判断。

---

### 9. 魔法数字 — 月份天数近似

**文件:** `fincore/metrics/drawdown.py:504, 590`  
**严重程度:** LOW  
**类别:** Code quality

**描述:** 使用 `days / 30` 将天数换算为月份，实际月份长度各异，可能误导使用者。

**建议修复:**

```python
# 方案 1：命名常量
APPROX_DAYS_PER_MONTH = 30.44  # 365.25 / 12
return days / APPROX_DAYS_PER_MONTH

# 方案 2：注释说明
# 返回近似月份数（假定约 30 天/月）
```

---

### 10. `standardize_data` 除零 / NaN 风险

**文件:** `fincore/utils/common_utils.py:528`  
**严重程度:** LOW  
**类别:** Error handling

**描述:** `(x - np.mean(x)) / np.std(x)` 在 `np.std(x) == 0` 或空数组时会产生 `inf`/`nan`。

**建议修复:**

```python
def standardize_data(x):
    x = np.asarray(x)
    if len(x) == 0:
        return x
    std = np.std(x)
    if std == 0 or not np.isfinite(std):
        return np.zeros_like(x)  # 或返回原数组 / 抛出异常
    return (x - np.mean(x)) / std
```

---

### 11. HTML 报告潜在 XSS — `add_stats_table`

**文件:** `fincore/viz/html_backend.py:85`  
**严重程度:** LOW  
**类别:** Security

**描述:** 非浮点型 `val` 使用 `formatted = str(val)` 直接插入 HTML，未转义。若 stats 包含用户可控字符串，存在 XSS 风险。

**建议修复:**

```python
formatted = _html.escape(str(val))
```

---

### 12. `Information` 指标缺失 `information_ratio` 处理

**文件:** `fincore/metrics/yearly.py`  
**严重程度:** LOW  
**类别:** Edge case

**描述:** `information_ratios` 在 benchmark 对齐后可能产生空组，`groupby(...).apply()` 结果可能含 NaN，需要文档说明或显式处理。

---

## 后续修复建议

1. **优先处理 HIGH / MEDIUM：** 先修复路径遍历和除零等问题。
2. **统一输入校验：** 利用 `fincore/validation.py` 的 `validate_*` 在关键入口做校验。
3. **补充测试：** 为空输入、单元素序列、NaN、对齐异常等添加边界测试。
4. **安全审计：** 对所有接受路径和用户可控字符串的接口做一次安全排查。

---

## 评审依据

- BMAD 代码评审流程：`_bmad/bmm/workflows/4-implementation/code-review/`
- 项目上下文：`_bmad-output/project-context.md`
- 评审方式：对抗式、以发现具体问题为主

---

_本报告用于指导后续修复，建议逐项处理并按需补充测试与文档。_
