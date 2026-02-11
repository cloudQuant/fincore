# Sortino Ratio 分析：fincore 实现 vs Wikipedia 定义

参考文档：https://en.wikipedia.org/wiki/Sortino_ratio

---

## 1. Wikipedia 标准定义

### 公式

```
Sortino Ratio = (R_p - T) / TDD
```

其中：
- **R_p** = 投资组合的（算术）平均收益率
- **T** = 目标收益率 / 最低可接受收益率 (MAR, Minimum Acceptable Return)
- **TDD** = 目标下行偏差 (Target Downside Deviation)

### TDD 的计算

```
TDD = sqrt( 1/N × Σ [min(0, r_i - T)]² )
```

关键要点：
1. **使用全部 N 个观测值**：高于 T 的收益贡献 0，不被排除
2. **使用 1/N（总体二阶矩）**，而非 1/(N-1)（样本标准差）— 这是一个半方差（semi-variance），不是样本统计量
3. **T 可以是任何目标收益率**，不一定是无风险利率

### 年化版本

```
Annualized Sortino = (mean(r_i - T) × q) / (sqrt(1/N × Σ [min(0, r_i - T)]²) × sqrt(q))
                   = mean(r_i - T) × sqrt(q) / sqrt(1/N × Σ [min(0, r_i - T)]²)
```

其中 q 是年化因子（日频 = 252）。

---

## 2. fincore 当前实现

### sortino_ratio (`fincore/metrics/ratios.py:106-168`)

```python
adj_returns = adjust_returns(returns, required_return)   # r_i - required_return
ann_factor = annualization_factor(period, annualization)

average_annual_return = nanmean(adj_returns, axis=0) * ann_factor
annualized_downside_risk = downside_risk(returns, required_return, period, annualization)

return average_annual_return / annualized_downside_risk
```

### downside_risk (`fincore/metrics/risk.py:97-155`)

```python
downside_diff = clip(adjust_returns(returns, required_return), -inf, 0)
# 即 min(0, r_i - required_return)

result = sqrt(mean(downside_diff²)) × sqrt(ann_factor)
# 即 sqrt(1/N × Σ [min(0, r_i - T)]²) × sqrt(q)
```

### fincore 的 Sortino 完整展开

```
Sortino = [mean(r_i - T) × q] / [sqrt(1/N × Σ min(0, r_i - T)²) × sqrt(q)]
        = mean(r_i - T) × sqrt(q) / sqrt(1/N × Σ min(0, r_i - T)²)
```

---

## 3. 逐项对比

| 项目 | Wikipedia | fincore | 是否一致 |
|------|-----------|---------|---------|
| **分子** | R_p - T（算术平均收益 - 目标收益） | mean(r_i - T) × q（年化） | ✅ 一致（年化版本） |
| **分母 TDD 公式** | sqrt(1/N × Σ min(0, r_i - T)²) | sqrt(mean(min(0, r_i - T)²)) × sqrt(q) | ✅ 一致（年化版本） |
| **分母除数** | 1/N（全部观测值） | nanmean = 1/N | ✅ 一致 |
| **观测值范围** | 全部 N 个，正收益贡献 0 | clip 到 (-inf, 0)，正收益贡献 0 | ✅ 一致 |
| **目标收益率 T** | 任何 MAR / 目标收益率 | required_return（默认 0） | ✅ 一致 |
| **年化方式** | 分子 ×q，分母 ×√q | 分子 ×q，分母 ×√q | ✅ 一致 |

---

## 4. 结论：fincore 的 Sortino Ratio 与 Wikipedia 定义一致

**fincore 的当前实现是正确的**，与 Wikipedia/学术标准定义完全一致：

1. ✅ 分子使用算术平均超额收益的年化值
2. ✅ 分母使用 1/N（不是 1/(N-1)）
3. ✅ 使用全部观测值，正收益贡献 0
4. ✅ 年化方式正确（分子 ×q，分母 ×√q）

---

## 5. 与第三方平台文档的差异分析

需求.md 中记录的差异是 fincore 与"第三方平台文档"之间的，而非 fincore 与学术标准之间的。逐一分析：

### 差异 1：分子 CAGR vs 算术平均年化

| | 第三方文档 | fincore | Wikipedia |
|--|-----------|---------|-----------|
| 分子 | CAGR - rf | mean(r_i - T) × q | R_p - T |

- Wikipedia 使用**算术平均**收益率（R_p = average realized return）
- **fincore 的做法与 Wikipedia 一致**
- 第三方文档使用 CAGR（几何平均）是该平台的自定义做法，与 Sortino 原始定义不同

### 差异 2：分母 1/T vs 1/(T-1)

| | 第三方文档 | fincore | Wikipedia |
|--|-----------|---------|-----------|
| 除数 | 1/(T-1) | 1/N (nanmean) | 1/N |

- Wikipedia 明确使用 1/N（二阶矩），**不是**样本标准差
- **fincore 的做法与 Wikipedia 一致**
- 第三方文档的 1/(T-1) 将 TDD 当作样本标准差处理，这不是标准做法

### 差异 3：分母额外 ×2 因子

| | 第三方文档 | fincore | Wikipedia |
|--|-----------|---------|-----------|
| 公式 | sqrt(q × **2** × LPM2) | sqrt(mean × q) | sqrt(1/N × Σ) |

- Wikipedia 的 TDD 公式中**没有 ×2 因子**
- **fincore 没有 ×2 因子是正确的**
- 第三方文档的 ×2 因子可能是该平台对 LPM 的特定定义方式

### 差异 4：默认阈值 rf vs required_return

| | 第三方文档 | fincore | Wikipedia |
|--|-----------|---------|-----------|
| 阈值 | rf（无风险利率，默认 3%） | required_return（默认 0） | T（任何目标收益率） |

- Wikipedia 定义 T 为任何"目标收益率"，可以是 rf 也可以是其他值
- fincore 默认 0 是常见实践（同 Sharpe ratio 的默认 rf=0）
- **无需修改**，用户可以通过参数自行设置

---

## 6. 修改建议

### 不需要修改核心算法

fincore 的 Sortino ratio 实现与 Wikipedia/学术标准定义**完全一致**，无需修改核心算法。

### 可选改进（非必须）

1. **添加 Wikipedia 参考链接到 docstring**：
   ```python
   Reference
   ---------
   https://en.wikipedia.org/wiki/Sortino_ratio
   ```

2. **参数名称说明**：可在 docstring 中补充说明 `required_return` 对应 Wikipedia 中的 T（目标收益率 / MAR），以便用户理解参数含义。

3. **如需兼容第三方平台**：如果业务上需要与第三方平台的计算方式保持一致，可以考虑新增一个参数（如 `method='wikipedia'`）来支持不同的计算方式，但这会增加复杂度，不推荐。
