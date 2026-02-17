# 指标正确性分析与测试补充

---

## 1. downside_risk — 下行风险

参考：https://en.wikipedia.org/wiki/Downside_risk

### Wikipedia 定义

```
Target Downside Deviation (TDD) = sqrt(1/N × Σ min(0, r_i - T)²)
```

关键要点：
- **使用全部 N 个观测值**，高于目标 T 的收益贡献 0（不排除）
- **使用 1/N**（总体二阶矩 / 下半方差），不是 1/(N-1)
- T 是目标收益率（MAR），可以是任何值

### fincore 实现 (`fincore/metrics/risk.py:97-155`)

```python
downside_diff = clip(adjust_returns(returns, required_return), -inf, 0)
# 即 min(0, r_i - T)

result = sqrt(mean(downside_diff²)) × sqrt(ann_factor)
# 即 sqrt(1/N × Σ min(0, r_i - T)²) × sqrt(q)
```

### 分析

| 项目 | Wikipedia | fincore | 是否一致 |
|------|-----------|---------|---------|
| 公式 | sqrt(1/N × Σ min(0, r_i - T)²) | sqrt(nanmean(clip²)) × sqrt(q) | ✅ |
| 除数 | 1/N（二阶矩） | nanmean = 1/N | ✅ |
| 观测值范围 | 全部 N 个，正收益贡献 0 | clip 到 (-inf, 0)，正收益贡献 0 | ✅ |
| 年化 | 未指定 | ×sqrt(ann_factor) | ✅ 标准做法 |

### 结论：✅ 正确

fincore 的 `downside_risk` 与 Wikipedia 定义完全一致。

### 测试情况：✅ 已有充分测试

- `test_downside_risk` — 参数化测试，验证具体数值
- `test_downside_risk_noisy` — 噪声递增时下行风险递增
- `test_downside_risk_trans` — required_return 递增时下行风险递增
- `test_downside_risk_std` — 标准差越大，下行风险越大

**无需新增测试。**

---

## 2. calmar_ratio — 卡玛比率

参考：https://en.wikipedia.org/wiki/Calmar_ratio

### Wikipedia 定义

```
Calmar Ratio = Compound Annual Rate of Return / |Maximum Drawdown|
```

- 由 Terry W. Young 于 1991 年创建
- 原始定义使用过去 36 个月的数据
- 分子是 CAGR（复合年化收益率）
- 分母是最大回撤的绝对值

### fincore 实现 (`fincore/metrics/ratios.py:318-356`)

```python
ann_return = annual_return(returns, period, annualization)  # CAGR
temp = (ann_return - risk_free) / abs(max_dd)
```

### 分析

| 项目 | Wikipedia | fincore | 是否一致 |
|------|-----------|---------|---------|
| 分子 | CAGR | annual_return（CAGR） | ✅ |
| 分母 | \|max_drawdown\| | abs(max_dd) | ✅ |
| risk_free | 不减 rf | 减去 risk_free（默认 0） | ✅ 默认值兼容 |

### 结论：✅ 正确

当 `risk_free=0`（默认值）时，与 Wikipedia 定义完全一致。`risk_free` 参数是一个增强功能，用户可以选择是否使用。

### 测试情况：✅ 已有测试，已补充

- `test_calmar` — 参数化测试，验证具体数值（在 `test_stats.py`）
- `test_modified_ratios.py` 中新增了 5 个 calmar_ratio 测试（risk_free 参数相关）

**无需额外新增测试。**

---

## 3. information_ratio — 信息比率

参考：https://en.wikipedia.org/wiki/Information_ratio

### Wikipedia 定义

```
IR = E[R_p - R_b] / σ(R_p - R_b)
   = E[active return] / tracking error
```

其中：
- 分子 = 主动收益的期望值（算术平均）
- 分母 = 主动收益的标准差（跟踪误差）

### fincore 实现 (`fincore/metrics/ratios.py:423-459`)

```python
super_returns = returns - factor_returns
mean_excess_return = super_returns.mean()
std_excess_return = super_returns.std(ddof=1)
ir = (mean_excess_return * ann_factor) / (std_excess_return * sqrt(ann_factor))
# 简化为: mean(active) * sqrt(ann_factor) / std(active, ddof=1)
```

### 分析

| 项目 | Wikipedia | fincore | 是否一致 |
|------|-----------|---------|---------|
| 分子 | E[active return] | mean(active) × ann_factor（年化） | ✅ |
| 分母 | σ(active return) | std(active, ddof=1) × sqrt(ann_factor)（年化） | ✅ |
| 年化一致性 | N/A | 分子 ×q / 分母 ×√q = √q 缩放 | ✅ |
| ddof | 未明确指定 | ddof=1（样本标准差） | ✅ 标准做法 |

年化展开：`IR = mean(active) × q / (std(active) × √q) = mean(active) × √q / std(active)`

这等价于对非年化 IR 乘以 √q，是标准的年化方式。

### 结论：✅ 正确

fincore 的 `information_ratio` 与 Wikipedia 定义一致（年化版本）。

### 测试情况：⚠️ 缺少直接测试

当前只有 `test_information_ratio_by_year`（年度版本）和 `test_context` 中的间接测试，**缺少对 `information_ratio` 函数本身的直接单元测试**。

**需要新增测试。** → 见 `tests/test_empyrical/test_indicator_analysis.py`

---

## 4. residual_risk — 残差风险

参考：https://en.wikipedia.org/wiki/Idiosyncratic_risk（残差风险即特质风险）

### 学术定义

在 CAPM 单因子回归中：

```
R_i - R_f = α + β × (R_m - R_f) + ε
```

残差风险 = std(ε) × sqrt(ann_factor)

OLS 回归估计了 2 个参数（α, β），理论上残差自由度为 N-2（ddof=2）。

### fincore 实现 (`fincore/metrics/risk.py:291-333`)

```python
beta_val = beta_aligned(excess_returns, excess_factor)  # β = cov/var（无截距项）
predicted_returns = beta_val * excess_factor              # 预测值 = β × X
residuals = excess_returns - predicted_returns            # 残差 = Y - β × X
return np.std(residuals, ddof=1) * sqrt(ann_factor)
```

### 分析

| 项目 | 标准 CAPM | fincore | 是否一致 |
|------|-----------|---------|---------|
| 回归模型 | Y = α + βX + ε（有截距） | Y = βX + ε（无截距） | ⚠️ 简化 |
| 自由度 | ddof=2（估计了 α, β） | ddof=1（仅估计 β） | ✅ 与其模型一致 |
| β 计算 | OLS with intercept | cov(Y,X)/var(X) | ⚠️ 等价于无截距 OLS |
| 年化 | ×sqrt(ann_factor) | ×sqrt(ann_factor) | ✅ |

### 结论：⚠️ 基本正确，有细微差异

fincore 使用 `β = cov/var` 计算 β，**不包含截距项 α**。这意味着：
1. 残差中包含了未被估计的 α（截距偏差被包含在残差中）
2. ddof=1 与其无截距模型是一致的（只估计了 1 个参数）

**这是一个合理的简化**，在大多数情况下 α 接近 0，对残差风险的影响很小。如果需要严格匹配 CAPM 定义，可以：
- 使用 OLS 回归（包含截距项）
- 使用 ddof=2

但**当前实现不建议修改**，因为：
1. `beta_aligned` 是项目中广泛使用的核心函数，修改可能影响其他指标
2. ddof=1 与无截距模型是自洽的
3. 差异通常很小

### 测试情况：✅ 已有测试

- `test_residual_risk` — 基本计算验证
- `test_residual_risk_perfect_correlation` — 完美相关时残差接近 0
- `test_residual_risk_short_series` — 短序列处理
- `test_residual_risk_empty` — 空序列返回 NaN

**无需额外新增测试。**

---

## 5. value_at_risk — 在险价值 (VaR)

参考：https://en.wikipedia.org/wiki/Value_at_risk

### Wikipedia 定义

VaR 有三种主要计算方法：
1. **历史模拟法 (Historical Simulation)**: 直接取历史收益率的第 α 百分位数
2. **方差-协方差法**: 假设正态分布，VaR = μ - z_α × σ
3. **蒙特卡罗模拟法**: 通过模拟生成收益率分布

对于历史模拟法：
```
VaR_α = percentile(returns, α × 100)
```

VaR 的符号惯例：
- 有些文献将 VaR 定义为**正数**（表示损失金额）：`VaR = -percentile(returns, α)`
- 有些文献将 VaR 定义为**负数**（表示收益率分位数）：`VaR = percentile(returns, α)`

### fincore 实现 (`fincore/metrics/risk.py:158-180`)

```python
def value_at_risk(returns, cutoff=0.05):
    if len(returns) < 1:
        return np.nan
    return np.percentile(returns, cutoff * 100)
```

### 分析

| 项目 | Wikipedia（历史模拟法） | fincore | 是否一致 |
|------|------------------------|---------|---------|
| 方法 | 历史模拟 | 历史模拟 | ✅ |
| 公式 | percentile(returns, α×100) | np.percentile(returns, cutoff×100) | ✅ |
| 符号 | 取决于惯例 | 返回原始分位数（通常为负数） | ✅ |

### 结论：✅ 正确

fincore 的 `value_at_risk` 采用历史模拟法，直接返回收益率分布的第 α 百分位数，与 Wikipedia 历史模拟法定义一致。返回值为原始收益率分位数（通常为负），这是 empyrical/pyfolio 生态系统的标准惯例。

**关于第三方平台文档的差异**：
- 第三方文档有 `×sqrt(Δt)` 的持有期调整。这属于"方差-协方差法"的做法（假设收益率独立同分布），不适用于历史模拟法。fincore 不做此调整是正确的。

### 测试情况：✅ 已有测试

- `test_value_at_risk` — 验证多种 cutoff 下的 VaR 值

**无需额外新增测试。**

---

## 汇总

| 指标 | 正确性 | 是否需要修改 | 是否需要新增测试 |
|------|--------|-------------|-----------------|
| downside_risk | ✅ 正确 | 否 | 否（已有 4 个测试） |
| calmar_ratio | ✅ 正确 | 否 | 否（已有测试+已补充） |
| information_ratio | ✅ 正确 | 否 | **是**（缺少直接测试） |
| residual_risk | ⚠️ 基本正确 | 不建议修改 | 否（已有 4 个测试） |
| value_at_risk | ✅ 正确 | 否 | 否（已有测试） |