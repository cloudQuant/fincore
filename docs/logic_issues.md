# Empyrical 源码逻辑问题分析

> 基于当前仓库状态，所有测试（约 1150 个）均为通过；下面列出的是在此基础上发现/确认的实现/设计层面的潜在问题和历史 bug 记录。并不保证穷尽。

## 1. 背景与范围

- 分析范围：
  - 核心库代码：`empyrical/`（重点是 `stats.py`、`perf_attrib.py`）
  - 部分测试代码：`tests/`（仅用于辅助理解实现是否被合理覆盖）
- 代码质量现状：
  - `pytest`：全量测试通过。
  - `flake8`（库代码）：仅有两处 `F841`（局部变量赋值未使用），其它为风格类问题。

下文按“已修复问题”和“潜在设计问题/死代码”两类来说明。

---

## 2. 已修复的逻辑问题

### 2.1 `perf_attrib` 未保留输入收益序列的时间索引频率

**位置：**

- 文件：`empyrical/perf_attrib.py`
- 函数：`perf_attrib`
- 关键代码（当前版本）：

```python
risk_exposures_portfolio = compute_exposures(positions,
                                             factor_loadings)

# Preserve the exact DatetimeIndex of the input returns (including freq)
risk_exposures_portfolio.index = returns.index

perf_attrib_by_factor = risk_exposures_portfolio.multiply(factor_returns)
common_returns = perf_attrib_by_factor.sum(axis='columns')

...
perf_attribution = pd.concat([perf_attrib_by_factor, returns_df],
                             axis='columns')

# Ensure perf_attribution index exactly matches the input returns index
perf_attribution.index = returns.index

return risk_exposures_portfolio, perf_attribution
```

**问题表现：**

- 旧实现中：
  - `risk_exposures_portfolio` 的索引来自 `compute_exposures` 的 `groupby(level='dt')` 结果；
  - `perf_attribution` 的索引来自 `concat`，未强制对齐 `returns.index`。
- 在 `tests/test_perf_attrib.py::test_perf_attrib_simple` 中，当用 `DataFrame.equals` 比较期望输出和实际输出时：
  - 数据、列名都完全一致；
  - 但 `DatetimeIndex.freq` 元数据不一致：
    - 期望：`freq='D'`
    - 实际：`freq=None`
  - 这导致 `equals()` 返回 `False`，属于典型“逻辑上相等但元数据不一致”的 bug。

**影响：**

- 所有依赖 `empyrical.perf_attrib` 的下游逻辑，如果同时依赖索引频率（例如使用 `freq` 做进一步重采样或校验），可能会得到不一致的行为。
- 单元测试在最初版本中由于少了断言，未能暴露这一问题。

**修复思路：**

- 明确约定：`perf_attrib` 的所有时间轴输出应与输入 `returns.index` 完全一致，包括：
  - 时间点；
  - `name`；
  - `freq` 等元数据。
- 修复方式：
  - 计算完 `risk_exposures_portfolio` 后，显式设置 `risk_exposures_portfolio.index = returns.index`。
  - 拼出 `perf_attribution` 后，显式设置 `perf_attribution.index = returns.index`。

**结论：**

- 修复后的行为更符合文档“以 datetime 为 index”的直观预期。
- 现有测试全部通过，说明改动对既有业务逻辑是后向兼容的（只是修正元数据）。

---

## 3. 潜在设计问题 / 死代码

### 3.1 `m_squared` 中 `ann_factor_return` 未使用（死代码）

**位置：**

- 文件：`empyrical/stats.py`
- 函数：`m_squared`（行号约 1230 附近）

**相关代码：**

```python
# Calculate annualized returns and volatilities
ann_return = annual_return(returns, period=period, annualization=annualization)
ann_vol = annual_volatility(returns, period=period, annualization=annualization)
ann_factor_return = annual_return(factor_returns, period=period, annualization=annualization)
ann_factor_vol = annual_volatility(factor_returns, period=period, annualization=annualization)

# Handle division by zero or negative volatility
...
    # M² = (Rp - Rf) * (σb / σp) + Rf
    excess_return = ann_return - risk_free
    risk_ratio = ann_factor_vol / ann_vol
    out = excess_return * risk_ratio + risk_free
```

**现象：**

- `ann_factor_return` 被计算出来但 **完全未参与后续计算**。
- flake8 报告：`F841 local variable 'ann_factor_return' is assigned to but never used`。

**分析：**

- 理论上，M²（Modigliani–Modigliani measure）的常见定义是：
  - \( M^2 = R_f + (R_p - R_f) \cdot \frac{\sigma_M}{\sigma_p} \)
  - 即用**基准波动率** `σ_M` 对组合 `R_p` 做风险匹配，再加回 `R_f`。
- 当前实现完全符合上述公式，只使用了：
  - `ann_return`（组合年化收益）；
  - `ann_vol`（组合年化波动）；
  - `ann_factor_vol`（基准年化波动）；
  - `risk_free`（无风险收益）。
- 因此：
  - 从 **数值上** 看，M² 的计算公式是合理的；
  - `ann_factor_return` 只是未使用的中间变量，属于轻微的“实现噪音”/死代码。

**可能的设计意图（推测）：**

- 作者可能曾打算提供“相对于基准收益的 M² 差值”等指标，需要 `ann_factor_return` 做进一步比较；
- 但最终实现只保留了标准 M²，相关部分被遗留为死代码。

**风险评估：**

- 这是一个**代码清洁度问题**，不是行为 bug：
  - 删除 `ann_factor_return` 不会改变任何对外行为；
  - 保留也不会影响当前结果，只是增加读者困惑。

---

### 3.2 `beta_aligned` / `beta` 中 `risk_free` 参数未真正生效（设计与实现不一致）

**位置：**

- 文件：`empyrical/stats.py`
- 函数：
  - `beta`（对外接口）
  - `beta_aligned`（内部核心实现，行号约 5145+）

**相关代码（节选）：**

```python
def beta(returns, factor_returns, risk_free=0.0, out=None):
    ...
    if not (isinstance(returns, np.ndarray) and
            isinstance(factor_returns, np.ndarray)):
        returns, factor_returns = _aligned_series(returns, factor_returns)

    return beta_aligned(
        returns,
        factor_returns,
        risk_free=risk_free,
        out=out,
    )
```

```python
def beta_aligned(returns, factor_returns, risk_free=0.0, out=None):
    """
    ...
    risk_free : int, float, optional
        Constant risk-free return throughout the period. For example, the
        interest rate on a three-month US treasury bill.
    ...
    """
    _risk_free = risk_free
    # Cache these as locals since we're going to call them multiple times.
    nan = np.nan
    isnan = np.isnan

    ...
    # 后续计算中，从未使用 risk_free 或 _risk_free
    # 只基于 returns 与 factor_returns 做协方差/方差比值，得到标准 beta
```

**现象：**

- `beta` 对外暴露了 `risk_free` 参数，并将其透传给 `beta_aligned`。
- `beta_aligned` 中：
  - `_risk_free = risk_free` 被赋值但完全未使用（flake8: `F841`）。
  - 实际计算完全是标准的 \(\beta = \frac{\mathrm{Cov}(R_p, R_m)}{\mathrm{Var}(R_m)}\)，**未减去任何无风险利率**。

**从 API 设计角度看：**

- 文档写明 `risk_free` 是 “Constant risk-free return throughout the period”；
- 在经典 CAPM 框架中，若考虑无风险利率，beta 更一般形式为对 `(R_p - R_f)` 与 `(R_m - R_f)` 的回归/协方差。
- 当前实现 **忽略 `risk_free`**，导致：
  - 用户传入非零 `risk_free` 时不会产生任何效果；
  - 文档与实际行为明显不一致。

**风险评估：**

- 在默认 `risk_free=0.0` 场景下，行为与常规“简化 beta”一致，因此现有测试全部通过。
- 对于依赖“有无风险收益调整 beta”的用户，这是一个**真实的设计缺陷**：
  - 调用签名暗示支持此功能；
  - 但实现实际上忽略参数。

**建议（仅供参考，不是现有改动）：**

- 如果将来要修复，可选两种策略：
  - 修改实现：在 `beta_aligned` 中对 `(returns - risk_free)` 和 `(factor_returns - risk_free)` 做回归/协方差。
  - 修改 API 文档：明确说明当前 beta 是相对于 0 无风险利率的“普通 beta”，`risk_free` 参数废弃或保留兼容但标记为 deprecated。

目前仓库中该问题**仅被记录和分析，未做行为层面的变更**。

---

## 4. 测试代码相关的逻辑/设计问题（简要）

虽然这里主要关注库代码，这里附带记录一条与测试相关，但影响真实行为验证的设计问题：

### 4.1 `test_perf_attrib_simple` 初始版本未对 DataFrame.equals 结果做断言

**位置：**

- 文件：`tests/test_perf_attrib.py`
- 用例：`PerfAttribTestCase.test_perf_attrib_simple`

**问题：**

- 初始版本中使用：

  ```python
  expected_perf_attrib_output.equals(perf_attrib_output)
  expected_exposures_portfolio.equals(exposures_portfolio)
  ```

  但**没有把结果放进断言**（例如 `assertTrue`），导致即便比较返回 `False`，测试也不会失败。

- 这直接掩盖了上文 2.1 所述的 `freq` 元数据问题。

**当前状态：**

- 已改为真正的断言，并在比较 `perf_attrib_output` 时做了适度的 `round(10)`，以规避极微小的浮点误差：

  ```python
  self.assertTrue(
      expected_perf_attrib_output.round(10).equals(
          perf_attrib_output.round(10)
      )
  )
  self.assertTrue(expected_exposures_portfolio.equals(exposures_portfolio))
  ```

---

## 5. 总结与后续建议

- 目前确认的逻辑层面问题主要集中在：
  - `perf_attrib` 的索引频率元数据未对齐（已修复）；
  - `beta_aligned` / `beta` 中 `risk_free` 参数未真正参与计算（设计与文档不一致）；
  - `m_squared` 中存在未使用变量 `ann_factor_return`，属于死代码而非行为 bug。
- 测试侧的历史问题（如 `DataFrame.equals` 未断言）也已经修复，并通过了全量 `pytest` 验证。

**建议：**

1. 若未来要增强库的金融含义一致性，优先考虑：
   - 设计并实现真正考虑无风险利率的 `beta` 版本；
   - 或在文档中明确当前 beta 的定义，避免用户误解。
2. 对其它复杂指标（例如一些高级风险比率、回归指标），可以采用类似这次 `perf_attrib` 的方式：
   - 用更“强”的测试（断言 + 调试输出）去对齐实现与金融定义；
   - 允许测试短暂变红，查清差异后再做修复。

这份文档可以作为项目中关于“逻辑缺陷与设计注意事项”的补充说明，帮助未来维护者快速理解当前实现的“坑”与历史决策。
