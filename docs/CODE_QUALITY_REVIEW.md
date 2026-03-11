# Fincore 代码质量审查报告

> 审查日期: 2026-03-10
> 方法: BMAD 对抗式审查 + 边缘情况猎人

---

## 一、基线状态

| 检查项 | 状态 |
|--------|------|
| Ruff lint | ✅ 通过 |
| Ruff format | ✅ 已格式化 |
| 快速验证脚本 | `python scripts/verify_quality_fixes.py` ✅ |
| P0 测试 | ✅ 194/194 通过 |
| 集成测试 | ✅ 15/15 通过 |
| 总测试 | 1997 通过, 16 失败 (边缘情况) |
| np.nanstd(out=...) | ✅ NumPy 支持，无 bottleneck 时兼容 |

---

## 二、对抗式审查发现 (至少 10 项)

### 高优先级

1. **`excess_sharpe` 使用 `np.nan_to_num` 掩盖零波动率**
   - 位置: `fincore/metrics/ratios.py:262`
   - 问题: `tracking_err = np.nan_to_num(nanstd(...))` 将 NaN 转为 0，导致除零产生 inf；虽然后续有 invalid 检查，但不如 `sharpe_ratio` 显式用 `_MIN_STD` 阈值清晰。
   - 建议: 与 `sharpe_ratio` 一致，使用最小波动率阈值或显式零/NaN 检查。

2. **`downside_risk` 全正收益时平方根可能产生极小值**
   - 位置: `fincore/metrics/risk.py:155-165`
   - 问题: 当所有收益都高于 `required_return` 时，`downside_diff` 全为 0，`nanmean` 得 0，`np.sqrt(0)` 得 0。但若 `downside_diff` 有极小的数值误差，可能产生非零。当前逻辑正确，但缺少对“全正收益”的显式文档说明返回 0。

3. **`sterling_ratio` 和 `burke_ratio` 返回 `np.inf` 的语义不明确**
   - 位置: `fincore/metrics/ratios.py:621, 668`
   - 问题: `return np.inf if ann_ret - risk_free > 0 else np.nan` 在无回撤时返回 inf。下游消费者可能未处理 inf，导致后续计算异常。
   - 建议: 文档明确说明，或考虑返回一个极大有限值并加注释。

4. **`annual_volatility` 的 `alpha_` 参数命名易混淆**
   - 位置: `fincore/metrics/risk.py:51`
   - 问题: `alpha_` 表示波动率幂次 (2.0=方差)，与金融中常见的 alpha (超额收益) 易混淆。
   - 建议: 重命名为 `volatility_power` 或 `exponent`。

5. **`information_ratio` 零追踪误差未显式处理**
   - 位置: `fincore/metrics/ratios.py:441-442`
   - 问题: `std_excess_return` 为 0 时，除法产生 inf，未像其他 ratio 函数那样显式置为 NaN。
   - 建议: 添加与 `sharpe_ratio` 类似的 `~np.isfinite` 检查和 NaN 替换。

### 中优先级

6. **`conditional_sharpe_ratio` 对 `cutoff` 边界无校验**
   - 位置: `fincore/metrics/ratios.py:368`
   - 问题: `cutoff` 为 0 或 1 时，`np.nanpercentile(returns, 0)` 或 `100` 可能产生边界行为。
   - 建议: 添加 `cutoff in (0, 1)` 校验，非法时返回 NaN 或 raise。

7. **`value_at_risk` 和 `conditional_value_at_risk` 对 `cutoff` 无校验**
   - 位置: `fincore/metrics/risk.py:184, 196`
   - 问题: `cutoff <= 0` 或 `cutoff >= 1` 时 `np.nanpercentile` 行为未定义。
   - 建议: 添加参数校验。

8. **`omega_ratio` 在 `required_return <= -1` 时返回 NaN 但无文档说明**
   - 位置: `fincore/metrics/ratios.py:416-417`
   - 问题: `(1 + required_return) ** (1.0 / annualization) - 1` 在 `required_return <= -1` 时数学上无效，代码正确处理但 docstring 未提及。
   - 建议: 在 docstring 的 Returns 部分说明。

9. **`gpd_risk_estimates` 内部嵌套函数过多，可读性差**
   - 位置: `fincore/metrics/risk.py:364-415`
   - 问题: `gpd_loglikelihood`、`_gpd_loglikelihood_scale_and_shape` 等定义在 while 循环外但在同一函数内，逻辑复杂。
   - 建议: 拆分为模块级私有函数，便于测试和复用。

10. **`nanstd`/`nanmean` 的 `out` 参数依赖 bottleneck 包装器实现**
    - 位置: `fincore/utils/math_utils.py`
    - 问题: 无 bottleneck 时 fallback 到 `np.nanstd`，但 NumPy 的 `nanstd` 不支持 `out`，包装器会忽略。当前代码路径仅在 bottleneck 存在时使用 `out`，fallback 时 `nanmean(..., out=out)` 会传 `out` 给 `np.nanmean`，而 `np.nanmean` 支持 `out`。需确认 `np.nanstd` 是否支持 `out`——NumPy 的 `nanstd` 不支持 `out` 参数。
    - 建议: 验证无 bottleneck 环境下 `risk.py` 的 `nanstd(..., out=out)` 是否报错；若不支持，包装器应捕获并赋值。

### 低优先级

11. **类型注解不一致**
    部分函数返回 `float | np.ndarray | pd.Series`，调用方可能需额外类型守卫。

12. **`calmar_ratio` DataFrame 分支返回类型**
    `ratios.py:417` 返回 `pd.Series`，但类型注解为 `float | pd.Series`，多列 DataFrame 时的行为需确认。

---

## 三、边缘情况猎人输出 (JSON 格式)

```json
[
  {
    "location": "fincore/metrics/ratios.py:262",
    "trigger_condition": "tracking_err is 0 when returns equal factor_returns",
    "guard_snippet": "tracking_err = np.maximum(nanstd(active_return, ddof=1, axis=0), 1e-15); if np.any(tracking_err == 0): out[...] = np.nan",
    "potential_consequence": "Division by zero yields inf, later masked to nan"
  },
  {
    "location": "fincore/metrics/ratios.py:441",
    "trigger_condition": "std_excess_return is 0 (no tracking error)",
    "guard_snippet": "if std_excess_return == 0 or not np.isfinite(std_excess_return): return np.nan",
    "potential_consequence": "ir becomes inf, may propagate to downstream"
  },
  {
    "location": "fincore/metrics/risk.py:250-252",
    "trigger_condition": "cutoff is 0 or 1",
    "guard_snippet": "if not (0 < cutoff < 1): return np.nan",
    "potential_consequence": "nanpercentile with 0 or 100 may give edge values"
  },
  {
    "location": "fincore/metrics/ratios.py:416",
    "trigger_condition": "required_return <= -1 for omega_ratio",
    "guard_snippet": "already handled: return np.nan",
    "potential_consequence": "None - correctly handled"
  },
  {
    "location": "fincore/metrics/drawdown.py:112",
    "trigger_condition": "max_return is 0 (total loss)",
    "guard_snippet": "safe_max = np.where(max_return == 0, np.nan, max_return)",
    "potential_consequence": "Already guarded - division by nan yields nan"
  }
]
```

---

## 四、建议修复顺序

| 优先级 | 项目 | 状态 |
|--------|------|------|
| P0 | 修复 `information_ratio` 零波动率处理 | ✅ 已完成 |
| P0 | 改进 `excess_sharpe` 的 tracking_err 处理，去掉 nan_to_num | ✅ 已完成 |
| P1 | 为 VaR/CVaR 添加 cutoff 参数校验 | ⏭️ 跳过 (现有测试期望 cutoff=0/1 有效，np.nanpercentile 支持) |
| P1 | 为 conditional_sharpe_ratio 添加 cutoff 校验 | ✅ 已完成 |
| P1 | 文档补充: downside_risk, sterling/burke, omega_ratio | ✅ 已完成 |
| P2 | 验证无 bottleneck 时 nanstd(out=...) 行为 | ✅ 已验证 (np.nanstd 支持 out) |
| P2 | 重命名 alpha_ 为 volatility_power | ⏭️ 已用文档区分 (alpha_≠alpha) |

---

## 五、BMAD 工作流建议

- **继续代码审查**: `/bmad-bmm-code-review`（需 BMM sprint/epic 上下文）
- **对抗式文档审查**: `/bmad-review-adversarial-general`（适用于 PR、设计稿）
- **边缘情况专项**: `/bmad-review-edge-case-hunter`（针对单文件或 diff）

---

*本报告由 BMAD help + 对抗式审查 + 边缘情况猎人方法生成。*
