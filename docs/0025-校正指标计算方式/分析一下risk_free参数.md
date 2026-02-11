分析一下项目中使用到risK_free这个无风险参数的所有函数，如果在一个函数中调用这个函数，是否给传递了risk_free参数。

需要和现有的使用上保持兼容，如果主函数缺少risk_free这个参数，就传递一个risk_free=0的默认情况。

分析一下都哪些函数需要修改，给出一个列表，放到下面的文档下。

---

## 分析结果

### 一、已有 risk_free 参数且正确传递的函数（无需修改）

以下函数已有 `risk_free` 参数，且在内部调用其他函数时正确传递了该参数：

| 文件 | 函数 | 默认值 | 传递情况 |
|------|------|--------|---------|
| `metrics/ratios.py` | `sharpe_ratio` | 0 | 直接使用 `adjust_returns(returns, risk_free)` |
| `metrics/ratios.py` | `adjusted_sharpe_ratio` | 0.0 | 传递给 `sharpe_ratio(returns, risk_free)` |
| `metrics/ratios.py` | `omega_ratio` | 0.0 | 直接使用 `returns - risk_free - return_threshold` |
| `metrics/ratios.py` | `cal_treynor_ratio` | 0.0 | 传递给 `beta_aligned(..., risk_free)` |
| `metrics/ratios.py` | `treynor_ratio` | 0.0 | 传递给 `cal_treynor_ratio(..., risk_free, ...)` |
| `metrics/ratios.py` | `m_squared` | 0.0 | 直接使用 `ann_return - risk_free` |
| `metrics/ratios.py` | `sterling_ratio` | 0.0 | 直接使用 `ann_ret - risk_free` |
| `metrics/ratios.py` | `burke_ratio` | 0.0 | 直接使用 `ann_ret - risk_free` |
| `metrics/ratios.py` | `kappa_three_ratio` | 0.0 | 直接使用 `ann_ret - risk_free` |
| `metrics/alpha_beta.py` | `beta_aligned` | 0.0 | 直接使用 `returns - risk_free` |
| `metrics/alpha_beta.py` | `alpha_aligned` | 0.0 | 传递给 `beta_aligned(..., risk_free)` 和 `adjust_returns(..., risk_free)` |
| `metrics/alpha_beta.py` | `alpha_beta_aligned` | 0.0 | 传递给 `beta_aligned` 和 `alpha_aligned` |
| `metrics/alpha_beta.py` | `beta` | 0.0 | 传递给 `beta_aligned(..., risk_free=risk_free)` |
| `metrics/alpha_beta.py` | `alpha` | 0.0 | 传递给 `alpha_aligned(..., risk_free=risk_free, ...)` |
| `metrics/alpha_beta.py` | `alpha_beta` | 0.0 | 传递给 `alpha_beta_aligned(..., risk_free=risk_free, ...)` |
| `metrics/alpha_beta.py` | `_conditional_alpha_beta` | 0.0 | 直接使用 `returns_clean - risk_free` |
| `metrics/alpha_beta.py` | `up_alpha_beta` | 0.0 | 传递给 `_conditional_alpha_beta(..., risk_free, ...)` |
| `metrics/alpha_beta.py` | `down_alpha_beta` | 0.0 | 传递给 `_conditional_alpha_beta(..., risk_free, ...)` |
| `metrics/alpha_beta.py` | `annual_alpha` | 0.0 | 传递给 `alpha(..., risk_free, ...)` |
| `metrics/alpha_beta.py` | `annual_beta` | 0.0 | 传递给 `beta(..., risk_free, ...)` |
| `metrics/alpha_beta.py` | `alpha_percentile_rank` | 0.0 | 传递给 `alpha(..., risk_free, ...)` |
| `metrics/risk.py` | `residual_risk` | 0.0 | 直接使用 `returns_aligned - risk_free` |
| `metrics/timing.py` | `treynor_mazuy_timing` | 0.0 | 直接使用 `returns_aligned - risk_free` |
| `metrics/timing.py` | `henriksson_merton_timing` | 0.0 | 直接使用 `returns_aligned - risk_free` |
| `metrics/timing.py` | `market_timing_return` | 0.0 | 传递给 `treynor_mazuy_timing(..., risk_free)` |
| `metrics/timing.py` | `cornell_timing` | 0.0 | 直接使用 `returns_clean - risk_free` |
| `metrics/yearly.py` | `sharpe_ratio_by_year` | 0 | 传递给 `sharpe_ratio(..., risk_free=risk_free, ...)` |
| `metrics/rolling.py` | `roll_alpha` | 0.0 | 传递给 `alpha_aligned(..., risk_free, ...)` |
| `metrics/rolling.py` | `roll_beta` | 0.0 | 直接使用 `returns_aligned - risk_free` |
| `metrics/rolling.py` | `roll_alpha_beta` | 0.0 | 传递给 `alpha_beta_aligned(..., risk_free, ...)` |
| `metrics/rolling.py` | `roll_sharpe_ratio` | 0.0 | 直接使用 `returns - risk_free` |

### 二、Empyrical 类中已有 risk_free 的 @_dual_method（无需修改）

| 方法 | 默认值 | 传递目标 |
|------|--------|---------|
| `sterling_ratio` | 0.0 | → `ratios.sterling_ratio(..., risk_free, ...)` |
| `burke_ratio` | 0.0 | → `ratios.burke_ratio(..., risk_free, ...)` |
| `kappa_three_ratio` | 0.0 | → `ratios.kappa_three_ratio(..., risk_free, ...)` |
| `roll_sharpe_ratio` | 0.0 | → `rolling.roll_sharpe_ratio(..., risk_free, ...)` |
| `treynor_ratio` | 0.0 | → `ratios.treynor_ratio(..., risk_free, ...)` |
| `m_squared` | 0.0 | → `ratios.m_squared(..., risk_free, ...)` |
| `residual_risk` | 0.0 | → `risk.residual_risk(..., risk_free, ...)` |
| `roll_alpha` | 0.0 | → `rolling.roll_alpha(..., risk_free, ...)` |
| `roll_beta` | 0.0 | → `rolling.roll_beta(..., risk_free, ...)` |
| `roll_alpha_beta` | 0.0 | → `rolling.roll_alpha_beta(..., risk_free, ...)` |
| `regression_annual_return` | 0.0 | → `alpha(...)` 和 `beta(...)` |

### 三、缺少 risk_free 参数，但内部逻辑需要使用的函数（需要修改）

以下函数**没有** `risk_free` 参数，但按照文档的指标算法定义，它们在概念上应该支持 risk_free，或者它们调用了使用 risk_free 的函数但没有传递：

| 文件 | 函数 | 当前签名 | 问题 | 建议修改 |
|------|------|---------|------|---------|
| `metrics/ratios.py` | `calmar_ratio` | `(returns, period, annualization)` | 文档定义 `Calmar = (r_annual - rf) / MD`，当前不减 rf | 添加 `risk_free=0` 参数，分子改为 `ann_return - risk_free` |
| `metrics/ratios.py` | `sortino_ratio` | `(returns, required_return=0, period, annualization, ...)` | 参数名为 `required_return`（默认0），不叫 `risk_free`；但文档中 Sortino 的阈值就是 rf | **无需修改签名**（`required_return` 已可作为 rf 传入），但调用方需注意传入 rf 值 |
| `metrics/ratios.py` | `conditional_sharpe_ratio` | `(returns, cutoff, period, annualization)` | 内部计算 `mean / std * sqrt(q)`，未减去 rf | 添加 `risk_free=0` 参数，excess = `conditional_returns - risk_free` |
| `metrics/ratios.py` | `information_ratio` | `(returns, factor_returns, period, annualization)` | 内部直接用 `returns - factor_returns`，未涉及 rf | **无需修改**（IR 通常不使用 rf，是相对基准的指标） |
| `metrics/ratios.py` | `common_sense_ratio` | `(returns)` | 调用 `tail_ratio` 和 `win_rate`，与 rf 无关 | **无需修改** |
| `metrics/ratios.py` | `stability_of_timeseries` | `(returns)` | 与 rf 无关 | **无需修改** |
| `metrics/ratios.py` | `capture` | `(returns, factor_returns, period)` | 调用 `annual_return`，与 rf 无关 | **无需修改** |
| `metrics/ratios.py` | `up_capture` | `(returns, factor_returns, period)` | 同上 | **无需修改** |
| `metrics/ratios.py` | `down_capture` | `(returns, factor_returns, period)` | 同上 | **无需修改** |
| `metrics/ratios.py` | `up_down_capture` | `(returns, factor_returns, period)` | 同上 | **无需修改** |

---

### 四、需要修改的函数清单

#### 4.1 `calmar_ratio` — 添加 `risk_free=0` 参数

**文件**: `fincore/metrics/ratios.py:316`

**当前签名**:
```python
def calmar_ratio(returns, period=DAILY, annualization=None):
```

**建议修改为**:
```python
def calmar_ratio(returns, risk_free=0, period=DAILY, annualization=None):
```

**内部修改**: `temp = ann_return / abs(max_dd)` → `temp = (ann_return - risk_free) / abs(max_dd)`

**兼容性**: 默认值 `risk_free=0` 保持现有行为不变。

#### 4.2 `conditional_sharpe_ratio` — 添加 `risk_free=0` 参数

**文件**: `fincore/metrics/ratios.py:270`

**当前签名**:
```python
def conditional_sharpe_ratio(returns, cutoff=0.05, period=DAILY, annualization=None):
```

**建议修改为**:
```python
def conditional_sharpe_ratio(returns, cutoff=0.05, risk_free=0, period=DAILY, annualization=None):
```

**内部修改**: 在计算 `mean_ret` 前减去 risk_free:
```python
conditional_returns = conditional_returns - risk_free
```

**兼容性**: 默认值 `risk_free=0` 保持现有行为不变。

---

### 五、Empyrical 类及 _registry.py 对应修改

当底层函数签名变更后，需要同步检查以下位置：

| 位置 | 函数/方法 | 修改内容 |
|------|----------|---------|
| `_registry.py` CLASSMETHOD_REGISTRY | `calmar_ratio` | **无需修改**（注册表只记录函数名，参数透传） |
| `_registry.py` CLASSMETHOD_REGISTRY | `conditional_sharpe_ratio` | **无需修改**（同上） |
| `empyrical.py` | 无显式 `calmar_ratio` 的 `@_dual_method` | **无需修改**（通过注册表自动透传） |
| `empyrical.py` | 无显式 `conditional_sharpe_ratio` 的 `@_dual_method` | **无需修改**（通过注册表自动透传） |

> 注：由于 `calmar_ratio` 和 `conditional_sharpe_ratio` 在 `_registry.py` 中通过 `CLASSMETHOD_REGISTRY` 注册，Empyrical 类通过 `_LazyMethod` 描述符直接委托到底层函数，参数自动透传。添加 `risk_free=0` 默认参数后，所有现有调用方式完全兼容。

---

### 六、总结

**需要修改的函数（2个）**：

1. **`calmar_ratio`** (`fincore/metrics/ratios.py:316`) — 添加 `risk_free=0` 参数
2. **`conditional_sharpe_ratio`** (`fincore/metrics/ratios.py:270`) — 添加 `risk_free=0` 参数

**不需要修改的函数**：
- 所有已有 `risk_free` 参数的函数（30个）— 参数已正确传递
- `sortino_ratio` — 已有 `required_return` 参数可作为 rf 传入
- `information_ratio` — IR 定义不涉及 rf
- `capture` / `up_capture` / `down_capture` / `up_down_capture` — 捕获率定义不涉及 rf
- `common_sense_ratio` / `stability_of_timeseries` — 与 rf 无关
- Empyrical 类和 `_registry.py` — 通过注册表自动透传，底层函数改签名后自动兼容