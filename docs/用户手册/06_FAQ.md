# 常见问题 (FAQ)

## Q1: `import fincore` 很慢？

fincore 使用延迟加载，`import fincore` 应在 ~0.05 秒内完成。如果很慢，检查：

- 是否有其他包在导入时触发了重量级依赖
- 使用 `python -X importtime -c "import fincore"` 分析导入耗时

## Q2: 收益率应该用简单收益还是对数收益？

fincore 默认使用**简单收益率**（arithmetic returns）：

```python
returns = prices.pct_change().dropna()  # 简单收益
```

大部分指标（夏普、最大回撤等）均基于简单收益。如果你的数据是对数收益，需要先转换：

```python
simple_returns = np.exp(log_returns) - 1
```

## Q3: 为什么夏普比率和其他工具算出来不一样？

常见原因：

1. **年化因子不同**: fincore 默认 `period="daily"`, 年化因子 = 252。部分工具用 365 或 260。
2. **无风险利率**: fincore 默认 `risk_free=0`。其他工具可能使用国债利率。
3. **样本标准差 vs 总体标准差**: fincore 使用 `ddof=1`（样本标准差）。

可通过参数调整：

```python
Empyrical.sharpe_ratio(returns, risk_free=0.02/252, period="daily")
```

## Q4: 如何处理含 NaN 的收益率序列？

```python
# 方法 1: 删除 NaN
returns = returns.dropna()

# 方法 2: 前向填充
returns = returns.fillna(method="ffill")

# 方法 3: 填充为 0（假设无收益）
returns = returns.fillna(0)
```

fincore 的大部分函数内部使用 `np.nanmean` / `np.nanstd`，可容忍少量 NaN，但建议在输入前清洗数据。

## Q5: 如何对比多个策略？

```python
from fincore import Empyrical
import pandas as pd

strategies = {"A": returns_a, "B": returns_b}
comparison = pd.DataFrame({
    name: Empyrical.perf_stats(r)
    for name, r in strategies.items()
})
print(comparison)
```

## Q6: PDF 报告生成失败？

PDF 渲染依赖 `weasyprint` 或类似 HTML-to-PDF 工具链。如果缺少依赖：

```bash
# macOS
brew install pango

# Ubuntu
sudo apt-get install libpango-1.0-0 libpangocairo-1.0-0

pip install weasyprint
```

如果 PDF 不可用，可以先生成 HTML，然后用浏览器打印为 PDF。

## Q7: 如何使用分钟/小时级数据？

通过 `period` 参数指定数据频率：

```python
from fincore import Empyrical

# 小时数据
sr = Empyrical.sharpe_ratio(hourly_returns, period="hourly")

# 分钟数据
sr = Empyrical.sharpe_ratio(minute_returns, period="minutely")
```

支持的 period 值：`"daily"`, `"weekly"`, `"monthly"`, `"hourly"`, `"minutely"`

## Q8: Empyrical 实例方法和类方法有什么区别？

| 调用方式 | 是否需要传 returns | 适用场景 |
|---------|-------------------|---------|
| `Empyrical.sharpe_ratio(returns)` | 是 | 快速计算单个指标 |
| `emp.sharpe_ratio(returns)` | 是 | 同上 |
| `emp.win_rate()` | 否（自动填充） | 绑定数据后批量计算 |

实例方法通过 `@_dual_method` 装饰器实现自动参数填充。

## Q9: 如何扩展自定义指标？

fincore 的注册表机制支持扩展。最简单的方式：

```python
# 直接作为函数使用
def my_custom_ratio(returns):
    return returns.mean() / returns.std() * np.sqrt(252)

result = my_custom_ratio(returns)
```

## Q10: 支持哪些可视化后端？

| 后端 | 用途 |
|------|------|
| `matplotlib` | 静态图表（PDF/PNG），Pyfolio tear sheet |
| `html` | 自包含 HTML 报告 |
| `bokeh` | 交互式图表（需安装 bokeh） |
| `plotly` | 交互式图表（需安装 plotly） |
