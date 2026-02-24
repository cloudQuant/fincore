# AnalysisContext

The `AnalysisContext` provides lazy, cached metric computation with export capabilities.

## Basic Usage

```python
import fincore

ctx = fincore.analyze(returns, factor_returns=benchmark)

# Metrics computed on first access, then cached
print(ctx.sharpe_ratio)
print(ctx.max_drawdown)
print(ctx.annual_return)
print(ctx.alpha)
print(ctx.beta)
```

## Performance Stats

```python
stats = ctx.perf_stats()  # pandas Series with all key metrics
print(stats)
```

## Export

```python
# JSON
json_str = ctx.to_json()

# Dictionary
d = ctx.to_dict()

# HTML report (self-contained, no extra dependencies)
ctx.to_html(path="report.html")
```

## API Reference

::: fincore.core.context.AnalysisContext
