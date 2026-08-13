# AnalysisContext

The `AnalysisContext` provides lazy, cached metric computation with export capabilities.

## Basic Usage

```python
import pandas as pd

import fincore

index = pd.date_range("2024-01-02", periods=5, freq="B")
returns = pd.Series([0.01, -0.005, 0.002, 0.004, -0.001], index=index)
benchmark = pd.Series([0.008, -0.003, 0.001, 0.002, 0.0], index=index)

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
# JSON text
json_str = ctx.to_json()

# JSON file
ctx.to_json(path="report.json")

# Dictionary
d = ctx.to_dict()

# HTML report (self-contained, no extra dependencies)
ctx.to_html(path="report.html")

# Plot -> ReportArtifacts (requires fincore[viz] for matplotlib)
artifacts = ctx.plot(backend="matplotlib")
```

## Snapshot semantics and cache invalidation

Inputs are defensively snapshotted: mutating the caller's series after
`analyze()` does not change cached results. `replace_data()` atomically swaps
inputs and invalidates every cached metric:

```python
ctx.replace_data(returns=returns + 0.001)
```

## API Reference

::: fincore.core.context.AnalysisContext
