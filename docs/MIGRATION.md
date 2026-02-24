# Migrating from empyrical to fincore

This guide helps users migrate from the original `empyrical` library to `fincore`.

## Overview

`fincore` is a continuation of the `empyrical` analytics stack, maintained by cloudQuant. It provides the same functionality with additional features and improvements.

## Installation

### empyrical (Original)
```bash
pip install empyrical
```

### fincore (New)
```bash
pip install fincore

# With visualization support
pip install "fincore[viz]"

# With all optional dependencies
pip install "fincore[all]"
```

## API Compatibility

### Fully Compatible

Most `empyrical` functions work identically in `fincore`:

```python
# Before (empyrical)
from empyrical import sharpe_ratio, max_drawdown, alpha_beta
sharpe = sharpe_ratio(returns, risk_free=0.02)
mdd = max_drawdown(returns)
alpha, beta = alpha_beta(returns, benchmark_returns)

# After (fincore)
from fincore import sharpe_ratio, max_drawdown, alpha_beta
sharpe = sharpe_ratio(returns, risk_free=0.02)
mdd = max_drawdown(returns)
alpha, beta = alpha_beta(returns, benchmark_returns)
```

### Class-Based API

`fincore` provides the `Empyrical` class for stateful operations:

```python
from fincore import Empyrical

emp = Empyrical(returns=returns, factor_returns=benchmark_returns)
sharpe = emp.sharpe_ratio()
alpha = emp.alpha()
beta = emp.beta()
```

## New Features

### 1. AnalysisContext (Recommended)

The new `AnalysisContext` API provides lazy, cached metric computation:

```python
import fincore

# One-liner analysis with automatic caching
ctx = fincore.analyze(returns, factor_returns=benchmark)

# Metrics are computed on first access and cached
print(f"Sharpe: {ctx.sharpe_ratio}")
print(f"Alpha: {ctx.alpha}")

# Export to JSON
ctx.to_json(path="metrics.json")

# Generate HTML report
ctx.to_html(path="report.html")
```

### 2. RollingEngine

Batch rolling metric computation in a single call:

```python
from fincore.core.engine import RollingEngine

engine = RollingEngine(returns, factor_returns=benchmark, window=60)
results = engine.compute(['sharpe', 'volatility', 'max_drawdown', 'beta'])
```

### 3. Data Providers

Unified interface for fetching financial data:

```python
from fincore.data import YahooFinanceProvider, TushareProvider

# Yahoo Finance (free)
yahoo = YahooFinanceProvider()
data = yahoo.fetch("AAPL", start="2020-01-01", end="2024-12-31")

# Tushare (Chinese A-shares)
tushare = TushareProvider(token="YOUR_TOKEN")
data = tushare.fetch("000001.SZ", start="2020-01-01", end="2024-12-31")
```

### 4. Portfolio Optimization

```python
from fincore.optimization import efficient_frontier, risk_parity

# Efficient frontier
ef = efficient_frontier(returns, n_points=50)

# Risk parity portfolio
rp = risk_parity(returns)
```

### 5. Pluggable Visualization

Multiple visualization backends:

```python
import fincore

ctx = fincore.analyze(returns, factor_returns=benchmark)

# Matplotlib
ctx.plot(backend="matplotlib")

# Self-contained HTML
ctx.plot(backend="html")

# Plotly (if installed)
ctx.plot(backend="plotly")
```

## Import Changes

| empyrical | fincore |
|-----------|----------|
| `import empyrical` | `import fincore` or `from fincore import Empyrical` |
| `empyrical.SHARPE_RATIO` | `fincore.constants.DAILY` (use period constants) |
| `from empyrical import *` | `from fincore import sharpe_ratio, max_drawdown, ...` |

## Breaking Changes

None. `fincore` maintains full backward compatibility with `empyrical`.

## Performance Improvements

- **Import time**: ~0.06s (vs ~0.5s for empyrical)
- **Lazy loading**: Heavy submodules load only on first use
- **Cached computation**: `AnalysisContext` avoids redundant calculations

## Migration Steps

### Step 1: Update Requirements

Replace in your `requirements.txt`:
```
# Remove
empyrical

# Add
fincore>=1.0.0
```

### Step 2: Update Imports

Find and replace:
```bash
# In your code files
from empyrical import → from fincore import
import empyrical → import fincore
```

### Step 3: Test Your Code

Run your existing tests. Most should work without changes.

### Step 4: Adopt New Features (Optional)

Gradually adopt new features like `AnalysisContext` for better performance.

## Frequently Asked Questions

### Q: Is fincore a drop-in replacement for empyrical?

A: Yes, for most use cases. The core API is fully compatible.

### Q: What Python versions are supported?

A: Python 3.11, 3.12, and 3.13.

### Q: Can I use both empyrical and fincore in the same project?

A: Yes, but it's recommended to migrate completely to avoid confusion.

### Q: Where can I get help?

A:
- GitHub: https://github.com/cloudQuant/fincore
- Documentation: See `docs/` directory

### Q: How do I report bugs or request features?

A: Please open an issue on GitHub with a minimal reproducible example.

## Summary of New Features

| Feature | empyrical | fincore |
|---------|-----------|---------|
| Core Metrics | ✅ | ✅ |
| Lazy Loading | ❌ | ✅ |
| AnalysisContext | ❌ | ✅ |
| RollingEngine | ❌ | ✅ |
| Data Providers | ❌ | ✅ |
| Portfolio Optimization | ❌ | ✅ |
| Pluggable Viz | ❌ | ✅ |
| Python 3.13 | ❌ | ✅ |

## Next Steps

1. Install `fincore`
2. Update your imports
3. Run your tests
4. Explore new features like `AnalysisContext`
5. Enjoy improved performance!
