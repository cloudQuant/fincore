# Core Concepts

## Data Model

Most APIs operate on daily (or intraday) return series:

- **`returns`**: `pd.Series` of simple (non-cumulative) returns with `DatetimeIndex`
- **`factor_returns`**: optional benchmark returns aligned to `returns`
- **`positions`**: optional `pd.DataFrame` with one column per asset plus `cash`
- **`transactions`**: optional `pd.DataFrame` with `amount`, `price`, `symbol` columns

## Three API Levels

### 1. Flat API (Simplest)
```python
import fincore
fincore.sharpe_ratio(returns)
```

### 2. Empyrical Class (150+ methods)
```python
from fincore import Empyrical
Empyrical.sharpe_ratio(returns)
```

### 3. AnalysisContext (Recommended)
```python
ctx = fincore.analyze(returns, factor_returns=benchmark)
ctx.sharpe_ratio  # lazy, cached
```

## Lazy Loading Architecture

`import fincore` loads in ~0.04s. Heavy submodules (matplotlib, scipy) are deferred via `__getattr__` until first access.

## Period Constants

```python
from fincore.constants import DAILY, WEEKLY, MONTHLY, QUARTERLY, YEARLY
```

These control annualization factors across all metrics.
