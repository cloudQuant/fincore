# Core Concepts

## Data Model

Most APIs operate on daily (or intraday) return series:

- **`returns`**: `pd.Series` of simple (non-cumulative) returns with `DatetimeIndex`
- **`factor_returns`**: optional benchmark returns aligned to `returns`
- **`positions`**: optional `pd.DataFrame` with one column per asset plus `cash`
- **`transactions`**: optional `pd.DataFrame` with `amount`, `price`, `symbol` columns

## Three API surfaces

fincore 0.3.0 exposes three clearly separated surfaces. Equal names do not
imply equal semantics across them:

### 1. Enhanced semantics — flat API and `AnalysisContext` (Recommended)

The flat API is bound to enhanced `fincore.metrics` implementations with
documented divergences (e.g. `week_year="iso"`, validation exceptions):

```python
import fincore
fincore.sharpe_ratio(returns)
```

`AnalysisContext` is the recommended stateful, cached API:

```python
ctx = fincore.analyze(returns, factor_returns=benchmark)
ctx.sharpe_ratio  # lazy, cached
```

### 2. Strict compatibility — `fincore.empyrical`

The frozen empyrical 0.6.0 surface (54/54 C0, 49/49 C1, core callables C3):

```python
from fincore import empyrical
empyrical.sharpe_ratio(returns)

from fincore import Empyrical
Empyrical.sharpe_ratio(returns)   # class-level, explicit returns
emp = Empyrical(returns=returns)
emp.sharpe_ratio()                # instance-level, state-bound
```

### 3. pyfolio façade — `fincore.pyfolio`

The frozen pyfolio 0.9.6 profile of 11 tear-sheet workflows (all C1, main
chains C4). `from fincore import Pyfolio` requires the `pyfolio` extra.

See [Compatibility](../development/compatibility.md) for the C0–C4 matrix.

## Lazy Loading Architecture

`import fincore` loads in ~0.04s. Heavy submodules (matplotlib, scipy) are deferred via `__getattr__` until first access.

## Period Constants

```python
from fincore.constants import DAILY, WEEKLY, MONTHLY, QUARTERLY, YEARLY
```

These control annualization factors across all metrics.
