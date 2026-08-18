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

## Data providers and snapshots

Data providers are `provider_required` capabilities: each needs its optional
extra and a working transport. A broken or missing optional SDK surfaces as a
controlled `fincore.exceptions.DependencyError` (which names the required extra),
never a raw third-party error:

```python
from fincore.data import YahooFinanceProvider

provider = YahooFinanceProvider()  # raises DependencyError if yfinance is missing/broken
```

For offline tests, inject an in-memory client so the provider logic runs
without the SDK:

```python
provider = YahooFinanceProvider(client=fake_client)
```

To make an external-data analysis reproducible, wrap the fetched frame in a
`DataSnapshot`, which freezes the source, request interval, as-of timestamp,
price-adjustment convention, and a SHA256 of the data — without ever recording
secret configuration:

```python
from fincore.data.snapshots import DataSnapshot

snapshot = DataSnapshot.from_frame(
    frame, provider="yahoo",
    requested_start="2024-01-01", requested_end="2024-12-31",
    as_of="2024-12-31T23:59:59Z",
)
manifest = snapshot.to_manifest()  # provenance only; no API keys or raw data
```
