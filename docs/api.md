# API Guide

This page documents the intended stable entry points. Internal modules may change.

## Top-Level Flat API

The `fincore` package re-exports a set of common metrics as flat functions:

```python
import fincore

sr = fincore.sharpe_ratio(returns)
md = fincore.max_drawdown(returns)
```

## `fincore.Empyrical`

`Empyrical` provides:

- Class-level access to a large registry of metrics (e.g. `Empyrical.sharpe_ratio(returns)`).
- A small set of instance helpers that can auto-fill `returns` / `factor_returns`.

```python
from fincore import Empyrical

emp = Empyrical(returns=returns)
dd_days = emp.max_drawdown_days()
```

## `fincore.Pyfolio`

`Pyfolio` extends `Empyrical` with tear sheet and plotting functions.

```python
from fincore import Pyfolio

pf = Pyfolio(returns=returns)
pf.create_returns_tear_sheet(returns)
```

## Reports

Generate a dynamic HTML or PDF report:

```python
from fincore.report import create_strategy_report

create_strategy_report(returns, output="report.html")
```

## Data Providers

Some attribution/data workflows depend on provider callbacks to avoid hard dependencies on specific data sources.

Fama-French factors (provider injection):

```python
from fincore.attribution import fama_french

def my_provider(start: str, end: str, library: str):
    ...

fama_french.set_ff_provider(my_provider)
df = fama_french.fetch_ff_factors("2020-01-01", "2020-12-31", library="french")
```

