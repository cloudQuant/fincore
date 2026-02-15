# User Guide

## Installation

From source (recommended for development):

```bash
pip install -e ".[dev]"
```

Minimal runtime install:

```bash
pip install .
```

## Core Data Model

Most APIs operate on daily (or intraday) return series and related optional inputs.

- `returns`: `pd.Series` of simple (non-cumulative) returns indexed by `pd.DatetimeIndex`.
- `benchmark_rets`: optional `pd.Series` aligned to `returns`.
- `positions`: optional `pd.DataFrame` indexed like `returns`, with one column per asset plus a `cash` column.
- `transactions`: optional `pd.DataFrame` indexed by timestamp, with at least `amount`, `price`, `symbol` columns.

Timezone handling:

- Prefer timezone-aware indices (e.g. UTC). Many utilities assume consistent timezone handling when aligning data.

## Quickstart

### Flat API (function style)

```python
import numpy as np
import pandas as pd
import fincore

dates = pd.date_range("2023-01-01", periods=252, freq="B", tz="UTC")
returns = pd.Series(np.random.normal(0.0003, 0.01, len(dates)), index=dates, name="strategy")

print("Sharpe:", fincore.sharpe_ratio(returns))
print("MaxDD:", fincore.max_drawdown(returns))
print("CAGR:", fincore.annual_return(returns))
```

### `Empyrical` (class interface)

```python
from fincore import Empyrical

stats = Empyrical.perf_stats(returns)
dd_table = Empyrical.gen_drawdown_table(returns, top=5)
```

You can also bind data to an instance and call selected helpers without re-passing `returns`:

```python
emp = Empyrical(returns=returns)
print(emp.max_drawdown_days())
print(emp.win_rate())
```

## Strategy Reports (HTML/PDF)

Generate a report with progressively richer sections as you provide more inputs:

```python
from fincore.report import create_strategy_report

out = create_strategy_report(
    returns,
    title="My Strategy",
    output="report.html",  # or report.pdf
)
print(out)
```

Notes:

- PDF rendering requires the optional PDF toolchain used by `fincore.report.render_pdf`.
- If you only have returns, the report will only include the returns-focused sections.

