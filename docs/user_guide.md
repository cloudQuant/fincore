# User Guide

## Install

```bash
pip install fincore
pip install "fincore[visualization]"     # optional charts
pip install "fincore[report-pdf]"        # optional PDF renderer
pip install "fincore[report-xlsx]"       # optional XLSX renderer
```

For source work, install `pip install -e ".[dev]"`. Python 3.11+ is required.

## Inputs

Canonical functions consume explicit labelled inputs:

- `returns`: simple periodic `pd.Series`, normally with a `DatetimeIndex`.
- `benchmark_returns`: optional return series aligned by timestamp.
- `positions`: portfolio-value panel with one column per asset and any required
  cash column.
- `transactions`: timestamped transaction panel with the fields required by the
  particular portfolio workflow.

The runtime's `AnalysisSnapshot` copies inputs on ingest. Direct domain calls
remain the most compact option for one calculation.

## Metrics

```python
import numpy as np
import pandas as pd

from fincore.metrics.drawdown import max_drawdown
from fincore.metrics.ratios import sharpe_ratio
from fincore.metrics.yearly import annual_return

dates = pd.date_range("2024-01-02", periods=252, freq="B", tz="UTC")
returns = pd.Series(np.random.default_rng(42).normal(0.0003, 0.01, len(dates)), index=dates)

print("Sharpe:", sharpe_ratio(returns))
print("Max drawdown:", max_drawdown(returns))
print("Annual return:", annual_return(returns))
```

## Report workflow

```python
from fincore.report.portfolio.compute import build_portfolio_report
from fincore.report.renderers.html import write_html

document = build_portfolio_report(returns, positions=positions)
write_html(document, "portfolio-report.html")
```

The report model is independent from its renderer, so the same document can be
rendered as HTML, PDF, XLSX, or an interactive chart after installing the
matching optional capability.

## Factor research

Use `fincore.factor_analysis.data` to prepare factor data, then call the
specific owning module for analysis, portfolio construction, inference, costs,
or rendering. The executable end-to-end reference is
[`examples/factor_analysis_quickstart.py`](../examples/factor_analysis_quickstart.py).

## Breaking-change boundary

0.5 does not provide old package-shaped imports, façade classes, root metric
aliases, or compatibility extras. Map an integration to a business capability
and select its canonical leaf module; do not use an import shim. See
[MIGRATION.md](MIGRATION.md).
