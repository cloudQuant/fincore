# Quick start

fincore 0.5 is a domain-oriented performance-analysis platform. The root
package is a namespace index; import each operation from the focused module
that owns it. The examples below are exercised by `tests/docs/test_examples.py`.

## Metrics

```python
import pandas as pd

from fincore.metrics.drawdown import max_drawdown
from fincore.metrics.ratios import sharpe_ratio
from fincore.metrics.yearly import annual_return

returns = pd.Series([0.01, -0.005, 0.002, 0.004])

print(sharpe_ratio(returns))
print(max_drawdown(returns))
print(annual_return(returns))
```

## Build a portfolio report, then render it

Report construction is analytical; rendering is a separate projection of the
immutable `ReportDocument`.

```python
import pandas as pd

from fincore.report.portfolio.compute import build_portfolio_report
from fincore.report.renderers.html import write_html

dates = pd.date_range("2024-01-02", periods=5, freq="B")
returns = pd.Series([0.01, -0.005, 0.002, 0.004, -0.001], index=dates)
positions = pd.DataFrame({"AAA": 100.0, "BBB": -30.0, "cash": 80.0}, index=dates)

document = build_portfolio_report(returns, positions=positions, rolling_window=3)
artifacts = write_html(document, "portfolio-report.html")
print(artifacts.named_artifacts["file"])
```

## Factor analysis

The checked-in quickstart is offline and deterministic. It covers canonical
input preparation, analysis, portfolio inputs, and an optional headless
matplotlib summary:

```bash
pip install "fincore[visualization]"
MPLBACKEND=Agg python examples/factor_analysis_quickstart.py
```

For a focused integration, import directly from the relevant owning module:
`fincore.factor_analysis.data`, `analysis`, `performance`, `portfolio`,
`costs`, `inference`, or `render_matplotlib`.

## More domains

- Cash-flow-aware returns: `fincore.performance.cashflows`
- Positions, transactions, and capacity: `fincore.portfolio`
- Risk diagnostics and validation reports: `fincore.risk`
- Attribution: `fincore.attribution.performance`
- Optimisation: `fincore.optimization`
- Simulation: `fincore.simulation`

Use the [migration guide](migration.md) when replacing pre-0.5 imports.
