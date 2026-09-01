# Migrating to fincore 0.5.0

## Scope of the breaking change

fincore 0.5 consolidates the former Empyrical, Pyfolio, and Alphalens capability
areas into a single domain-oriented architecture. The capability set remains in
scope; the old package-shaped APIs do not. This is an intentional breaking
change, not a deprecation layer.

Removed surfaces include:

- `fincore.empyrical`, `fincore.pyfolio`, and `fincore.alphalens`;
- root-level metric functions and façade classes;
- stateful compatibility contexts and profile-specific tear-sheet entry points;
- compatibility alias extras and dynamic import aliases.

The 0.5 package root is an index of canonical domains only. Public executable
APIs live in leaf modules so each operation has one implementation path.

## Capability map

| Capability | Canonical 0.5 location |
| --- | --- |
| returns, drawdown, alpha/beta, ratios, rolling metrics, statistics | `fincore.metrics.*` |
| cash-flow-aware TWR and performance disclosures | `fincore.performance.*` |
| positions, transactions, capacity, round trips | `fincore.portfolio.*` |
| portfolio reporting | `fincore.report.portfolio.compute` and `fincore.report.renderers.*` |
| factor preparation, forward returns, IC, turnover, portfolios, costs, inference | `fincore.factor_analysis.*` |
| Brinson and factor performance attribution | `fincore.attribution.*` |
| VaR, diagnostics, calibration, EVT, GARCH | `fincore.risk.*` |
| allocation optimisation | `fincore.optimization.*` |
| Monte Carlo, bootstrap, scenarios | `fincore.simulation.*` |
| data, extensions, visualisation, runtime services | `fincore.data.*`, `extensions.*`, `viz.*`, `runtime.*` |

## Direct metrics

```python
import pandas as pd

from fincore.metrics.drawdown import max_drawdown
from fincore.metrics.ratios import sharpe_ratio
from fincore.metrics.yearly import annual_return

returns = pd.Series([0.01, -0.005, 0.002, 0.004])

summary = {
    "sharpe_ratio": sharpe_ratio(returns),
    "max_drawdown": max_drawdown(returns),
    "annual_return": annual_return(returns),
}
```

Do not change an old import to `from fincore import ...`; root-level callable
exports were removed. Import the leaf function that owns the relevant semantic
contract.

## Report workflows

Portfolio analysis now produces a canonical immutable `ReportDocument`. Each
renderer projects that same model without recomputing financial results.

```python
import pandas as pd

from fincore.report.portfolio.compute import build_portfolio_report
from fincore.report.renderers.html import write_html

dates = pd.date_range("2024-01-02", periods=5, freq="B")
returns = pd.Series([0.01, -0.005, 0.002, 0.004, -0.001], index=dates)

document = build_portfolio_report(returns)
artifact = write_html(document, "portfolio-report.html")
```

Use a PDF/XLSX renderer only after installing `fincore[report-pdf]` or
`fincore[report-xlsx]`; model construction remains in the core package.

## Factor analysis workflows

Factor analysis is now a unified domain with separate layers for input
preparation, model calculation, portfolio construction, costs, statistical
inference, and optional rendering. Start from an owning module, for example:

```python
from fincore.factor_analysis.analysis import analyze_factor
from fincore.factor_analysis.data import get_clean_factor_and_forward_returns
from fincore.factor_analysis.performance import mean_return_by_quantile
```

The deterministic offline example at
[`examples/factor_analysis_quickstart.py`](../examples/factor_analysis_quickstart.py)
shows the complete preparation-to-render path. Install
`fincore[visualization]` when using matplotlib rendering.

## Upgrade checklist

1. Upgrade to Python 3.11+ and install `fincore>=0.5.0`.
2. Classify every old integration by business capability rather than by package
   or function name.
3. Replace it with the owning domain operation in the table above.
4. Install only the extras required by the chosen renderers, inference engines,
   or data providers.
5. Test production-shaped data including missing values, timestamps, alignment,
   cash flows, and output artifacts.
6. Remove all dependencies on the old import paths; a successful migration has
   no compatibility shim in its dependency graph.

## Support boundary

The repository verifies canonical capability scenarios, report semantics,
extension snapshots, removed legacy surfaces, package contents, and executable
documentation. It does not claim byte-for-byte or call-signature compatibility
with retired upstream packages. Validate application-level semantics before
deploying this breaking release.
