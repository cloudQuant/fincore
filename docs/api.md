# API Guide

fincore 0.5 is a breaking, domain-oriented API. The package root is a
namespace index, not a flat callable surface. Import each function or model
from the leaf module that owns its financial contract.

## Metrics

```python
from fincore.metrics.drawdown import max_drawdown
from fincore.metrics.ratios import sharpe_ratio
from fincore.metrics.yearly import annual_return

summary = {
    "sharpe_ratio": sharpe_ratio(returns),
    "maximum_drawdown": max_drawdown(returns),
    "annual_return": annual_return(returns),
}
```

Other metric families live in `fincore.metrics.returns`,
`fincore.metrics.rolling`, `fincore.metrics.risk`, and
`fincore.metrics.statistics`.

## Runtime orchestration

Use the runtime only when an application needs an immutable input boundary,
catalog resolution, batch planning, or provenance-bearing `Result` objects.
For a single domain calculation, call the leaf function directly.

```python
from fincore.runtime.builtins import builtin_catalog
from fincore.runtime.engine import run

result = run(
    "metrics.ratios.sharpe_ratio",
    {"returns": returns},
    catalog=builtin_catalog(),
)
print(result.value)
```

## Portfolio reports

Build one immutable report document and render that document without repeating
financial calculation.

```python
from fincore.report.portfolio.compute import build_portfolio_report
from fincore.report.renderers.html import write_html

document = build_portfolio_report(returns, positions=positions)
artifacts = write_html(document, "portfolio-report.html")
```

PDF and XLSX renderers are in `fincore.report.renderers.pdf` and
`fincore.report.renderers.xlsx` and require their matching capability extras.

## Domain map

| Domain | Owning modules |
| --- | --- |
| Metrics and time-series analytics | `fincore.metrics.*` |
| Cash-flow performance and inference | `fincore.performance.*` |
| Positions, transactions, capacity, round trips | `fincore.portfolio.*` |
| Factor preparation, analysis, portfolios, inference | `fincore.factor_analysis.*` |
| Brinson and factor-performance attribution | `fincore.attribution.*` |
| Risk models, calibration, backtesting, diagnostics | `fincore.risk.*` |
| Allocation optimisation and simulation | `fincore.optimization.*`, `fincore.simulation.*` |
| Report models and renderers | `fincore.report.*` |
| Data providers, extensions, visualisation, runtime | `fincore.data.*`, `fincore.extensions.*`, `fincore.viz.*`, `fincore.runtime.*` |

See [the migration guide](MIGRATION.md) for the 0.5 boundary and the
[MkDocs API reference](../mkdocs_docs/api/index.md) for focused module pages.
