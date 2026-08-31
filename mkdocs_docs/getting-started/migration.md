# Migration to fincore 0.5

fincore **0.5.0.dev0** is a breaking unified-core release. It retains the
analytical capability areas associated with Empyrical, Pyfolio, and Alphalens,
but it intentionally removes their path-shaped compatibility layers. There is
no deprecated fallback period and no compatibility extra to install.

## Replace imports by purpose

| Previous purpose | 0.5 destination |
| --- | --- |
| scalar and rolling performance metrics | focused modules below `fincore.metrics` |
| cash-flow-aware performance calculation | `fincore.performance.cashflows` |
| tear-sheet style portfolio analysis | `fincore.report.portfolio.compute` plus a renderer below `fincore.report.renderers` |
| factor cleaning, returns, IC, turnover, and factor portfolios | focused modules below `fincore.factor_analysis` |
| performance attribution | `fincore.attribution.performance` |
| root-level flat functions, stateful analysis contexts, or rolling engines | compose direct domain operations for the required workflow |

For example:

```python
import pandas as pd

from fincore.metrics.drawdown import max_drawdown
from fincore.metrics.ratios import sharpe_ratio

returns = pd.Series([0.01, -0.005, 0.002, 0.004])
print(sharpe_ratio(returns), max_drawdown(returns))
```

## Reporting model

Instead of invoking a tear-sheet façade, build one report model and choose a
renderer explicitly:

```python
from fincore.report.portfolio.compute import build_portfolio_report
from fincore.report.renderers.html import write_html

document = build_portfolio_report(returns)
write_html(document, "portfolio-report.html")
```

This separates computational semantics from presentation and lets HTML, PDF,
XLSX, and interactive output share a single document model.

## Migration checklist

1. Raise your project to Python 3.11 or newer.
2. Inventory all old imports and identify the capability each call supplies.
3. Replace each call with the owning 0.5 domain module; do not substitute a
   root-level alias.
4. Select only the required capability extras.
5. Validate numerical outputs and generated reports against your own production
   data and edge cases.

The detailed repository-level guide is also available in the
[repository migration document](https://github.com/cloudQuant/fincore/blob/master/docs/MIGRATION.md).
