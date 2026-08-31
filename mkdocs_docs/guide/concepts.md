# Domain model and workflow boundaries

fincore works with labelled pandas inputs. Most direct metric functions accept
a return `Series`; portfolio and reporting workflows optionally accept aligned
positions, transactions, benchmarks, and metadata.

## One operation, one owner

The public API is deliberately not a flat convenience layer. A function lives
in the domain responsible for its semantics:

```python
from fincore.metrics.ratios import sharpe_ratio
from fincore.performance.cashflows import cashflow_adjusted_twr
from fincore.report.portfolio.compute import build_portfolio_report
```

This keeps validation, numerical behaviour, and optional dependencies local to
the operation rather than routing calls through package-shaped compatibility
layers.

## Immutable boundaries

Canonical report documents, runtime artifacts, data snapshots, and extension
snapshots are immutable at their public boundary. Callers can retain a result
without later calculations silently changing it.

## Optional capabilities

Core analysis does not import renderers, providers, Bayesian engines, or
interactive backends eagerly. Install the named extra for the capability you
select, such as `fincore[visualization]` or `fincore[report-xlsx]`.

## Removed 0.5 surfaces

Upstream-shaped module paths, root-level calls, stateful analysis context
objects, and rolling-engine facades were removed. See the
[migration guide](../getting-started/migration.md) for the purpose-based
replacement map.
