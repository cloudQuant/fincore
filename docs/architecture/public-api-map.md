# Public API Map

Fincore 0.5 has one product model: direct domain leaf APIs plus the optional
runtime catalog. The package root exports only versioning, errors, and domain
namespaces. It does not export financial leaf functions, façade classes,
profiles, or dynamic aliases.

| Domain | Canonical ownership examples | Catalog role |
| --- | --- | --- |
| `metrics` | `metrics.ratios.sharpe_ratio`, `metrics.drawdown.max_drawdown` | direct metrics operations |
| `performance` | `performance.returns.twr`, `performance.cashflows.cashflow_adjusted_twr` | direct performance operations |
| `portfolio` | `portfolio.positions.gross_lev`, `portfolio.transactions.get_turnover` | direct portfolio operations |
| `factor_analysis` | `factor_analysis.data.prepare_factor_data`, `factor_analysis.performance.factor_returns` | direct factor operations |
| `attribution` | `attribution.performance.perf_attrib`, `attribution.brinson.brinson_attribution` | direct attribution operations |
| `risk` | `risk.diagnostics.walk_forward_var`, `risk.backtesting.backtest_var` | direct risk operations |
| `optimization` | `optimization.frontier.efficient_frontier`, `optimization.risk_parity.risk_parity` | direct optimisation operations |
| `simulation` | `simulation.monte_carlo.MonteCarlo`, `simulation.bootstrap.bootstrap` | direct simulation operations |
| `report` | `report.portfolio.compute.build_portfolio_report`, `report.renderers.html.write_html` | report build operations |
| `data`, `extensions`, `viz` | focused provider, snapshot, and renderer modules | integration boundaries |
| `runtime` | `runtime.builtins.builtin_catalog`, `runtime.engine.run` | immutable orchestration only |

## Invariants

1. Every required public leaf capability has one owner and one canonical
   implementation path.
2. `runtime.run` resolves the same callable that direct domain use invokes; it
   adds snapshots, planning, result metadata, and provenance rather than a
   duplicate formula.
3. Reports consume canonical domain results. Renderers do not calculate
   financial metrics.
4. Old package-family modules, root aliases, profile tables, and dynamic
   registries are not public surfaces.

Use [`fincore.runtime.builtins.builtin_catalog`](../../fincore/runtime/builtins.py)
to inspect the registered operation IDs in a particular source or wheel build.
