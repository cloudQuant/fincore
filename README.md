# fincore | Quantitative Performance & Risk Analytics

<p align="center">
  <img src="https://img.shields.io/badge/version-0.5.0.dev0-blueviolet.svg" alt="Version 0.5.0.dev0"/>
  <img src="https://img.shields.io/badge/status-Beta-orange.svg" alt="Status: Beta"/>
  <img src="https://img.shields.io/badge/python-3.11%2B-brightgreen.svg" alt="Python 3.11+"/>
  <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="MIT License"/>
</p>

<p align="center">
  <a href="https://cloudquant.github.io/fincore/">Documentation</a> ·
  <a href="docs/MIGRATION.md">0.5 migration</a> ·
  <a href="CONTRIBUTING.md">Contributing</a> ·
  <a href="CHANGELOG.md">Changelog</a>
</p>

## What fincore 0.5 is

**fincore** is a unified Python platform for quantitative performance analysis.
It keeps the analytical capabilities historically associated with Empyrical,
Pyfolio, and Alphalens, but rebuilds them as one low-coupling core rather than
three package-shaped APIs. The public contract is organized by domain and each
capability has one canonical implementation path.

Version **0.5.0.dev0** is a deliberately breaking pre-release. It does **not**
provide `fincore.empyrical`, `fincore.pyfolio`, `fincore.alphalens`, flat root
metric functions, compatibility aliases, or façade classes. Update imports to
the focused domain modules described in the [migration guide](docs/MIGRATION.md).

| Domain | Use it for | Canonical examples |
| --- | --- | --- |
| `fincore.metrics` | returns, drawdown, ratios, rolling and statistical metrics | `metrics.ratios.sharpe_ratio` |
| `fincore.performance` | cash-flow-aware return calculation and performance inference | `performance.cashflows.cashflow_adjusted_twr` |
| `fincore.portfolio` | positions, transactions, capacity, and round trips | `portfolio.positions.gross_lev` |
| `fincore.report` | immutable report documents and HTML/PDF/XLSX renderers | `report.portfolio.compute.build_portfolio_report` |
| `fincore.factor_analysis` | factor preparation, analysis, inference, cost, and rendering | `factor_analysis.analysis.analyze_factor` |
| `fincore.risk` | risk models, diagnostics, calibration, EVT, and GARCH | `risk.diagnostics.walk_forward_var` |
| `fincore.attribution` | allocation and factor-performance attribution | `attribution.performance.perf_attrib` |
| `fincore.optimization`, `simulation`, `data`, `viz`, `extensions`, `runtime` | specialized analysis and platform services | import the owning leaf module |

The package root is only a namespace index. Import executable functions and
models from their owning leaf module, not from `fincore` itself.

## Install

```bash
pip install fincore

# Optional capabilities
pip install "fincore[factor-analysis]"
pip install "fincore[visualization]"
pip install "fincore[interactive]"
pip install "fincore[report-pdf]"
pip install "fincore[report-xlsx]"
pip install "fincore[bayesian]"
pip install "fincore[data-yahoo]"
pip install "fincore[data-alphavantage]"
pip install "fincore[data-pandas-datareader]"
pip install "fincore[data-cn]"
pip install "fincore[all]"
```

For a source checkout, use `pip install -e ".[dev]"`. Python **3.11+** is
required. `pyproject.toml` is the dependency source of truth.

## Quick start: metrics

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

## Portfolio report workflow

Compute a portable report model once, then choose a renderer. Renderers do not
repeat analytical computation.

```python
import pandas as pd

from fincore.report.portfolio.compute import build_portfolio_report
from fincore.report.renderers.html import write_html

index = pd.date_range("2024-01-02", periods=5, freq="B")
returns = pd.Series([0.01, -0.005, 0.002, 0.004, -0.001], index=index)
positions = pd.DataFrame({"AAA": 100.0, "BBB": -30.0, "cash": 80.0}, index=index)

document = build_portfolio_report(returns, positions=positions, rolling_window=3)
artifacts = write_html(document, "portfolio-report.html")
print(artifacts.named_artifacts["file"])
```

Use `fincore[report-pdf]` or `fincore[report-xlsx]` only when selecting the
corresponding renderer.

## Factor analysis

Factor analysis is a first-class domain, not an Alphalens wrapper. The
repository includes an offline deterministic quickstart that prepares inputs,
computes the canonical model, derives portfolio inputs, and renders a headless
summary:

```bash
pip install "fincore[visualization]"
MPLBACKEND=Agg python examples/factor_analysis_quickstart.py
```

For application code, import the exact operation from its owning module, such
as `fincore.factor_analysis.data`, `analysis`, `performance`, `portfolio`, or
`inference`.

## Design commitments

- One implementation path per public operation; no re-exported legacy API shells.
- Immutable input and extension snapshots at domain boundaries.
- Structured runtime errors with operation and parameter context.
- Report models separated from renderers and artifact lifecycle management.
- Optional dependencies isolated by explicit capability extras.
- Tests protect canonical imports, removed legacy surfaces, report semantics,
  package contents, and executable documentation.

## Development and verification

```bash
/Users/yunjinqi/opt/anaconda3/bin/conda run -n base python -m pytest -o addopts='' tests/docs tests/packaging -q
/Users/yunjinqi/opt/anaconda3/bin/conda run -n base python -m ruff check fincore scripts tests
```

The release-quality and provenance records live under `docs/quality/`. A local
test pass is not a published-release claim.

## License and third-party notices

fincore is MIT licensed; see [LICENSE](LICENSE). Retained third-party material
keeps its own licensing and attribution in [NOTICE](NOTICE),
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), and
[THIRD_PARTY_LICENSES](THIRD_PARTY_LICENSES/).
