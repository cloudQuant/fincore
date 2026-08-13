# fincore | Quantitative Performance & Risk Analytics

**fincore** is a Python library for calculating common financial risk and performance metrics. It continues the empyrical analytics stack under active maintenance by cloudQuant. Current version: **0.3.0** (Beta), Python 3.11+.

## Three API surfaces

- **Strict compatibility** — `fincore.empyrical`: the frozen empyrical 0.6.0 surface (54/54 C0, 49/49 C1, core callables C3).
- **pyfolio façade** — `fincore.pyfolio`: the frozen pyfolio 0.9.6 profile of 11 workflows (C1 all, main chains C4). Requires `fincore[pyfolio]`.
- **Enhanced semantics** — `fincore.metrics`, the flat API, and `AnalysisContext`: fincore's own interfaces with documented divergences.

See [Compatibility](development/compatibility.md) for the full C0–C4 matrix.

## Features

- **150+ Financial Metrics** — returns, risk, drawdown, alpha/beta, capture ratios, timing
- **AnalysisContext** — one-liner `fincore.analyze()` with lazy cached computation
- **RollingEngine** — batch rolling metrics in a single call
- **Pluggable Visualization** — Matplotlib, HTML, Plotly, Bokeh backends
- **Portfolio Optimization** — efficient frontier, risk parity, constrained optimization
- **Monte Carlo Simulation** — bootstrap, scenario testing, path simulation
- **Performance Attribution** — Brinson, Fama-French, style analysis
- **Lazy Imports** — `import fincore` in ~0.04s

## Quick Example

```python
import pandas as pd

import fincore

index = pd.date_range("2024-01-02", periods=5, freq="B")
returns = pd.Series([0.01, -0.005, 0.002, 0.004, -0.001], index=index)
benchmark = pd.Series([0.008, -0.003, 0.001, 0.002, 0.0], index=index)

ctx = fincore.analyze(returns, factor_returns=benchmark)

print(f"Sharpe: {ctx.sharpe_ratio:.4f}")
print(f"Max DD: {ctx.max_drawdown:.4f}")

ctx.to_html(path="report.html")
```

Every example in these docs runs as a real test in `tests/docs/test_examples.py`.

## Installation

```bash
pip install fincore

# With visualization extras
pip install "fincore[pyfolio]"
pip install "fincore[interactive]"

# Everything
pip install "fincore[all]"
```

## Project status

| Metric | Value | Source |
|--------|-------|--------|
| Version | 0.3.0 (Beta) | `pyproject.toml` (single metadata source) |
| Python | 3.11, 3.12, 3.13 | `requires-python = ">=3.11"` |
| Quality numbers | machine-generated | [current-baseline.md](https://github.com/cloudQuant/fincore/blob/master/docs/quality/current-baseline.md) |
| Release readiness | itemized checklist | [release-candidate-checklist.md](https://github.com/cloudQuant/fincore/blob/master/docs/quality/release-candidate-checklist.md) |
| Platforms | macOS, Linux, Windows | CI matrix |
| License | Apache 2.0 | [LICENSE](https://github.com/cloudQuant/fincore/blob/master/LICENSE) |
