# fincore | Quantitative Performance & Risk Analytics

**fincore** is a Python library for calculating common financial risk and performance metrics. It continues the empyrical analytics stack under active maintenance by cloudQuant.

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
import fincore

ctx = fincore.analyze(returns, factor_returns=benchmark)

print(f"Sharpe: {ctx.sharpe_ratio:.4f}")
print(f"Max DD: {ctx.max_drawdown:.4f}")
print(f"Alpha:  {ctx.alpha:.6f}")

ctx.to_html(path="report.html")
```

## Installation

```bash
pip install fincore

# With visualization
pip install "fincore[viz]"

# Everything
pip install "fincore[all]"
```

## Project Stats

| Metric | Value |
|--------|-------|
| Source files | 85 Python files, ~26,700 lines |
| Tests | 1,800 passing |
| Docstring coverage | 92% |
| Python versions | 3.11, 3.12, 3.13 |
| Platforms | macOS, Linux, Windows |
| License | Apache 2.0 |
