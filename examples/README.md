# Examples

This folder contains runnable scripts demonstrating core fincore workflows.

## Quick Start

- `quick_start.py`: demonstrates all 5 API styles (flat API, Empyrical class, Empyrical instance, AnalysisContext, strategy report) in one script.

```bash
python examples/quick_start.py
```

## Report Generation

- `generate_report.py`: generate HTML/PDF strategy reports with increasing amounts of input data.

```bash
python examples/generate_report.py
```

## End-to-End Analysis

- `analyze_with_fincore.py`: load a sample backtest log (from `examples/011_abberation/logs`) and produce a multi-page PDF using Empyrical, Pyfolio, flat API, and viz backends.

```bash
python examples/analyze_with_fincore.py
```

## Risk Models

- `risk_models.py`: EVT (Extreme Value Theory) tail risk analysis + GARCH/EGARCH/GJR-GARCH conditional volatility models.

```bash
python examples/risk_models.py
```

## Portfolio Optimization

- `portfolio_optimization.py`: efficient frontier, risk parity, and constrained optimization (max Sharpe, min variance, target return).

```bash
python examples/portfolio_optimization.py
```

## Monte Carlo Simulation

- `monte_carlo_simulation.py`: Monte Carlo path simulation, bootstrap confidence intervals, and simulation-based VaR/CVaR.

```bash
python examples/monte_carlo_simulation.py
```

## Performance Attribution

- `performance_attribution.py`: Brinson attribution, style analysis, and regression-based factor attribution.

```bash
python examples/performance_attribution.py
```

## Strategy Backtest Analysis

- `011_abberation/`: complete backtest analysis workflow for the AbberationStrategy, including data loading, analysis, and report generation.

