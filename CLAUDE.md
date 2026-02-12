# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

fincore is a Python library for quantitative finance risk and performance analytics. Originally based on empyrical from Quantopian, it continues development under cloudQuant with 150+ financial metrics, lazy-loading architecture, and self-contained HTML/PDF report generation.

## Development Commands

### Installation

```bash
# Install in development mode with dev and visualization dependencies
pip install -e ".[dev,viz]"

# Install all optional dependencies (viz, bayesian, datareader)
pip install -e ".[all]"

# Install from source
pip install -U .
```

### Testing

```bash
# Run all tests (1299 tests) with 4 parallel workers
pytest tests/ -n 4

# Run with coverage
pytest tests/ --cov=fincore --cov-report=term-missing

# Run specific test suites
pytest tests/test_empyrical/          # Empyrical metrics tests
pytest tests/test_core/               # AnalysisContext, RollingEngine tests
pytest tests/test_pyfolio/            # Pyfolio tearsheet tests

# Run a single test
pytest tests/test_core/test_context.py::TestCaching

# Run tests across Python versions (requires conda environments)
./test_python_versions_simple.sh
```

### Linting and Type Checking

```bash
# Ruff linting
ruff check fincore/

# Ruff formatting
ruff format fincore/

# Type checking
mypy fincore/
```

## Architecture Overview

### Top-Level Structure

- **`__init__.py`** - Lazy-loading module with flat API. Common functions (`sharpe_ratio`, `max_drawdown`, etc.) import directly without submodule access. Uses `__getattr__` to defer heavy module loading.
- **`empyrical.py`** - Main facade class with 150+ methods. Uses `_registry.py` to auto-generate methods from metrics modules via `@_populate_from_registry` decorator.
- **`pyfolio.py`** - Pyfolio tearsheet class extending Empyrical with additional plotting functionality.
- **`report.py`** - `create_strategy_report()` for generating HTML/PDF strategy reports with progressive detail based on provided data.
- **`_registry.py`** - Central registry mapping Empyrical method names to underlying metric functions. Eliminates ~1000 lines of boilerplate delegation code.

### Core Components

**`core/context.py`** - AnalysisContext class:
- Lazy, cached metric computation using `@cached_property`
- Recommended entry point via `fincore.analyze(returns, factor_returns=benchmark)`
- Exports: `perf_stats()`, `to_dict()`, `to_json()`, `to_html()`, `plot()`

**`core/engine.py`** - RollingEngine class:
- Batch rolling metric computation in single call
- Available metrics: `sharpe`, `volatility`, `max_drawdown`, `beta`, `sortino`, `mean_return`
- Use when computing multiple rolling metrics to avoid redundant iteration

### Metrics Organization

Located in `fincore/metrics/` with 17+ modules:

| Module | Purpose |
|--------|---------|
| `basic.py` | Utilities (align, annualize, flatten, annualization_factor) |
| `returns.py` | Return calculations (simple_returns, cum_returns, aggregate_returns) |
| `drawdown.py` | Drawdown analytics (max_drawdown, drawdown periods, recovery) |
| `risk.py` | Volatility, VaR, CVaR, downside_risk, tail_ratio |
| `ratios.py` | Sharpe, Sortino, Calmar, Omega, Information, Capture ratios |
| `alpha_beta.py` | Alpha, beta (regular and aligned), capture ratios |
| `rolling.py` | Rolling window metrics with vectorized roll_max_drawdown |
| `stats.py` | Skewness, kurtosis, stability, hurst_exponent, correlation |
| `consecutive.py` | Streaks (max_consecutive_up/down, gain/loss events) |
| `yearly.py` | Annual/breakdown metrics by year |
| `timing.py` | Market-timing measures (treynor_mazuy, henriksson_merton) |
| `positions.py` | Position analysis (exposure, concentration, leverage) |
| `transactions.py` | Transaction analysis (turnover, slippage) |
| `round_trips.py` | Round-trip trade statistics |
| `perf_attrib.py` | Performance attribution and factor decomposition |
| `perf_stats.py` | Aggregated performance statistics |
| `bayesian.py` | Bayesian analysis (requires pymc) |

### Lazy Loading Architecture

The project uses lazy loading at three levels:

1. **Top-level (`__init__.py`)**: `Empyrical`, `Pyfolio`, `analyze`, and flat API functions load on first access via `__getattr__`
2. **Metrics (`metrics/__init__.py`)**: 17 submodules load via `__getattr__` when accessed
3. **Empyrical class**: Methods generated from registry use `_resolve_module()` to lazy-load metric modules

This keeps `import fincore` fast (~0.06s) by avoiding heavy transitive imports.

### Visualization System

Pluggable backend system via `VizBackend` protocol:

- **`viz/base.py`** - Protocol definition and `get_backend(name)` resolver
- **`viz/matplotlib_backend.py`** - Matplotlib implementation (requires matplotlib)
- **`viz/html_backend.py`** - Self-contained HTML reports (no external dependencies)

Backends implement: `plot_returns()`, `plot_drawdown()`, `plot_rolling_sharpe()`, `plot_monthly_heatmap()`

### Constants (`constants/`)

Period constants for annualization:
- `DAILY=252`, `WEEKLY=52`, `MONTHLY=12`, `QUARTERLY=4`, `YEARLY=1`
- `APPROX_BDAYS_PER_YEAR=252`
- `FACTOR_PARTITIONS` dict for period conversion

### Type Definitions

`_types.py` contains centralized type aliases and NamedTuples used across the codebase.

## Key Design Patterns

### Registry-Based Method Generation

The `_registry.py` module contains four registries:
- `STATIC_METHODS` - Utility helpers exposed as static methods
- `CLASSMETHOD_REGISTRY` - Simple forwarding to module functions
- `DUAL_RETURNS_REGISTRY` - Auto-fills `returns` from instance
- `DUAL_RETURNS_FACTOR_REGISTRY` - Auto-fills `returns` AND `factor_returns` from instance

This enables Empyrical class to expose 150+ methods without manual delegation code.

### Dual Method Pattern

The `@_dual_method` descriptor allows methods to work both as class-level calls (passing returns explicitly) and instance calls (auto-filling returns from instance state).

Example:
```python
# Class-level usage
Empyrical.sharpe_ratio(returns, risk_free=0.02)

# Instance-level usage
emp = Empyrical(returns=returns)
emp.sharpe_ratio()  # returns auto-filled from instance
```

## Python Version Support

- **Minimum**: Python 3.11
- **Tested**: 3.11, 3.12, 3.13
- CI runs on all three versions via GitHub Actions

## Test Data

Test fixtures use CSV files in `tests/test_data/`:
- `returns.csv` - Price returns data
- `factor_returns.csv` - Benchmark/factor returns
- `positions.csv`, `factor_loadings.csv`, `residuals.csv` - Performance attribution

`tests/index_data/` contains 100+ global market index data files for testing.

## Optional Dependencies

- **viz**: matplotlib, seaborn, ipython - for matplotlib visualization backend
- **bayesian**: pymc - for Bayesian analysis functions
- **datareader**: pandas-datareader - data fetching utilities

## Important Notes

1. **Annualization**: Most risk-adjusted metrics use annualization factors from constants. Check if metric needs annualization.

2. **NaN Handling**: All functions use custom NaN-aware operations from utils with bottleneck optimization when available.

3. **Return Type**: Library expects simple returns (not log returns) as input.

4. **DataFrame Support**: When DataFrames are passed, metrics calculate for each column independently.

5. **Report Generation**: `create_strategy_report()` generates progressively detailed reports based on provided data (returns → +benchmark → +positions → +transactions → +trades).
