# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

fincore is a Python library for quantitative finance risk and performance analytics. Originally based on empyrical from Quantopian, it continues development under cloudQuant with 150+ financial metrics, lazy-loading architecture, and self-contained HTML/PDF report generation. Python 3.11+ required.

## Development Commands

### Installation

```bash
pip install -e ".[dev,viz]"        # Dev mode with dev + visualization deps
pip install -e ".[all]"            # All optional deps (viz, bayesian, datareader)
```

### Testing

```bash
# Default: runs all tests in parallel (pytest-xdist auto), excludes slow/integration/serial
pytest tests/

# Run specific test suites
pytest tests/test_empyrical/
pytest tests/test_core/
pytest tests/test_pyfolio/

# Run a single test
pytest tests/test_core/test_context.py::TestCaching

# By priority marker (P0 = critical core metrics, P1 = high, P2 = medium, P3 = low)
pytest tests/ -m p0
pytest tests/ -m "p0 or p1"

# Include slow/integration tests (excluded by default via addopts)
pytest tests/ -m "slow"
pytest tests/ -m "integration"

# Serial tests (must run without xdist parallelism)
pytest tests/ -m "serial" -n 0

# Coverage (min threshold: 60%)
pytest tests/ --cov=fincore --cov-report=term-missing

# Cross-Python versions (requires conda environments)
./scripts/test_python_versions_simple.sh
```

Note: `pyproject.toml` sets `addopts = "--strict-markers --tb=short -q -ra -n auto --dist=loadscope"`, so parallel execution is the default.

### Linting and Quality

```bash
ruff check fincore/                # Lint (line-length 120, target py311)
ruff format fincore/               # Format
mypy fincore/                      # Type check
bandit -r fincore/ -c pyproject.toml  # Security scan
```

## Architecture Overview

### Entry Points and Facades

- **`__init__.py`** — Lazy-loading module with flat API. Common functions (`sharpe_ratio`, `max_drawdown`, etc.) import directly via `__getattr__` without submodule access.
- **`empyrical.py`** — Main facade class with 150+ methods auto-generated from `_registry.py` via `@_populate_from_registry`. Uses `_dual_method` descriptor so methods work both as class-level calls (passing returns explicitly) and instance calls (auto-filling from instance state).
- **`pyfolio.py`** — Extends Empyrical with pyfolio-style tearsheet plotting.
- **`_registry.py`** — Four registries (`STATIC_METHODS`, `CLASSMETHOD_REGISTRY`, `DUAL_RETURNS_REGISTRY`, `DUAL_RETURNS_FACTOR_REGISTRY`) mapping method names to metric functions, eliminating ~1000 lines of delegation boilerplate.

### Core Components

- **`core/context.py`** — `AnalysisContext`: lazy, cached metric computation via `@cached_property`. Recommended entry point: `fincore.analyze(returns, factor_returns=benchmark)`. Exports `perf_stats()`, `to_dict()`, `to_json()`, `to_html()`, `plot()`.
- **`core/engine.py`** — `RollingEngine`: batch rolling metric computation (sharpe, volatility, max_drawdown, beta, sortino, mean_return) in a single pass.

### Metrics (`fincore/metrics/`)

17 modules organized by domain: `basic`, `returns`, `drawdown`, `risk`, `ratios`, `alpha_beta`, `rolling`, `stats`, `consecutive`, `yearly`, `timing`, `positions`, `transactions`, `round_trips`, `perf_attrib`, `perf_stats`, `bayesian`. See the module-level docstrings for available functions.

### Additional Subpackages

| Package | Purpose |
|---------|---------|
| `tearsheets/` | Pyfolio-style tearsheet plotting (returns, risk, positions, transactions, round_trips, capacity, perf_attrib, bayesian) |
| `viz/` | Pluggable visualization backends via `VizBackend` protocol — matplotlib, HTML (no deps), plotly, bokeh |
| `report/` | HTML/PDF strategy report generation (`create_strategy_report()`) |
| `optimization/` | Portfolio optimization (efficient frontier, risk parity, constrained) |
| `risk/` | Risk models (EVT, GARCH) |
| `simulation/` | Monte Carlo simulation, bootstrap |
| `attribution/` | Performance attribution (Brinson, Fama-French, style analysis) |
| `hooks/` | Event hooks system |
| `plugin/` | Plugin system for extensibility |
| `constants/` | Period constants (`DAILY=252`, `WEEKLY=52`, `MONTHLY=12`, etc.) and interesting periods |
| `utils/` | NaN-aware operations (with bottleneck optimization), alignment utilities |

### Lazy Loading Architecture

Three-level lazy loading keeps `import fincore` fast (~0.04s):

1. **Top-level** (`__init__.py`): `Empyrical`, `Pyfolio`, `analyze`, flat API functions load on first access via `__getattr__`
2. **Metrics** (`metrics/__init__.py`): 17 submodules load via `__getattr__` when accessed
3. **Empyrical class**: Registry methods use `_resolve_module()` to lazy-load metric modules

### Dual Method Pattern

```python
# Class-level: pass returns explicitly
Empyrical.sharpe_ratio(returns, risk_free=0.02)

# Instance-level: returns auto-filled from instance state
emp = Empyrical(returns=returns)
emp.sharpe_ratio()
```

## Key Conventions

1. **Simple returns expected** — The library expects simple returns (not log returns) as input throughout.
2. **NaN handling** — All functions use custom NaN-aware operations from `utils/` with bottleneck optimization when available.
3. **DataFrame support** — When DataFrames are passed, metrics calculate for each column independently.
4. **Annualization** — Most risk-adjusted metrics use annualization factors from `constants/`. Always check if a metric needs an annualization factor.
5. **Report generation** — `create_strategy_report()` generates progressively detailed reports based on provided data (returns -> +benchmark -> +positions -> +transactions -> +trades).
6. **Line length** — 120 characters (configured in ruff).

## Test Data

- `tests/test_data/` — `returns.csv`, `factor_returns.csv`, `positions.csv`, `factor_loadings.csv`, `residuals.csv`
- `tests/index_data/` — 100+ global market index CSV files
