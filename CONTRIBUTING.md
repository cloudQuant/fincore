# Contributing to fincore

Thank you for your interest in contributing to fincore! This guide will help you get started.

## Development Setup

```bash
# Clone the repository
git clone https://github.com/cloudQuant/fincore
cd fincore

# Create a virtual environment (conda recommended)
conda create -n fincore-dev python=3.11
conda activate fincore-dev

# Install with development dependencies
pip install -e ".[dev,viz]"
```

## Running Tests

```bash
# Run all tests (parallel, default)
pytest tests/

# Run tests sequentially (useful for debugging)
pytest tests/ --override-ini="addopts="

# Run specific test modules
pytest tests/test_empyrical/          # Empyrical metrics
pytest tests/test_core/               # AnalysisContext, RollingEngine, Viz
pytest tests/test_pyfolio/            # Pyfolio tearsheets

# Run with coverage
pytest tests/ --cov=fincore --cov-report=term-missing
```

## Code Quality

### Linting & Formatting

We use [Ruff](https://docs.astral.sh/ruff/) for both linting and formatting:

```bash
# Check for lint issues
ruff check fincore/ tests/

# Auto-fix lint issues
ruff check --fix fincore/ tests/

# Format code
ruff format fincore/ tests/
```

### Type Checking

```bash
python -m mypy fincore/core fincore/metrics fincore/plugin fincore/data \
    fincore/optimization fincore/attribution fincore/report fincore/risk \
    fincore/simulation fincore/utils fincore/viz fincore/empyrical.py \
    fincore/tearsheets fincore/pyfolio.py --ignore-missing-imports
```

## Project Structure

```
fincore/
├── __init__.py          # Lazy top-level exports (Empyrical, Pyfolio, analyze)
├── _types.py            # Centralized type aliases & NamedTuples
├── _registry.py         # Method registry for Empyrical metaclass
├── empyrical.py         # Empyrical facade class (150+ methods)
├── pyfolio.py           # Pyfolio tearsheet class (extends Empyrical)
├── constants/           # Period constants (DAILY, WEEKLY, ...)
├── core/
│   ├── context.py       # AnalysisContext — lazy cached metric computation
│   └── engine.py        # RollingEngine — batch rolling metrics
├── metrics/             # 17 metric modules (returns, drawdown, risk, ...)
├── viz/                 # Visualization backends (matplotlib, HTML, plotly, bokeh)
├── tearsheets/          # Legacy plotting functions (used by Pyfolio)
├── attribution/         # Brinson, Fama-French attribution
├── optimization/        # Portfolio optimization (frontier, risk parity)
├── simulation/          # Monte Carlo, bootstrap, scenario simulation
├── risk/                # EVT, GARCH risk models
├── report/              # HTML/PDF report generation
├── plugin/              # Plugin registry and event hooks
└── utils/               # Shared helpers (nanmean, nanstd, ...)
```

## Contribution Workflow

1. **Fork** the repository on GitHub
2. **Create a feature branch**: `git checkout -b feature/my-feature`
3. **Make changes** — follow the coding conventions below
4. **Add tests** for new functionality
5. **Run the full test suite**: `pytest tests/`
6. **Lint your code**: `ruff check fincore/ tests/`
7. **Commit** with a clear message: `git commit -m "Add: new metric X"`
8. **Push** to your fork: `git push origin feature/my-feature`
9. **Open a Pull Request** against the `main` branch

## Coding Conventions

- **Python 3.11+** — use modern syntax (`X | Y` unions, `match`, etc.)
- **Line length**: 120 characters max (enforced by Ruff)
- **Docstrings**: All public functions and classes must have docstrings (NumPy style)
- **Type hints**: Preferred for all public function signatures
- **Imports**: Use explicit imports; no star imports (`from x import *`)
- **Lazy loading**: Heavy dependencies (matplotlib, scipy submodules) should use lazy imports via `__getattr__`

### Docstring Example

```python
def sharpe_ratio(
    returns: pd.Series,
    risk_free: float = 0.0,
    period: str = DAILY,
) -> float:
    """Calculate the annualized Sharpe ratio.

    Parameters
    ----------
    returns : pd.Series
        Non-cumulative simple returns.
    risk_free : float, optional
        Risk-free rate per period. Default 0.0.
    period : str, optional
        Data frequency for annualization. Default ``DAILY``.

    Returns
    -------
    float
        Annualized Sharpe ratio.
    """
```

## Adding a New Metric

1. Add the implementation to the appropriate module in `fincore/metrics/`
2. Export it in `fincore/metrics/__init__.py` if needed
3. Register it in `fincore/_registry.py` (`CLASSMETHOD_REGISTRY`) so `Empyrical.your_metric()` works
4. Add tests in `tests/test_empyrical/` or the relevant test module
5. Update documentation if the metric is user-facing

## Adding a Visualization Backend

1. Create a new file in `fincore/viz/` (e.g., `my_backend.py`)
2. Implement the `VizBackend` protocol from `fincore/viz/base.py`
3. Register it in `fincore/viz/base.py`'s `get_backend()` function
4. Add tests in `tests/test_core/test_viz.py`

## Reporting Issues

- Use GitHub Issues for bug reports and feature requests
- Include a minimal reproducible example when reporting bugs
- Mention your Python version and OS

## License

By contributing, you agree that your contributions will be licensed under the [Apache License 2.0](LICENSE).
