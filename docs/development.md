# Development Guide

## Environment

This project targets **Python 3.11+** (see `pyproject.toml`).

Install editable with dev tooling:

```bash
pip install -e ".[dev,viz]"
```

## Tests

The test suite currently has **1800 tests** across multiple modules.

```bash
# Run all tests (parallel by default via pytest-xdist)
pytest tests/

# Run tests sequentially (useful for debugging)
pytest tests/ --override-ini="addopts="

# Run specific test suites
pytest tests/test_empyrical/          # Empyrical metrics tests
pytest tests/test_core/               # AnalysisContext, RollingEngine, Viz tests
pytest tests/test_pyfolio/            # Pyfolio tearsheet tests
pytest tests/test_tearsheets/         # Tearsheet plotting tests
pytest tests/benchmarks/              # Performance benchmarks

# Run a single test
pytest tests/test_core/test_context.py::TestCaching -v
```

Integration tests are gated to avoid network dependency in CI:

```bash
FINCORE_RUN_INTEGRATION_TESTS=1 pytest tests/test_data/test_providers_integration.py -q
```

## Linting and Formatting

[Ruff](https://docs.astral.sh/ruff/) is used for both linting and formatting:

```bash
# Check for lint issues
ruff check fincore/ tests/

# Auto-fix lint issues
ruff check --fix fincore/ tests/

# Format code
ruff format fincore/ tests/
```

Configuration is in `pyproject.toml` under `[tool.ruff]`.

## Type Checking

Mypy is run over selected modules:

```bash
python -m mypy fincore/core fincore/metrics fincore/plugin fincore/data \
    fincore/optimization fincore/attribution fincore/report fincore/risk \
    fincore/simulation fincore/utils fincore/viz fincore/empyrical.py \
    fincore/tearsheets fincore/pyfolio.py --ignore-missing-imports
```

## Coverage

```bash
# Terminal report with missing lines
pytest tests/ --cov=fincore --cov-report=term-missing

# HTML report
pytest tests/ --cov=fincore --cov-report=html
open htmlcov/index.html
```

## Cross-Platform Testing

```bash
# Unix/Linux/macOS
./test_python_versions_simple.sh

# Windows
test_python_versions_simple.bat
```

## Key Architecture Decisions

- **Lazy imports**: `import fincore` loads in ~0.04s; heavy submodules (matplotlib, scipy) are deferred via `__getattr__`
- **Registry-based methods**: `fincore/_registry.py` auto-generates 100+ `Empyrical` class methods via metaclass
- **Star import elimination**: All `__init__.py` files use explicit imports + `__all__`
- **Docstring coverage**: Target is 90%+ across all public modules

