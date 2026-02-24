# Development Guide

## Environment

```bash
pip install -e ".[dev,viz]"
```

## Tests

```bash
# All tests (parallel)
pytest tests/

# Sequential (debugging)
pytest tests/ --override-ini="addopts="

# With coverage
pytest tests/ --cov=fincore --cov-report=term-missing
```

## Linting

```bash
ruff check fincore/ tests/
ruff format fincore/ tests/
```

## Type Checking

```bash
python -m mypy fincore/core fincore/metrics fincore/plugin fincore/data \
    fincore/optimization fincore/attribution fincore/report fincore/risk \
    fincore/simulation fincore/utils fincore/viz fincore/empyrical.py \
    fincore/tearsheets fincore/pyfolio.py --ignore-missing-imports
```

## Architecture

- **Lazy imports**: `import fincore` ~0.04s
- **Registry-based methods**: `fincore/_registry.py` auto-generates 100+ `Empyrical` class methods
- **Star import elimination**: All `__init__.py` use explicit imports + `__all__`
