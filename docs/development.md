# Development Guide

## Environment

This project targets Python 3.11+ (see `pyproject.toml`).

Install editable with dev tooling:

```bash
pip install -e ".[dev]"
```

## Tests

Run unit tests:

```bash
pytest tests/ -q
```

Run tests in parallel:

```bash
pytest tests/ -n auto
```

Integration tests are gated to avoid network dependency in CI:

```bash
FINCORE_RUN_INTEGRATION_TESTS=1 pytest tests/test_data/test_providers_integration.py -q
```

## Linting and Formatting

Ruff:

```bash
ruff check fincore/ tests/
ruff format fincore/ tests/
```

## Type Checking

Mypy is run over selected modules in CI:

```bash
python -m mypy fincore/core fincore/metrics fincore/plugin fincore/data fincore/optimization fincore/attribution fincore/report fincore/risk fincore/simulation fincore/utils fincore/viz fincore/empyrical.py fincore/tearsheets fincore/pyfolio.py --ignore-missing-imports
```

## Coverage

```bash
pytest tests/ --cov=fincore --cov-report=term-missing
```

