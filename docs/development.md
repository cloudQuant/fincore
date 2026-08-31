# Development Guide

## Environment

fincore targets Python 3.11+. Install the editable source with development and
visualisation dependencies:

```bash
pip install -e ".[dev,visualization]"
```

## Tests

The source tree uses canonical-domain tests, runtime contracts, packaging
checks, documentation examples, and benchmark regressions.

```bash
# Full suite, overriding local parallel defaults when diagnosis needs one process
pytest -o addopts='' tests -q --tb=short --maxfail=0

# Focused domains
pytest -o addopts='' tests/test_metrics tests/portfolio tests/factor_analysis -q
pytest -o addopts='' tests/runtime tests/parity tests/packaging tests/docs -q
pytest -o addopts='' tests/benchmarks -q
```

The local performance budget gate is part of the canonical runtime contract:

```bash
python scripts/check_performance.py
```

## Quality tools

```bash
ruff check fincore tests scripts examples benchmarks .github
ruff format --check fincore tests scripts examples benchmarks .github
python -m mkdocs build --strict
```

## Architecture rules

- Each public capability has one owning leaf implementation and one registered
  `operation_id`.
- Domain kernels do not depend on report rendering, runtime orchestration, or
  compatibility-era support packages.
- `runtime` provides immutable snapshots, catalog composition, execution
  records, and extension boundaries; it does not duplicate domain formulas.
- Report builders compute models once; renderer modules only project those
  models and manage artifacts.
- Optional dependencies are capability-oriented extras, never package-family
  aliases.

Use the current [API map](architecture/public-api-map.md) and
[migration guide](MIGRATION.md) when moving a capability between modules.
