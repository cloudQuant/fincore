# Development guide

## Environment

```bash
pip install -e ".[dev,visualization]"
```

## Verification

```bash
pytest -o addopts='' tests -q --tb=short --maxfail=0
ruff check fincore tests scripts examples benchmarks
ruff format --check fincore tests scripts examples benchmarks
python -m mypy fincore --ignore-missing-imports
python -m mkdocs build --strict
```

Package checks build source-only staging copies so ignored `build/lib` output
cannot contaminate a wheel. Use the canonical operation and domain test suites
when modifying an API; do not add root aliases or compatibility profiles.
