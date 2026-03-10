# Performance Benchmarks

This document describes the performance benchmark suite for fincore.

## Overview

The benchmark suite measures execution time for critical financial metrics to detect performance regressions. All benchmarks are automated via pytest-benchmark.

## Running Benchmarks

### Local Development

```bash
# Run all benchmarks (output JSON for comparison)
pytest tests/benchmarks/ tests/test_import_time.py \
  --benchmark-only \
  --benchmark-json=benchmark-results.json \
  -n 0  # Disable parallel execution

# Run specific benchmark group
pytest tests/benchmarks/ --benchmark-only -k "core_metrics" -n 0

# Compare against previous run
pytest tests/benchmarks/ --benchmark-only --benchmark-compare \
  --benchmark-compare-fail=mean:10%  # Fail if mean regresses by 10%
```

### CI/CD

The `ci-enhanced.yml` workflow automatically runs benchmarks on every PR and push:

- **Regression Threshold**: 125% (alert if 25% slower)
- **Fail-on-alert**: Enabled for main branch
- **Storage**: Benchmarks cached and compared across runs

## Benchmark Groups

| Group | Description | Tests | Target |
|-------|-------------|-------|--------|
| `core_metrics` | Sharpe, max drawdown, annual return/volatility | 4 | <50µs |
| `returns` | Cumulative returns, aggregation | 2 | <10ms |
| `rolling` | Rolling Sharpe ratio | 1 | <50ms |
| `attribution` | Performance attribution | 1 | <100ms |
| `optimization` | Efficient frontier | 1 | <1s |
| `import_time` | Module import validation | 1 | <100ms |
| `memory` | Large dataset memory usage | 1 | <10ms |

## Current Performance (2026-02-25)

```
Core Metrics:
  - sharpe_ratio:        ~15µs
  - max_drawdown:        ~31µs
  - annual_return:       ~35µs
  - annual_volatility:   ~11µs

Returns:
  - cum_returns:         ~107µs
  - aggregate_returns:   ~7ms (yearly aggregation)

Rolling:
  - rolling_sharpe:      ~200µs (252-window)

Attribution:
  - perf_attrib:         ~2.3ms

Optimization:
  - efficient_frontier:  ~20ms (20 points, 3 assets)

Import:
  - import fincore:      ~0.27ms (target: <100ms) ✅
```

## Regression Detection

When a benchmark exceeds the regression threshold:

1. **CI**: Fails the build with detailed comparison
2. **PR**: Automatic comment with performance delta
3. **Local**: Use `--benchmark-compare` to see changes

Example output:
```
FAILED test_sharpe_ratio_benchmark
  Regression: 1.35x slower (15.0µs → 20.2µs)
  Threshold: 1.25x
```

## Adding New Benchmarks

```python
import pytest

@pytest.mark.p2
@pytest.mark.benchmark(group="your_group")
def test_your_benchmark(benchmark, sample_data):
    """Benchmark description (target: <Xms)."""
    result = benchmark(your_function, sample_data)
    assert result is not None
    assert benchmark.stats.stats.median < 0.001  # <1ms
```

## Configuration

Benchmark settings in `pyproject.toml`:

```toml
[tool.benchmark]
min_rounds = 5
max_time = 1.0
calibration_precision = 10
timer = "time.perf_counter"
```

## References

- [pytest-benchmark documentation](https://pytest-benchmark.readthedocs.io/)
- CI workflow: `.github/workflows/ci-enhanced.yml`
- Benchmark tests: `tests/benchmarks/`
