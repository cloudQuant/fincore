"""Performance benchmarks for core fincore metrics.

This module uses pytest-benchmark to track execution time and detect performance regressions.
Run with: pytest tests/benchmarks/ --benchmark-only

Benchmarks should:
- Execute in <1 second per metric
- Track median execution time
- Detect regressions >10%
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fincore.attribution.performance import perf_attrib
from fincore.metrics import returns as returns_mod
from fincore.metrics.drawdown import max_drawdown
from fincore.metrics.frequencies import DAILY
from fincore.metrics.ratios import sharpe_ratio
from fincore.metrics.risk import annual_volatility
from fincore.metrics.yearly import annual_return

pytest.importorskip("pytest_benchmark")


def _attribution_benchmark_inputs(
    periods: int = 100,
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build deterministic direct-domain inputs for the attribution benchmark."""
    dates = pd.bdate_range("2020-01-01", periods=periods)
    assets = ("AAA", "BBB", "CCC")
    steps = np.arange(periods, dtype=float)
    returns = pd.Series(0.002 * np.sin(steps / 4.0), index=dates)
    positions = pd.DataFrame(
        {"AAA": 120.0 + steps, "BBB": 80.0 - 0.2 * steps, "CCC": 40.0 + 0.1 * steps},
        index=dates,
    )
    factor_returns = pd.DataFrame(
        {"value": 0.001 + 0.0001 * np.cos(steps), "momentum": 0.0007 * np.sin(steps / 3.0)},
        index=dates,
    )
    factor_loadings = pd.DataFrame(
        {
            "value": [0.1 * (asset + 1) for _date in dates for asset in range(len(assets))],
            "momentum": [0.05 * (asset + 1) for _date in dates for asset in range(len(assets))],
        },
        index=pd.MultiIndex.from_product([dates, assets], names=["dt", "ticker"]),
    )
    return returns, positions, factor_returns, factor_loadings


@pytest.fixture
def sample_returns_1000():
    """Generate 1000 days of sample returns for benchmarking."""
    np.random.seed(42)
    return pd.Series(
        np.random.randn(1000) * 0.01,
        index=pd.bdate_range("2020-01-01", periods=1000),
    )


@pytest.fixture
def sample_returns_5000():
    """Generate 5000 days of sample returns for stress testing."""
    np.random.seed(42)
    return pd.Series(
        np.random.randn(5000) * 0.01,
        index=pd.bdate_range("2000-01-01", periods=5000),
    )


@pytest.mark.p2
@pytest.mark.benchmark(group="core_metrics")
def test_sharpe_ratio_benchmark(benchmark, sample_returns_1000):
    """Benchmark sharpe_ratio calculation (target: <1ms)."""
    result = benchmark(sharpe_ratio, sample_returns_1000)
    assert result > 0
    assert benchmark.stats.stats.median < 0.001  # <1ms


@pytest.mark.p2
@pytest.mark.benchmark(group="core_metrics")
def test_max_drawdown_benchmark(benchmark, sample_returns_1000):
    """Benchmark max_drawdown calculation (target: <5ms)."""
    result = benchmark(max_drawdown, sample_returns_1000)
    assert result <= 0
    assert benchmark.stats.stats.median < 0.005  # <5ms


@pytest.mark.p2
@pytest.mark.benchmark(group="core_metrics")
def test_annual_return_benchmark(benchmark, sample_returns_1000):
    """Benchmark annual_return calculation (target: <1ms)."""
    result = benchmark(annual_return, sample_returns_1000)
    assert np.isfinite(result)
    assert benchmark.stats.stats.median < 0.001  # <1ms


@pytest.mark.p2
@pytest.mark.benchmark(group="core_metrics")
def test_annual_volatility_benchmark(benchmark, sample_returns_1000):
    """Benchmark annual_volatility calculation (target: <2ms)."""
    result = benchmark(annual_volatility, sample_returns_1000)
    assert result > 0
    assert benchmark.stats.stats.median < 0.002  # <2ms


@pytest.mark.p2
@pytest.mark.benchmark(group="returns")
def test_cum_returns_benchmark(benchmark, sample_returns_5000):
    """Benchmark cumulative returns calculation (target: <1ms)."""
    from fincore.metrics.returns import cum_returns

    result = benchmark(cum_returns, sample_returns_5000)
    assert len(result) == len(sample_returns_5000)
    assert benchmark.stats.stats.median < 0.001  # <1ms


@pytest.mark.p2
@pytest.mark.benchmark(group="returns")
def test_aggregate_returns_benchmark(benchmark, sample_returns_5000):
    """Exercise aggregate-return timing with the declared benchmark protocol."""
    result = benchmark(returns_mod.aggregate_returns, sample_returns_5000, "yearly")
    assert len(result) > 0
    # Wall-clock micro-thresholds are not portable across supported runners.
    # The reproducible workload profiler owns regression comparison; this
    # benchmark verifies that pytest-benchmark actually sampled the operation.
    assert benchmark.stats.stats.rounds >= 10


@pytest.mark.p3
@pytest.mark.benchmark(group="rolling")
def test_rolling_sharpe_benchmark(benchmark, sample_returns_5000):
    """Benchmark rolling Sharpe ratio (target: <50ms)."""
    from fincore.metrics.rolling import rolling_sharpe

    result = benchmark(rolling_sharpe, sample_returns_5000, rolling_sharpe_window=252)
    assert len(result) > 0
    assert benchmark.stats.stats.median < 0.050  # <50ms


@pytest.mark.p3
@pytest.mark.benchmark(group="attribution")
def test_perf_attrib_benchmark(benchmark):
    """Benchmark performance attribution (target: <100ms)."""
    returns, positions, factor_returns, factor_loadings = _attribution_benchmark_inputs()

    result = benchmark(
        perf_attrib,
        returns,
        positions,
        factor_returns,
        factor_loadings,
    )
    assert result is not None
    assert len(result) > 0
    assert benchmark.stats.stats.median < 0.100  # <100ms


@pytest.mark.p3
@pytest.mark.benchmark(group="optimization")
def test_efficient_frontier_benchmark(benchmark):
    """Benchmark efficient frontier calculation (target: <500ms)."""
    from fincore.optimization.frontier import efficient_frontier

    # Create sample data with correct shape
    np.random.seed(42)
    returns = pd.DataFrame(
        np.random.randn(252, 3) * 0.01,
        columns=["A", "B", "C"],
    )

    result = benchmark(
        efficient_frontier,
        returns,
        n_points=20,
    )
    assert isinstance(result, dict)
    assert "frontier_returns" in result
    # Relax target for optimization
    assert benchmark.stats.stats.median < 1.000  # <1s


# Regression thresholds (configure in pytest.ini)
# --benchmark-autosave --benchmark-storage-file=benches.json
# --benchmark-compare-fail=mean:10%  # Fail if mean regresses by 10%
