"""Performance benchmarks for fincore.

This module contains benchmarks for measuring performance of key operations.
Run with: pytest tests/benchmarks/ --benchmark-only
"""

import time

import numpy as np
import pandas as pd
import pytest


def test_import_time(benchmark):
    """Benchmark the time to import fincore."""

    def import_fincore():
        # Use subprocess to measure fresh import
        import subprocess
        import sys
        result = subprocess.run(
            [sys.executable, "-c", "import fincore"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Import failed: {result.stderr}")

    # Run multiple times for stable measurement
    benchmark(import_fincore)


def test_analyze_context_creation(benchmark):
    """Benchmark creating an AnalysisContext."""
    import fincore

    # Create sample data
    dates = pd.bdate_range('2020-01-01', periods=2520)
    returns = pd.Series(np.random.normal(0.001, 0.02, 2520), index=dates)
    factor_returns = pd.Series(np.random.normal(0.0005, 0.015, 2520), index=dates)

    def create_context():
        return fincore.analyze(returns, factor_returns=factor_returns)

    result = benchmark(create_context)
    # Verify it worked
    assert hasattr(result, 'sharpe_ratio')


@pytest.mark.parametrize("size", [252, 2520, 25200])
def test_sharpe_ratio_calculation(benchmark, size):
    """Benchmark sharpe_ratio calculation with different data sizes."""
    from fincore import sharpe_ratio

    returns = pd.Series(np.random.normal(0.001, 0.02, size))

    def calc_sharpe():
        return sharpe_ratio(returns)

    benchmark(calc_sharpe)


@pytest.mark.parametrize("size", [252, 2520, 25200])
def test_max_drawdown_calculation(benchmark, size):
    """Benchmark max_drawdown calculation with different data sizes."""
    from fincore import max_drawdown

    returns = pd.Series(np.random.normal(0.001, 0.02, size))

    def calc_max_dd():
        return max_drawdown(returns)

    benchmark(calc_max_dd)


@pytest.mark.parametrize("n_assets", [10, 50, 100])
def test_multi_asset_metrics(benchmark, n_assets):
    """Benchmark calculating metrics for multiple assets simultaneously."""
    from fincore import max_drawdown, sharpe_ratio

    # Create multi-asset returns
    returns = pd.DataFrame(
        np.random.normal(0.001, 0.02, (2520, n_assets)),
        columns=[f"Asset_{i}" for i in range(n_assets)]
    )

    def calc_metrics():
        sharpe = sharpe_ratio(returns)
        mdd = max_drawdown(returns)
        return sharpe, mdd

    benchmark(calc_metrics)


def test_rolling_metrics(benchmark):
    """Benchmark rolling metrics calculation."""
    from fincore.metrics.rolling import roll_sharpe_ratio, roll_max_drawdown

    returns = pd.Series(np.random.normal(0.001, 0.02, 2520))
    window = 252

    def calc_rolling():
        sharpe = roll_sharpe_ratio(returns, window)
        mdd = roll_max_drawdown(returns, window)
        return sharpe, mdd

    benchmark(calc_rolling)


def test_batch_rolling_engine(benchmark):
    """Benchmark RollingEngine for batch rolling metrics."""
    from fincore.core.engine import RollingEngine

    dates = pd.bdate_range('2020-01-01', periods=2520)
    returns = pd.Series(np.random.normal(0.001, 0.02, 2520), index=dates)
    factor_returns = pd.Series(np.random.normal(0.0005, 0.015, 2520), index=dates)

    engine = RollingEngine(returns, factor_returns=factor_returns, window=252)

    def batch_compute():
        return engine.compute(['sharpe', 'volatility', 'max_drawdown', 'beta'])

    result = benchmark(batch_compute)
    # Verify it worked
    assert 'sharpe' in result


def test_performance_stats_generation(benchmark):
    """Benchmark generating full performance statistics."""
    import fincore

    dates = pd.bdate_range('2020-01-01', periods=2520)
    returns = pd.Series(np.random.normal(0.001, 0.02, 2520), index=dates)
    factor_returns = pd.Series(np.random.normal(0.0005, 0.015, 2520), index=dates)

    ctx = fincore.analyze(returns, factor_returns=factor_returns)

    def get_stats():
        return ctx.perf_stats()

    result = benchmark(get_stats)
    # Verify it worked
    assert 'Annual return' in result


@pytest.mark.benchmark(group="memory")
def test_memory_usage_large_dataset(benchmark):
    """Benchmark memory usage with large datasets."""
    import fincore

    # Large dataset: 10 years of hourly data for 100 assets
    n_points = 25200  # ~100 trading days * 252 years
    n_assets = 100

    returns = pd.DataFrame(
        np.random.normal(0.0001, 0.01, (n_points, n_assets)),
        index=pd.date_range('2000-01-01', periods=n_points, freq='h')
    )

    def process_large_data():
        ctx = fincore.analyze(returns.iloc[:, 0])  # Single asset
        stats = ctx.perf_stats()
        return stats

    result = benchmark(process_large_data)
    assert 'Annual return' in result


class TestComparison:
    """Benchmarks comparing different approaches."""

    def test_lazy_vs_eager_import(self, benchmark):
        """Compare lazy import vs full import."""
        import subprocess
        import sys

        # Lazy import (just fincore)
        def lazy_import():
            result = subprocess.run(
                [sys.executable, "-c", "import fincore; print('OK')"],
                capture_output=True,
                text=True,
            )
            return result

        # Full import (all submodules)
        def eager_import():
            result = subprocess.run(
                [sys.executable, "-c",
                 "import fincore; _ = fincore.Empyrical; _ = fincore.Pyfolio; print('OK')"],
                capture_output=True,
                text=True,
            )
            return result

        lazy_result = lazy_import()
        eager_result = eager_import()

        print("\nImport comparison:")
        print(f"  Lazy import completed: {lazy_result.returncode == 0}")
        print(f"  Eager import completed: {eager_result.returncode == 0}")
