"""Comprehensive performance benchmarks for ALL P0 core metrics.

This module ensures all critical (P0) metrics have performance benchmarks
to detect regressions and maintain acceptable execution times.

Run with: pytest tests/benchmarks/test_p0_metrics_performance.py --benchmark-only

P0 Metrics (Critical):
- sharpe_ratio
- sortino_ratio
- max_drawdown
- annual_return
- annual_volatility
- alpha
- beta
- cum_returns
- cum_returns_final
- value_at_risk
- conditional_value_at_risk

Performance Targets:
- Simple metrics: <5ms for 1000 data points
- Complex metrics: <20ms for 1000 data points
- Rolling metrics: <50ms for 5000 data points
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# Import all P0 metrics
from fincore import (
    sharpe_ratio,
    sortino_ratio,
    max_drawdown,
    annual_return,
    annual_volatility,
    alpha,
    beta,
    cum_returns,
    cum_returns_final,
    value_at_risk,
    calmar_ratio,
    omega_ratio,
    information_ratio,
    tail_ratio,
    downside_risk,
)
from fincore.metrics.risk import conditional_value_at_risk
from fincore.constants import DAILY


# ==============================================================================
# Test Fixtures - Different data sizes
# ==============================================================================


@pytest.fixture
def small_returns():
    """Small dataset: 252 points (1 year)."""
    np.random.seed(42)
    return pd.Series(
        np.random.randn(252) * 0.01,
        index=pd.bdate_range("2023-01-01", periods=252),
    )


@pytest.fixture
def medium_returns():
    """Medium dataset: 1000 points (~4 years)."""
    np.random.seed(42)
    return pd.Series(
        np.random.randn(1000) * 0.01,
        index=pd.bdate_range("2020-01-01", periods=1000),
    )


@pytest.fixture
def large_returns():
    """Large dataset: 5000 points (~20 years)."""
    np.random.seed(42)
    return pd.Series(
        np.random.randn(5000) * 0.01,
        index=pd.bdate_range("2000-01-01", periods=5000),
    )


@pytest.fixture
def factor_returns(medium_returns):
    """Factor returns for alpha/beta testing."""
    np.random.seed(123)
    return pd.Series(
        np.random.randn(len(medium_returns)) * 0.008,
        index=medium_returns.index,
    )


# ==============================================================================
# P0 Risk-Adjusted Return Metrics Benchmarks
# ==============================================================================


@pytest.mark.p0
@pytest.mark.benchmark(group="p0_risk_adjusted")
def test_sharpe_ratio_performance(benchmark, medium_returns):
    """Sharpe ratio: target <5ms for 1000 points."""
    result = benchmark(sharpe_ratio, medium_returns, period=DAILY)
    assert np.isfinite(result)
    # Allow up to 10ms for safety margin
    assert benchmark.stats.stats.median < 0.010


@pytest.mark.p0
@pytest.mark.benchmark(group="p0_risk_adjusted")
def test_sortino_ratio_performance(benchmark, medium_returns):
    """Sortino ratio: target <5ms for 1000 points."""
    result = benchmark(sortino_ratio, medium_returns, period=DAILY)
    assert np.isfinite(result)
    assert benchmark.stats.stats.median < 0.010


@pytest.mark.p0
@pytest.mark.benchmark(group="p0_risk_adjusted")
def test_calmar_ratio_performance(benchmark, medium_returns):
    """Calmar ratio: target <10ms for 1000 points."""
    result = benchmark(calmar_ratio, medium_returns, period=DAILY)
    assert np.isfinite(result)
    assert benchmark.stats.stats.median < 0.015


@pytest.mark.p0
@pytest.mark.benchmark(group="p0_risk_adjusted")
def test_omega_ratio_performance(benchmark, medium_returns):
    """Omega ratio: target <10ms for 1000 points."""
    result = benchmark(omega_ratio, medium_returns)
    assert np.isfinite(result)
    assert benchmark.stats.stats.median < 0.015


@pytest.mark.p0
@pytest.mark.benchmark(group="p0_risk_adjusted")
def test_information_ratio_performance(benchmark, medium_returns, factor_returns):
    """Information ratio: target <10ms for 1000 points."""
    result = benchmark(information_ratio, medium_returns, factor_returns)
    assert np.isfinite(result)
    assert benchmark.stats.stats.median < 0.015


# ==============================================================================
# P0 Drawdown Metrics Benchmarks
# ==============================================================================


@pytest.mark.p0
@pytest.mark.benchmark(group="p0_drawdown")
def test_max_drawdown_performance(benchmark, medium_returns):
    """Max drawdown: target <5ms for 1000 points."""
    result = benchmark(max_drawdown, medium_returns)
    assert result <= 0
    assert benchmark.stats.stats.median < 0.010


# ==============================================================================
# P0 Return Metrics Benchmarks
# ==============================================================================


@pytest.mark.p0
@pytest.mark.benchmark(group="p0_returns")
def test_annual_return_performance(benchmark, medium_returns):
    """Annual return: target <5ms for 1000 points."""
    result = benchmark(annual_return, medium_returns, period=DAILY)
    assert np.isfinite(result)
    assert benchmark.stats.stats.median < 0.010


@pytest.mark.p0
@pytest.mark.benchmark(group="p0_returns")
def test_cum_returns_performance(benchmark, medium_returns):
    """Cumulative returns: target <5ms for 1000 points."""
    result = benchmark(cum_returns, medium_returns)
    assert isinstance(result, pd.Series)
    assert len(result) == len(medium_returns)
    assert benchmark.stats.stats.median < 0.010


@pytest.mark.p0
@pytest.mark.benchmark(group="p0_returns")
def test_cum_returns_final_performance(benchmark, medium_returns):
    """Cumulative returns (final): target <5ms for 1000 points."""
    result = benchmark(cum_returns_final, medium_returns)
    assert np.isfinite(result)
    assert benchmark.stats.stats.median < 0.010


# ==============================================================================
# P0 Volatility Metrics Benchmarks
# ==============================================================================


@pytest.mark.p0
@pytest.mark.benchmark(group="p0_volatility")
def test_annual_volatility_performance(benchmark, medium_returns):
    """Annual volatility: target <5ms for 1000 points."""
    result = benchmark(annual_volatility, medium_returns, period=DAILY)
    assert result > 0
    assert benchmark.stats.stats.median < 0.010


@pytest.mark.p0
@pytest.mark.benchmark(group="p0_volatility")
def test_downside_risk_performance(benchmark, medium_returns):
    """Downside risk: target <5ms for 1000 points."""
    result = benchmark(downside_risk, medium_returns, period=DAILY)
    assert result >= 0
    assert benchmark.stats.stats.median < 0.010


# ==============================================================================
# P0 Alpha/Beta Metrics Benchmarks
# ==============================================================================


@pytest.mark.p0
@pytest.mark.benchmark(group="p0_alpha_beta")
def test_alpha_performance(benchmark, medium_returns, factor_returns):
    """Alpha: target <10ms for 1000 points."""
    result = benchmark(alpha, medium_returns, factor_returns, period=DAILY)
    assert np.isfinite(result)
    assert benchmark.stats.stats.median < 0.015


@pytest.mark.p0
@pytest.mark.benchmark(group="p0_alpha_beta")
def test_beta_performance(benchmark, medium_returns, factor_returns):
    """Beta: target <10ms for 1000 points."""
    result = benchmark(beta, medium_returns, factor_returns)
    assert np.isfinite(result)
    assert benchmark.stats.stats.median < 0.015


# ==============================================================================
# P0 Risk Metrics Benchmarks
# ==============================================================================


@pytest.mark.p0
@pytest.mark.benchmark(group="p0_risk")
def test_value_at_risk_performance(benchmark, medium_returns):
    """Value at Risk (VaR): target <5ms for 1000 points."""
    result = benchmark(value_at_risk, medium_returns, 0.05)
    assert np.isfinite(result)
    assert benchmark.stats.stats.median < 0.010


@pytest.mark.p0
@pytest.mark.benchmark(group="p0_risk")
def test_conditional_value_at_risk_performance(benchmark, medium_returns):
    """Conditional VaR (CVaR): target <5ms for 1000 points."""
    result = benchmark(conditional_value_at_risk, medium_returns, 0.05)
    assert np.isfinite(result)
    assert benchmark.stats.stats.median < 0.010


@pytest.mark.p0
@pytest.mark.benchmark(group="p0_risk")
def test_tail_ratio_performance(benchmark, medium_returns):
    """Tail ratio: target <5ms for 1000 points."""
    result = benchmark(tail_ratio, medium_returns)
    assert result > 0
    assert benchmark.stats.stats.median < 0.010


# ==============================================================================
# Stress Tests - Large Datasets
# ==============================================================================


@pytest.mark.p0
@pytest.mark.benchmark(group="stress_test")
def test_sharpe_ratio_large_dataset(benchmark, large_returns):
    """Sharpe ratio on 5000 points: target <20ms."""
    result = benchmark(sharpe_ratio, large_returns, period=DAILY)
    assert np.isfinite(result)
    assert benchmark.stats.stats.median < 0.025


@pytest.mark.p0
@pytest.mark.benchmark(group="stress_test")
def test_max_drawdown_large_dataset(benchmark, large_returns):
    """Max drawdown on 5000 points: target <20ms."""
    result = benchmark(max_drawdown, large_returns)
    assert result <= 0
    assert benchmark.stats.stats.median < 0.025


@pytest.mark.p0
@pytest.mark.benchmark(group="stress_test")
def test_cum_returns_large_dataset(benchmark, large_returns):
    """Cumulative returns on 5000 points: target <10ms."""
    result = benchmark(cum_returns, large_returns)
    assert isinstance(result, pd.Series)
    assert len(result) == len(large_returns)
    assert benchmark.stats.stats.median < 0.015


# ==============================================================================
# Performance Regression Detection
# ==============================================================================


@pytest.mark.p0
@pytest.mark.benchmark(group="regression_check", min_rounds=10)
def test_sharpe_regression_check(benchmark, medium_returns):
    """Sharpe ratio with 10 rounds for regression detection."""
    result = benchmark(sharpe_ratio, medium_returns, period=DAILY)
    assert np.isfinite(result)
    # Check for consistency
    assert benchmark.stats.stats.stddev < benchmark.stats.stats.median * 0.5


@pytest.mark.p0
@pytest.mark.benchmark(group="regression_check", min_rounds=10)
def test_max_drawdown_regression_check(benchmark, medium_returns):
    """Max drawdown with 10 rounds for regression detection."""
    result = benchmark(max_drawdown, medium_returns)
    assert result <= 0
    # Check for consistency
    assert benchmark.stats.stats.stddev < benchmark.stats.stats.median * 0.5


# ==============================================================================
# DataFrame Performance Tests
# ==============================================================================


@pytest.mark.p0
@pytest.mark.benchmark(group="dataframe")
def test_sharpe_ratio_dataframe(benchmark, medium_returns):
    """Sharpe ratio with DataFrame input."""
    df = pd.DataFrame(
        {
            "strategy1": medium_returns,
            "strategy2": medium_returns * 1.1,
            "strategy3": medium_returns * 0.9,
        }
    )

    result = benchmark(sharpe_ratio, df, period=DAILY)
    assert isinstance(result, (pd.Series, np.ndarray))
    assert len(result) == 3
    assert benchmark.stats.stats.median < 0.020


# ==============================================================================
# Performance Summary
# ==============================================================================

# To generate performance summary:
# pytest tests/benchmarks/test_p0_metrics_performance.py --benchmark-only --benchmark-sort=mean
#
# To save baseline:
# pytest tests/benchmarks/ --benchmark-autosave --benchmark-storage-file=.benchmarks/baseline.json
#
# To compare against baseline:
# pytest tests/benchmarks/ --benchmark-compare=.benchmarks/baseline.json
# --benchmark-compare-fail=mean:10%  # Fail if mean regresses by 10%
