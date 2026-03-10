"""Metrics module fixtures for ratio and risk tests.

These fixtures provide specialized test data for metrics testing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def returns_with_benchmark():
    """Returns and benchmark series for capture ratios."""
    np.random.seed(42)
    returns = pd.Series(np.random.randn(252) * 0.01, index=pd.bdate_range("2020-01-01", periods=252))
    benchmark = returns * 0.5
    return returns, benchmark


@pytest.fixture
def fat_tailed_returns():
    """Fat-tailed distribution returns for tail_ratio testing."""
    np.random.seed(99)
    return pd.Series(np.random.standard_t(2, 1000) * 0.01, index=pd.bdate_range("2020-01-01", periods=1000))


@pytest.fixture
def volatility_clustering_returns():
    """Returns with volatility clustering for VaR/CVaR testing."""
    vol = np.concatenate([np.random.randn(50) * v for v in [0.01, 0.02, 0.01, 0.02]])
    return pd.Series(vol, index=pd.bdate_range("2020-01-01", periods=200))


@pytest.fixture
def positive_benchmark():
    """Benchmark with positive returns for up_capture testing."""
    np.random.seed(42)
    return pd.Series(np.random.randn(252) * 0.01 + 0.02, index=pd.bdate_range("2020-01-01", periods=252))


@pytest.fixture
def negative_benchmark():
    """Benchmark with negative returns for down_capture testing."""
    np.random.seed(99)
    return pd.Series(np.random.randn(252) * 0.01 - 0.01, index=pd.bdate_range("2020-01-01", periods=252))


@pytest.fixture
def skewed_returns():
    """Skewed returns for stability testing."""
    np.random.seed(42)
    positive = np.random.randn(126) * 0.01 + 0.05
    negative = np.random.randn(126) * 0.01 - 0.02
    returns = np.concatenate([positive, negative])
    return pd.Series(returns, index=pd.bdate_range("2020-01-01", periods=252))


@pytest.fixture
def stable_returns():
    """Stable returns series for stability testing."""
    np.random.seed(42)
    return pd.Series([0.01] * 252, index=pd.bdate_range("2020-01-01", periods=252))


@pytest.fixture
def extreme_returns():
    """Extreme returns for edge case testing."""
    np.random.seed(42)
    normal = np.random.randn(240) * 0.01
    extreme = np.array([0.5, -0.4, 0.3, -0.35, 0.45, -0.38, 0.25, -0.42, 0.33, -0.36, 0.4])
    returns = np.concatenate([normal, extreme])
    return pd.Series(returns, index=pd.bdate_range("2020-01-01", periods=252))
