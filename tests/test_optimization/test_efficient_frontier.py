"""Tests for fincore.optimization.efficient_frontier module.

Part of test_optimization.py split - Efficient frontier tests with P1 markers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fincore.optimization import efficient_frontier


@pytest.fixture
def sample_returns():
    """Generate correlated asset returns for testing."""
    np.random.seed(42)
    n_periods = 500
    # 4 assets with different characteristics
    mu = np.array([0.10, 0.15, 0.08, 0.12]) / 252
    corr = np.array(
        [
            [1.0, 0.5, 0.2, 0.3],
            [0.5, 1.0, 0.3, 0.4],
            [0.2, 0.3, 1.0, 0.1],
            [0.3, 0.4, 0.1, 1.0],
        ]
    )
    vols = np.array([0.15, 0.25, 0.10, 0.20]) / np.sqrt(252)
    cov = np.outer(vols, vols) * corr
    rets = np.random.multivariate_normal(mu, cov, size=n_periods)
    return pd.DataFrame(rets, columns=["A", "B", "C", "D"])


@pytest.mark.p1
class TestEfficientFrontier:
    """Tests for efficient_frontier function."""

    def test_basic(self, sample_returns):
        ef = efficient_frontier(sample_returns, n_points=20)

        assert len(ef["frontier_returns"]) == 20
        assert len(ef["frontier_volatilities"]) == 20
        assert ef["frontier_weights"].shape == (20, 4)
        assert ef["asset_names"] == ["A", "B", "C", "D"]

    def test_min_variance_lower_than_max_sharpe_vol(self, sample_returns):
        ef = efficient_frontier(sample_returns, n_points=10)
        assert ef["min_variance"]["volatility"] <= ef["max_sharpe"]["volatility"] + 1e-6

    def test_weights_sum_to_one(self, sample_returns):
        ef = efficient_frontier(sample_returns, n_points=10)
        for i in range(10):
            w = ef["frontier_weights"][i]
            if not np.any(np.isnan(w)):
                np.testing.assert_almost_equal(np.sum(w), 1.0, decimal=6)

    def test_no_short_weights_positive(self, sample_returns):
        ef = efficient_frontier(sample_returns, n_points=10, short_allowed=False)
        for i in range(10):
            w = ef["frontier_weights"][i]
            if not np.any(np.isnan(w)):
                assert np.all(w >= -1e-8)

    def test_frontier_returns_monotonic(self, sample_returns):
        ef = efficient_frontier(sample_returns, n_points=20)
        rets = ef["frontier_returns"]
        valid = ~np.isnan(rets)
        diffs = np.diff(rets[valid])
        assert np.all(diffs >= -1e-6)

    def test_too_few_assets_raises(self):
        df = pd.DataFrame({"A": [0.01, 0.02, 0.03]})
        with pytest.raises(ValueError, match="At least 2 assets"):
            efficient_frontier(df)

    def test_invalid_n_points_raises(self, sample_returns):
        with pytest.raises(ValueError, match="n_points must be >= 2"):
            efficient_frontier(sample_returns, n_points=1)

    def test_returns_with_nan_raises(self, sample_returns):
        bad = sample_returns.copy()
        bad.iloc[0, 0] = np.nan
        with pytest.raises(ValueError, match="contains NaN or infinite values"):
            efficient_frontier(bad, n_points=10)
