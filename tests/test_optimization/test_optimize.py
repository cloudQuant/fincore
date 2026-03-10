"""Tests for fincore.optimization.optimize module.

Part of test_optimization.py split - Optimize function tests with P1 markers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fincore.optimization import optimize


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
class TestOptimize:
    """Tests for the optimize function."""

    def test_max_sharpe(self, sample_returns):
        res = optimize(sample_returns, objective="max_sharpe")
        np.testing.assert_almost_equal(np.sum(res["weights"]), 1.0, decimal=6)
        assert res["sharpe"] > 0
        assert res["objective"] == "max_sharpe"

    def test_min_variance(self, sample_returns):
        res_mv = optimize(sample_returns, objective="min_variance")
        res_ms = optimize(sample_returns, objective="max_sharpe")
        assert res_mv["volatility"] <= res_ms["volatility"] + 1e-6

    def test_target_return(self, sample_returns):
        target = 0.10  # 10% annual
        res = optimize(sample_returns, objective="target_return", target_return=target)
        np.testing.assert_almost_equal(res["return"], target, decimal=3)

    def test_target_risk(self, sample_returns):
        target = 0.15  # 15% annual vol
        res = optimize(sample_returns, objective="target_risk", target_volatility=target)
        np.testing.assert_almost_equal(res["volatility"], target, decimal=3)

    def test_unknown_objective_raises(self, sample_returns):
        with pytest.raises(ValueError, match="Unknown objective"):
            optimize(sample_returns, objective="nonsense")

    def test_target_return_missing_raises(self, sample_returns):
        with pytest.raises(ValueError, match="target_return must be specified"):
            optimize(sample_returns, objective="target_return")

    def test_target_risk_missing_raises(self, sample_returns):
        with pytest.raises(ValueError, match="target_volatility must be specified"):
            optimize(sample_returns, objective="target_risk")

    def test_no_short_weights_positive(self, sample_returns):
        res = optimize(sample_returns, objective="max_sharpe", short_allowed=False)
        assert np.all(res["weights"] >= -1e-8)

    def test_returns_with_nan_raises(self, sample_returns):
        bad = sample_returns.copy()
        bad.iloc[0, 0] = np.nan
        with pytest.raises(ValueError, match="contains NaN or infinite values"):
            optimize(bad, objective="max_sharpe")

    def test_min_weight_greater_than_max_weight_raises(self, sample_returns):
        with pytest.raises(ValueError, match="min_weight must be <= max_weight"):
            optimize(sample_returns, objective="max_sharpe", min_weight=0.6, max_weight=0.5)

    def test_sector_constraints_without_map_raises(self, sample_returns):
        with pytest.raises(ValueError, match="sector_map is required"):
            optimize(sample_returns, objective="max_sharpe", sector_constraints={"Tech": (0.1, 0.9)})
