"""Tests for fincore.optimization module.

Covers efficient frontier, risk parity, and constrained optimization.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fincore.optimization import efficient_frontier, optimize, risk_parity


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


class TestEfficientFrontier:
    """Tests for efficient_frontier."""

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


class TestRiskParity:
    """Tests for risk_parity."""

    def test_basic(self, sample_returns):
        rp = risk_parity(sample_returns)
        assert len(rp["weights"]) == 4
        np.testing.assert_almost_equal(np.sum(rp["weights"]), 1.0, decimal=6)
        assert rp["volatility"] > 0

    def test_equal_risk_contribution(self, sample_returns):
        rp = risk_parity(sample_returns)
        rc = rp["risk_contributions"]
        rc_pct = rc / rc.sum()
        # All risk contributions should be ~25% for equal budget
        np.testing.assert_array_less(np.abs(rc_pct - 0.25), 0.05)

    def test_custom_risk_budget(self, sample_returns):
        budget = np.array([0.4, 0.3, 0.2, 0.1])
        rp = risk_parity(sample_returns, risk_budget=budget)
        np.testing.assert_almost_equal(np.sum(rp["weights"]), 1.0, decimal=6)

    def test_weights_positive(self, sample_returns):
        rp = risk_parity(sample_returns)
        assert np.all(rp["weights"] > 0)


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
