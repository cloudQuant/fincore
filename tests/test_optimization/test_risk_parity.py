"""Tests for fincore.optimization.risk_parity module.

Part of test_optimization.py split - Risk parity tests with P1 markers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fincore.optimization import risk_parity


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
class TestRiskParity:
    """Tests for risk_parity function."""

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

    def test_invalid_risk_budget_length_raises(self, sample_returns):
        with pytest.raises(ValueError, match="risk_budget length"):
            risk_parity(sample_returns, risk_budget=np.array([0.5, 0.5]))

    def test_negative_risk_budget_raises(self, sample_returns):
        with pytest.raises(ValueError, match="must be non-negative"):
            risk_parity(sample_returns, risk_budget=np.array([0.4, 0.3, -0.1, 0.4]))

    def test_zero_sum_risk_budget_raises(self, sample_returns):
        with pytest.raises(ValueError, match="positive sum"):
            risk_parity(sample_returns, risk_budget=np.array([0.0, 0.0, 0.0, 0.0]))

    def test_invalid_max_iter_raises(self, sample_returns):
        with pytest.raises(ValueError, match="max_iter must be >= 1"):
            risk_parity(sample_returns, max_iter=0)

    def test_returns_with_nan_raises(self, sample_returns):
        bad = sample_returns.copy()
        bad.iloc[0, 0] = np.nan
        with pytest.raises(ValueError, match="contains NaN or infinite values"):
            risk_parity(bad)

    def test_insufficient_observations_raises(self):
        """Test that insufficient observations raises ValueError."""
        np.random.seed(42)
        # Only 1 observation
        df = pd.DataFrame({"A": [0.01], "B": [0.02], "C": [0.03]})
        with pytest.raises(ValueError, match="At least 2 observations"):
            risk_parity(df)

    def test_2d_risk_budget_raises(self, sample_returns):
        """Test that 2D risk_budget raises ValueError."""
        budget_2d = np.array([[0.25, 0.25], [0.25, 0.25]])
        with pytest.raises(ValueError, match="risk_budget must be a 1D array"):
            risk_parity(sample_returns, risk_budget=budget_2d)

    def test_nan_risk_budget_raises(self, sample_returns):
        """Test that NaN in risk_budget raises ValueError."""
        budget_with_nan = np.array([0.25, np.nan, 0.25, 0.25])
        with pytest.raises(ValueError, match="risk_budget contains NaN or infinite values"):
            risk_parity(sample_returns, risk_budget=budget_with_nan)

    def test_inf_risk_budget_raises(self, sample_returns):
        """Test that inf in risk_budget raises ValueError."""
        budget_with_inf = np.array([0.25, np.inf, 0.25, 0.25])
        with pytest.raises(ValueError, match="risk_budget contains NaN or infinite values"):
            risk_parity(sample_returns, risk_budget=budget_with_inf)
