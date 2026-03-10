"""Tests for risk_parity objective function behavior.

Part of test_risk_parity_full_coverage.py split - Objective function tests with P2 markers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fincore.optimization import risk_parity


@pytest.mark.p2
class TestRiskParityObjectiveFunction:
    """Test internal objective function behavior."""

    def test_low_variance_portfolio(self):
        """Test with a portfolio that has very low variance."""
        np.random.seed(42)
        # Create data with very low volatility
        data = pd.DataFrame(
            np.random.randn(252, 4) * 0.0001,
            columns=["A", "B", "C", "D"],
            index=pd.date_range("2020-01-01", periods=252),
        )
        rp = risk_parity(data)

        # Should still converge
        assert np.all(np.isfinite(rp["weights"]))
        np.testing.assert_almost_equal(np.sum(rp["weights"]), 1.0, decimal=6)

    def test_high_variance_portfolio(self):
        """Test with a portfolio that has high variance."""
        np.random.seed(42)
        # Create data with high volatility
        data = pd.DataFrame(
            np.random.randn(252, 4) * 0.1,
            columns=["A", "B", "C", "D"],
            index=pd.date_range("2020-01-01", periods=252),
        )
        rp = risk_parity(data)

        # Should still converge
        assert np.all(np.isfinite(rp["weights"]))
        np.testing.assert_almost_equal(np.sum(rp["weights"]), 1.0, decimal=6)

    def test_zero_variance_portfolio_line_78(self):
        """Test when portfolio variance is essentially zero (line 78)."""
        # Create returns that are essentially all zeros
        # This will cause port_var < 1e-16 in _risk_contrib
        np.random.seed(42)
        data = pd.DataFrame(
            np.zeros((252, 4)),
            columns=["A", "B", "C", "D"],
            index=pd.date_range("2020-01-01", periods=252),
        )
        # Add tiny noise that still results in near-zero variance
        data = data + np.random.randn(252, 4) * 1e-20

        rp = risk_parity(data)

        # Should still converge with equal weights
        assert np.all(np.isfinite(rp["weights"]))
        np.testing.assert_almost_equal(np.sum(rp["weights"]), 1.0, decimal=6)

    def test_zero_risk_contribution_line_88(self):
        """Test when total risk contribution is essentially zero (line 88)."""
        # Create data where assets have zero correlation and very low variance
        # This can cause rc_total < 1e-16 in _objective
        np.random.seed(42)
        # Create independent assets with tiny variance
        cov = np.eye(4) * 1e-20
        means = np.zeros(4)
        data = np.random.multivariate_normal(means, cov, 252)
        data = pd.DataFrame(
            data,
            columns=["A", "B", "C", "D"],
            index=pd.date_range("2020-01-01", periods=252),
        )

        rp = risk_parity(data)

        # Should still converge
        assert np.all(np.isfinite(rp["weights"]))
        np.testing.assert_almost_equal(np.sum(rp["weights"]), 1.0, decimal=6)
