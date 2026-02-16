"""Tests for risk_parity module - additional coverage for uncovered lines.

This file tests edge cases and validation paths not covered in test_optimization.py.
"""

import numpy as np
import pandas as pd
import pytest

from fincore.optimization import risk_parity


class TestRiskParityEdgeCases:
    """Test edge cases for risk_parity."""

    def test_empty_dataframe_raises(self):
        """Test that empty DataFrame raises ValueError."""
        with pytest.raises(ValueError, match="must be a non-empty"):
            risk_parity(pd.DataFrame())

    def test_non_dataframe_raises(self):
        """Test that non-DataFrame input raises ValueError."""
        np.random.seed(42)
        data = np.random.randn(100, 4) * 0.01
        with pytest.raises(ValueError, match="must be a non-empty DataFrame"):
            risk_parity(data)

    def test_insufficient_observations_raises(self):
        """Test that less than 2 observations raises ValueError."""
        data = pd.DataFrame([[0.01, 0.02]], columns=["A", "B"])
        with pytest.raises(ValueError, match="At least 2 observations"):
            risk_parity(data)

    def test_nan_values_raises(self):
        """Test that NaN values raise ValueError."""
        np.random.seed(42)
        data = pd.DataFrame(
            [[0.01, np.nan], [0.03, 0.04], [0.02, 0.01]],
            columns=["A", "B"],
        )
        with pytest.raises(ValueError, match="contains NaN or infinite values"):
            risk_parity(data)

    def test_inf_values_raises(self):
        """Test that infinite values raise ValueError."""
        np.random.seed(42)
        data = pd.DataFrame(
            [[0.01, np.inf], [0.03, 0.04], [0.02, 0.01]],
            columns=["A", "B"],
        )
        with pytest.raises(ValueError, match="contains NaN or infinite values"):
            risk_parity(data)

    def test_max_iter_less_than_1_raises(self):
        """Test that max_iter < 1 raises ValueError."""
        np.random.seed(42)
        data = pd.DataFrame(
            np.random.randn(100, 4) * 0.01,
            columns=["A", "B", "C", "D"],
        )
        with pytest.raises(ValueError, match="max_iter must be >= 1"):
            risk_parity(data, max_iter=0)

    def test_max_iter_negative_raises(self):
        """Test that negative max_iter raises ValueError."""
        np.random.seed(42)
        data = pd.DataFrame(
            np.random.randn(100, 4) * 0.01,
            columns=["A", "B", "C", "D"],
        )
        with pytest.raises(ValueError, match="max_iter must be >= 1"):
            risk_parity(data, max_iter=-1)


class TestRiskParityRiskBudgetValidation:
    """Test risk_budget parameter validation."""

    def test_risk_budget_nan_raises(self):
        """Test that risk_budget with NaN raises ValueError."""
        np.random.seed(42)
        data = pd.DataFrame(
            np.random.randn(100, 4) * 0.01,
            columns=["A", "B", "C", "D"],
        )
        budget = np.array([0.25, 0.25, np.nan, 0.25])
        with pytest.raises(ValueError, match="risk_budget contains NaN or infinite"):
            risk_parity(data, risk_budget=budget)

    def test_risk_budget_inf_raises(self):
        """Test that risk_budget with inf raises ValueError."""
        np.random.seed(42)
        data = pd.DataFrame(
            np.random.randn(100, 4) * 0.01,
            columns=["A", "B", "C", "D"],
        )
        budget = np.array([0.25, 0.25, np.inf, 0.25])
        with pytest.raises(ValueError, match="risk_budget contains NaN or infinite"):
            risk_parity(data, risk_budget=budget)

    def test_risk_budget_2d_raises(self):
        """Test that 2D risk_budget raises ValueError."""
        np.random.seed(42)
        data = pd.DataFrame(
            np.random.randn(100, 4) * 0.01,
            columns=["A", "B", "C", "D"],
        )
        budget = np.array([[0.25, 0.25], [0.25, 0.25]])
        with pytest.raises(ValueError, match="risk_budget must be a 1D array"):
            risk_parity(data, risk_budget=budget)

    def test_risk_budget_wrong_length_raises(self):
        """Test that wrong length risk_budget raises ValueError."""
        np.random.seed(42)
        data = pd.DataFrame(
            np.random.randn(100, 4) * 0.01,
            columns=["A", "B", "C", "D"],
        )
        budget = np.array([0.5, 0.5])  # Only 2 elements for 4 assets
        with pytest.raises(ValueError, match="risk_budget length"):
            risk_parity(data, risk_budget=budget)

    def test_risk_budget_negative_raises(self):
        """Test that negative risk_budget raises ValueError."""
        np.random.seed(42)
        data = pd.DataFrame(
            np.random.randn(100, 4) * 0.01,
            columns=["A", "B", "C", "D"],
        )
        budget = np.array([0.3, 0.3, -0.1, 0.3])
        with pytest.raises(ValueError, match="risk_budget must be non-negative"):
            risk_parity(data, risk_budget=budget)

    def test_risk_budget_zero_sum_raises(self):
        """Test that zero-sum risk_budget raises ValueError."""
        np.random.seed(42)
        data = pd.DataFrame(
            np.random.randn(100, 4) * 0.01,
            columns=["A", "B", "C", "D"],
        )
        budget = np.array([0.0, 0.0, 0.0, 0.0])
        with pytest.raises(ValueError, match="risk_budget must have a positive sum"):
            risk_parity(data, risk_budget=budget)


class TestRiskParityWithCustomBudget:
    """Test risk_parity with custom risk budgets."""

    def test_custom_risk_budget_normalized(self):
        """Test that custom risk budget is normalized."""
        np.random.seed(42)
        data = pd.DataFrame(
            np.random.randn(252, 4) * 0.01,
            columns=["A", "B", "C", "D"],
            index=pd.date_range("2020-01-01", periods=252),
        )
        # Provide unnormalized budget
        budget = np.array([4.0, 3.0, 2.0, 1.0])
        rp = risk_parity(data, risk_budget=budget)

        # Should still work and sum to 1
        np.testing.assert_almost_equal(np.sum(rp["weights"]), 1.0, decimal=6)

    def test_asymmetric_risk_budget(self):
        """Test with asymmetric risk budget."""
        np.random.seed(42)
        data = pd.DataFrame(
            np.random.randn(252, 4) * 0.01,
            columns=["A", "B", "C", "D"],
            index=pd.date_range("2020-01-01", periods=252),
        )
        budget = np.array([0.5, 0.3, 0.15, 0.05])
        rp = risk_parity(data, risk_budget=budget)

        # Check that risk contributions align with budget
        rc = rp["risk_contributions"]
        rc_pct = rc / rc.sum()

        # Should be close to the budget (within tolerance)
        np.testing.assert_array_less(np.abs(rc_pct - budget), 0.1)


class TestRiskParityOutputStructure:
    """Test output structure of risk_parity."""

    def test_output_contains_all_keys(self):
        """Test that output contains all expected keys."""
        np.random.seed(42)
        data = pd.DataFrame(
            np.random.randn(252, 4) * 0.01,
            columns=["A", "B", "C", "D"],
            index=pd.date_range("2020-01-01", periods=252),
        )
        rp = risk_parity(data)

        expected_keys = ["weights", "risk_contributions", "volatility", "asset_names"]
        for key in expected_keys:
            assert key in rp

    def test_weights_sum_to_one(self):
        """Test that weights sum to 1."""
        np.random.seed(42)
        data = pd.DataFrame(
            np.random.randn(252, 4) * 0.01,
            columns=["A", "B", "C", "D"],
            index=pd.date_range("2020-01-01", periods=252),
        )
        rp = risk_parity(data)
        np.testing.assert_almost_equal(np.sum(rp["weights"]), 1.0, decimal=6)

    def test_weights_all_positive(self):
        """Test that all weights are positive (long-only)."""
        np.random.seed(42)
        data = pd.DataFrame(
            np.random.randn(252, 4) * 0.01,
            columns=["A", "B", "C", "D"],
            index=pd.date_range("2020-01-01", periods=252),
        )
        rp = risk_parity(data)
        assert np.all(rp["weights"] > 0)

    def test_risk_contributions_sum_to_volatility(self):
        """Test that risk contributions sum to portfolio volatility."""
        np.random.seed(42)
        data = pd.DataFrame(
            np.random.randn(252, 4) * 0.01,
            columns=["A", "B", "C", "D"],
            index=pd.date_range("2020-01-01", periods=252),
        )
        rp = risk_parity(data)

        # Risk contributions should sum to portfolio volatility
        rc_sum = rp["risk_contributions"].sum()
        np.testing.assert_almost_equal(rc_sum, rp["volatility"], decimal=6)

    def test_asset_names(self):
        """Test that asset_names are correctly extracted."""
        np.random.seed(42)
        columns = ["AAPL", "MSFT", "GOOGL", "TSLA"]
        data = pd.DataFrame(
            np.random.randn(252, 4) * 0.01,
            columns=columns,
            index=pd.date_range("2020-01-01", periods=252),
        )
        rp = risk_parity(data)

        assert rp["asset_names"] == columns


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
