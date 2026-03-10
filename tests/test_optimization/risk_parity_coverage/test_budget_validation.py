"""Tests for risk_parity risk budget validation.

Part of test_risk_parity_full_coverage.py split - Validation tests with P2 markers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fincore.optimization import risk_parity


@pytest.mark.p2
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
