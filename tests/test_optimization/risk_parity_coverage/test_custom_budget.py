"""Tests for risk_parity with custom risk budgets.

Part of test_risk_parity_full_coverage.py split - Custom budget tests with P2 markers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fincore.optimization import risk_parity


@pytest.mark.p2
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
