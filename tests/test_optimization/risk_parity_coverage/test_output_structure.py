"""Tests for risk_parity output structure.

Part of test_risk_parity_full_coverage.py split - Output structure tests with P2 markers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fincore.optimization import risk_parity


@pytest.mark.p2
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
