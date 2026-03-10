"""Tests for risk_parity edge cases and validation.

Part of test_risk_parity_full_coverage.py split - Edge case tests with P2 markers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fincore.optimization import risk_parity


@pytest.mark.p2
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
