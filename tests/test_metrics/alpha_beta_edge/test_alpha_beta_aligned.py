"""Tests for alpha_beta_aligned edge cases.

Tests for alpha_beta_aligned with NaN handling and zero variance.
Split from test_alpha_beta_edge_cases.py for maintainability.
"""

import numpy as np
import pandas as pd
import pytest

from fincore.metrics import alpha_beta


@pytest.mark.p2  # Medium: edge case tests
class TestAlphaBetaAlignedEdgeCases:
    """Test alpha_beta_aligned edge cases."""

    def test_alpha_beta_aligned_returns_nan_for_zero_factor_var(self):
        """Test alpha_beta_aligned returns NaN when factor variance is zero (line 423-425)."""
        # Create factor returns with zero variance
        returns = np.array([0.01, 0.02, 0.015])
        factor_returns = np.array([0.01, 0.01, 0.01])  # Zero variance

        result = alpha_beta.alpha_beta_aligned(
            returns,
            factor_returns,
            risk_free=0.0,
        )
        # Should return [nan, nan]
        assert len(result) == 2
        assert np.isnan(result[0])
        assert np.isnan(result[1])

    def test_alpha_beta_aligned_returns_nan_for_all_nan_data(self):
        """Test alpha_beta_aligned returns NaN when all data is NaN."""
        returns = np.array([np.nan, np.nan, np.nan])
        factor_returns = np.array([np.nan, np.nan, np.nan])

        result = alpha_beta.alpha_beta_aligned(
            returns,
            factor_returns,
            risk_free=0.0,
        )
        # Should return [nan, nan]
        assert len(result) == 2
        assert np.isnan(result[0])
        assert np.isnan(result[1])

    def test_alpha_beta_aligned_returns_nan_for_insufficient_clean_data(self):
        """Test alpha_beta_aligned returns NaN when less than 2 valid points remain after NaN removal (line 412-414)."""
        # Create data where only 1 valid point remains after removing NaNs
        returns = np.array([0.01, np.nan, np.nan])
        factor_returns = np.array([0.01, np.nan, np.nan])

        result = alpha_beta.alpha_beta_aligned(
            returns,
            factor_returns,
            risk_free=0.0,
        )
        # Should return [nan, nan]
        assert len(result) == 2
        assert np.isnan(result[0])
        assert np.isnan(result[1])

    def test_alpha_beta_aligned_with_nan_factor_variance(self):
        """Test alpha_beta_aligned returns NaN when factor variance is NaN (line 423-425)."""
        returns = np.array([0.01, np.nan, 0.02])
        factor_returns = np.array([0.01, np.nan, np.nan])

        result = alpha_beta.alpha_beta_aligned(
            returns,
            factor_returns,
            risk_free=0.0,
        )
        # Should return [nan, nan]
        assert len(result) == 2
        assert np.isnan(result[0])
        assert np.isnan(result[1])
