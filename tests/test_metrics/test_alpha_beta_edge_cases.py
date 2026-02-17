"""Tests for alpha_beta module - edge cases for full coverage."""

import numpy as np
import pandas as pd
import pytest

from fincore.metrics import alpha_beta


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


class TestAnnualAlphaEdgeCases:
    """Test annual_alpha edge cases."""

    def test_annual_alpha_returns_empty_series_for_empty_input(self):
        """Test annual_alpha returns empty Series for empty input (line 530, 537)."""
        returns = pd.Series([], dtype=float)
        factor_returns = pd.Series([], dtype=float)

        result = alpha_beta.annual_alpha(returns, factor_returns)
        assert isinstance(result, pd.Series)
        assert len(result) == 0

    def test_annual_alpha_returns_empty_series_for_non_datetime_index(self):
        """Test annual_alpha returns empty Series for non-DatetimeIndex (line 532-533)."""
        returns = pd.Series([0.01, 0.02, 0.015])
        factor_returns = pd.Series([0.005, 0.01, 0.008])

        result = alpha_beta.annual_alpha(returns, factor_returns)
        assert isinstance(result, pd.Series)
        assert len(result) == 0


class TestAnnualBetaEdgeCases:
    """Test annual_beta edge cases."""

    def test_annual_beta_returns_empty_series_for_empty_input(self):
        """Test annual_beta returns empty Series for empty input (line 582-583, 590)."""
        returns = pd.Series([], dtype=float)
        factor_returns = pd.Series([], dtype=float)

        result = alpha_beta.annual_beta(returns, factor_returns)
        assert isinstance(result, pd.Series)
        assert len(result) == 0

    def test_annual_beta_returns_empty_series_for_non_datetime_index(self):
        """Test annual_beta returns empty Series for non-DatetimeIndex (line 585-586)."""
        returns = pd.Series([0.01, 0.02, 0.015])
        factor_returns = pd.Series([0.005, 0.01, 0.008])

        result = alpha_beta.annual_beta(returns, factor_returns)
        assert isinstance(result, pd.Series)
        assert len(result) == 0
