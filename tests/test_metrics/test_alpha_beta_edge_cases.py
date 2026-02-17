"""Tests for alpha_beta module - edge cases for full coverage."""

import numpy as np
import pandas as pd
import pytest

from fincore.metrics import alpha_beta


class TestAlphaEdgeCases:
    """Test alpha function edge cases."""

    def test_alpha_with_returns_dataframe(self):
        """Test alpha when returns is a DataFrame."""
        returns = pd.DataFrame(
            {
                "asset1": [0.01, 0.02, 0.015, 0.008, 0.012],
                "asset2": [0.008, 0.015, 0.012, 0.006, 0.010],
            }
        )
        factor_returns = pd.Series([0.005, 0.01, 0.008, 0.004, 0.006])

        result = alpha_beta.alpha(returns, factor_returns)
        assert isinstance(result, pd.Series)
        assert len(result) == 2


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


class TestAlphaPercentileRankEdgeCases:
    """Test alpha_percentile_rank edge cases."""

    def test_alpha_percentile_rank_returns_nan_for_insufficient_data(self):
        """Test alpha_percentile_rank returns NaN for strategy with less than 3 returns (line 639-640, 645)."""
        strategy_returns = pd.Series([0.01, 0.02])
        all_strategies_returns = [
            pd.Series([0.01, 0.02, 0.015]),
        ]
        factor_returns = pd.Series([0.005, 0.01, 0.008])

        result = alpha_beta.alpha_percentile_rank(
            strategy_returns,
            all_strategies_returns,
            factor_returns,
        )
        assert np.isnan(result)

    def test_alpha_percentile_rank_returns_nan_for_nan_strategy_alpha(self):
        """Test alpha_percentile_rank returns NaN when strategy alpha is NaN (line 644-645)."""
        strategy_returns = pd.Series([np.nan, np.nan, np.nan])
        all_strategies_returns = [
            pd.Series([0.01, 0.02, 0.015]),
        ]
        factor_returns = pd.Series([0.005, 0.01, 0.008])

        result = alpha_beta.alpha_percentile_rank(
            strategy_returns,
            all_strategies_returns,
            factor_returns,
        )
        assert np.isnan(result)

    def test_alpha_percentile_rank_returns_nan_for_no_valid_peer_alphas(self):
        """Test alpha_percentile_rank returns NaN when all peer alphas are NaN or insufficient (line 650, 656)."""
        strategy_returns = pd.Series([0.01, 0.02, 0.015])
        all_strategies_returns = [
            pd.Series([np.nan, np.nan, np.nan]),
            pd.Series([0.01]),  # Too short
        ]
        factor_returns = pd.Series([0.005, 0.01, 0.008])

        result = alpha_beta.alpha_percentile_rank(
            strategy_returns,
            all_strategies_returns,
            factor_returns,
        )
        assert np.isnan(result)

    def test_alpha_percentile_rank_skips_insufficient_peer_returns(self):
        """Test alpha_percentile_rank skips peer returns with less than 3 data points (line 649-650)."""
        strategy_returns = pd.Series([0.01, 0.02, 0.015])
        all_strategies_returns = [
            pd.Series([0.008, 0.012, 0.010]),  # Valid
            pd.Series([0.01]),  # Too short - should be skipped
        ]
        factor_returns = pd.Series([0.005, 0.01, 0.008])

        result = alpha_beta.alpha_percentile_rank(
            strategy_returns,
            all_strategies_returns,
            factor_returns,
        )
        # Should have a valid result since there's one valid peer
        assert not np.isnan(result)
        assert 0 <= result <= 1
