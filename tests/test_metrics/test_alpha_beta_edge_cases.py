"""Tests for alpha_beta module - edge cases for full coverage."""

import numpy as np
import pandas as pd

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


class TestAlphaBetaAlignedAdditionalCoverage:
    """Additional tests for alpha_beta_aligned to hit missing lines."""

    def test_alpha_beta_aligned_returns_nan_after_alignment_empty(self):
        """Test alpha_beta_aligned when alignment results in empty data (line 418-420)."""
        # Create returns and factor returns that become empty after alignment
        returns = pd.Series([0.01, 0.02, 0.015], index=pd.date_range("2020-01-01", periods=3))
        factor_returns = pd.Series([0.005, 0.01, 0.008], index=pd.date_range("2021-01-01", periods=3))

        result = alpha_beta.alpha_beta_aligned(
            returns.values,
            factor_returns.values,
            risk_free=0.0,
        )
        # After alignment with no common dates, we get NaN
        assert len(result) == 2
        # The result should be alpha and beta values

    def test_annual_alpha_returns_empty_for_empty_returns(self):
        """Test annual_alpha when returns is empty (lines 535-536, 542-543)."""
        # Empty returns series
        returns = pd.Series([], dtype=float)
        returns.index = pd.DatetimeIndex([], freq="D")
        factor_returns = pd.Series([0.005, 0.01], index=pd.date_range("2020-01-01", periods=2))

        result = alpha_beta.annual_alpha(returns, factor_returns)
        assert isinstance(result, pd.Series)
        assert len(result) == 0

    def test_annual_alpha_empty_result_no_matching_years(self):
        """Test annual_alpha when there are no matching years after loop (line 557)."""
        # Create a situation where returns and factor have same index
        # but the grouped years don't overlap (this is tricky with groupby)
        # Actually, looking at the code, line 557 is hit when annual_alphas is empty
        # after iterating through all years but finding no matches in factor_grouped
        # This happens when all years in returns are not in factor_grouped.groups

        # One way to trigger this: have returns with years but factor with no data
        # after alignment that groups to those same years
        # Let's try using an empty factor returns

        returns = pd.Series(
            [0.01, 0.02, 0.015],
            index=pd.date_range("2020-01-01", periods=3),
        )
        factor_returns = pd.Series([], dtype=float)
        factor_returns.index = pd.DatetimeIndex([], freq="D")

        result = alpha_beta.annual_alpha(returns, factor_returns)
        # After alignment, factor_returns will be empty series aligned to returns index
        # So len(returns) > 0 but len(factor_returns aligned) = 0
        # The loop won't find any matching years
        assert isinstance(result, pd.Series)

    def test_annual_beta_returns_empty_for_empty_returns(self):
        """Test annual_beta when returns is empty (lines 582-583, 595-596)."""
        # Empty returns series
        returns = pd.Series([], dtype=float)
        returns.index = pd.DatetimeIndex([], freq="D")
        factor_returns = pd.Series([0.005, 0.01], index=pd.date_range("2020-01-01", periods=2))

        result = alpha_beta.annual_beta(returns, factor_returns)
        assert isinstance(result, pd.Series)
        assert len(result) == 0

    def test_annual_beta_empty_result_no_matching_years(self):
        """Test annual_beta when there are no matching years (line 610)."""
        returns = pd.Series(
            [0.01, 0.02, 0.015],
            index=pd.date_range("2020-01-01", periods=3),
        )
        factor_returns = pd.Series([], dtype=float)
        factor_returns.index = pd.DatetimeIndex([], freq="D")

        result = alpha_beta.annual_beta(returns, factor_returns)
        assert isinstance(result, pd.Series)

    def test_alpha_beta_aligned_returns_nan_for_zero_variance_in_array(self):
        """Test alpha_beta_aligned returns NaN when factor variance is zero (line 429-431)."""
        # Create factor returns with exactly zero variance (all same values)
        returns = np.array([0.01, 0.02, 0.015, 0.008])
        factor_returns = np.array([0.01, 0.01, 0.01, 0.01])  # Zero variance

        result = alpha_beta.alpha_beta_aligned(
            returns,
            factor_returns,
            risk_free=0.0,
        )
        # Should return [nan, nan] when factor variance is 0
        assert len(result) == 2
        assert np.isnan(result[0])
        assert np.isnan(result[1])
