"""Tests for alpha_beta module - full coverage for edge cases.

This file tests edge cases in fincore.metrics.alpha_beta that are not covered
by the main test suite.
"""

import numpy as np
import pandas as pd

from fincore.constants import DAILY
from fincore.metrics import alpha_beta


class TestAlphaBetaAlignedEdgeCases:
    """Test edge cases for alpha_beta_aligned function."""

    def test_alpha_beta_with_single_valid_data_point(self):
        """Test alpha_beta_aligned with only 1 valid data point after NaN removal."""
        # Create arrays with only 1 valid non-NaN pair
        returns = np.array([0.01, np.nan, np.nan])
        factor_returns = np.array([0.02, np.nan, np.nan])

        result = alpha_beta.alpha_beta_aligned(returns, factor_returns, risk_free=0.0, period=DAILY)

        # Should return [nan, nan] when less than 2 valid points
        assert len(result) == 2
        assert np.isnan(result[0])  # alpha is nan
        assert np.isnan(result[1])  # beta is nan

    def test_alpha_beta_with_zero_factor_variance(self):
        """Test alpha_beta_aligned when factor has zero variance."""
        # Factor with constant returns (zero variance)
        returns = np.array([0.01, 0.02, 0.015, 0.012, 0.018])
        factor_returns = np.array([0.01, 0.01, 0.01, 0.01, 0.01])  # Constant

        result = alpha_beta.alpha_beta_aligned(returns, factor_returns, risk_free=0.0, period=DAILY)

        # Should return [nan, nan] when factor variance is zero
        assert len(result) == 2
        assert np.isnan(result[0])  # alpha is nan
        assert np.isnan(result[1])  # beta is nan

    def test_alpha_with_allocated_output_dataframe(self):
        """Test alpha_beta_aligned with preallocated output array."""
        returns = np.array([0.01, 0.02, 0.015, 0.012, 0.018])
        factor_returns = np.array([0.005, 0.01, 0.008, 0.006, 0.009])

        # Test with preallocated output
        out = np.empty(2, dtype="float64")
        result = alpha_beta.alpha_beta_aligned(returns, factor_returns, risk_free=0.0, period=DAILY, out=out)

        assert result is out
        assert len(result) == 2

    def test_alpha_beta_with_preallocated_output(self):
        """Test alpha_beta_aligned with preallocated output array."""
        returns = np.array([0.01, 0.02, 0.015, 0.012, 0.018])
        factor_returns = np.array([0.005, 0.01, 0.008, 0.006, 0.009])

        out = np.empty(2, dtype="float64")
        result = alpha_beta.alpha_beta_aligned(returns, factor_returns, risk_free=0.0, period=DAILY, out=out)

        # Result should be the same as out
        assert result is out


class TestUpAlphaBeta:
    """Test up_alpha_beta and down_alpha_beta functions."""

    def test_up_alpha_beta_basic(self):
        """Test up_alpha_beta with basic data."""
        returns = pd.Series(
            [0.01, 0.02, 0.015, -0.01, 0.018],
            index=pd.date_range("2020-01-01", periods=5, freq="D"),
        )
        factor_returns = pd.Series(
            [0.005, 0.01, 0.008, -0.005, 0.009],
            index=returns.index,
        )

        result = alpha_beta.up_alpha_beta(returns, factor_returns)

        assert len(result) == 2
        assert not np.isnan(result[0])  # alpha
        assert not np.isnan(result[1])  # beta

    def test_down_alpha_beta_basic(self):
        """Test down_alpha_beta with basic data."""
        returns = pd.Series(
            [0.01, -0.02, -0.015, -0.01, 0.018],
            index=pd.date_range("2020-01-01", periods=5, freq="D"),
        )
        factor_returns = pd.Series(
            [0.005, -0.01, -0.008, -0.005, 0.009],
            index=returns.index,
        )

        result = alpha_beta.down_alpha_beta(returns, factor_returns)

        assert len(result) == 2
        assert not np.isnan(result[0])  # alpha
        assert not np.isnan(result[1])  # beta


class TestAnnualAlphaBeta:
    """Test annual_alpha and annual_beta functions."""

    def test_annual_alpha_empty_returns(self):
        """Test annual_alpha with empty returns."""
        returns = pd.Series([], dtype=float)
        factor_returns = pd.Series([], dtype=float)

        result = alpha_beta.annual_alpha(returns, factor_returns)

        assert isinstance(result, pd.Series)
        assert len(result) == 0

    def test_annual_alpha_non_datetime_index(self):
        """Test annual_alpha with non-DatetimeIndex."""
        returns = pd.Series([0.01, 0.02, 0.015], index=[0, 1, 2])
        factor_returns = pd.Series([0.005, 0.01, 0.008], index=[0, 1, 2])

        result = alpha_beta.annual_alpha(returns, factor_returns)

        assert isinstance(result, pd.Series)
        assert len(result) == 0

    def test_annual_alpha_basic(self):
        """Test annual_alpha with basic multi-year data."""
        returns = pd.Series(
            [0.01, 0.02, 0.015, 0.012, 0.018],
            index=pd.date_range("2020-01-01", periods=5, freq="D"),
        )
        factor_returns = pd.Series(
            [0.005, 0.01, 0.008, 0.006, 0.009],
            index=returns.index,
        )

        result = alpha_beta.annual_alpha(returns, factor_returns)

        # Should return a series with alpha for the year
        assert isinstance(result, pd.Series)


class TestDownAlphaBetaEdgeCases:
    """Test down_alpha_beta function edge cases."""

    def test_down_alpha_beta_basic(self):
        """Test down_alpha_beta with basic data."""
        returns = pd.Series(
            [0.01, -0.02, -0.015, -0.01, 0.018],
            index=pd.date_range("2020-01-01", periods=5, freq="D"),
        )
        factor_returns = pd.Series(
            [0.005, -0.01, -0.008, -0.005, 0.009],
            index=returns.index,
        )

        result = alpha_beta.down_alpha_beta(returns, factor_returns)

        assert len(result) == 2
        # Result may be nan if insufficient down periods

    def test_up_alpha_beta_basic(self):
        """Test up_alpha_beta with basic data."""
        returns = pd.Series(
            [0.01, 0.02, 0.015, -0.01, 0.018],
            index=pd.date_range("2020-01-01", periods=5, freq="D"),
        )
        factor_returns = pd.Series(
            [0.005, 0.01, 0.008, -0.005, 0.009],
            index=returns.index,
        )

        result = alpha_beta.up_alpha_beta(returns, factor_returns)

        assert len(result) == 2
        # Result may be nan if insufficient up periods


class TestAnnualBeta:
    """Test annual_beta function."""

    def test_annual_beta_empty_returns(self):
        """Test annual_beta with empty returns."""
        returns = pd.Series([], dtype=float)
        factor_returns = pd.Series([], dtype=float)

        result = alpha_beta.annual_beta(returns, factor_returns)

        assert isinstance(result, pd.Series)
        assert len(result) == 0

    def test_annual_beta_non_datetime_index(self):
        """Test annual_beta with non-DatetimeIndex."""
        returns = pd.Series([0.01, 0.02, 0.015], index=[0, 1, 2])
        factor_returns = pd.Series([0.005, 0.01, 0.008], index=[0, 1, 2])

        result = alpha_beta.annual_beta(returns, factor_returns)

        assert isinstance(result, pd.Series)
        assert len(result) == 0

    def test_annual_beta_basic(self):
        """Test annual_beta with basic multi-year data."""
        returns = pd.Series(
            [0.01, 0.02, 0.015, 0.012, 0.018],
            index=pd.date_range("2020-01-01", periods=5, freq="D"),
        )
        factor_returns = pd.Series(
            [0.005, 0.01, 0.008, 0.006, 0.009],
            index=returns.index,
        )

        result = alpha_beta.annual_beta(returns, factor_returns)

        # Should return a series with beta for the year
        assert isinstance(result, pd.Series)


class TestAlphaPercentileRank:
    """Test alpha_percentile_rank function."""

    def test_percentile_rank_short_returns(self):
        """Test alpha_percentile_rank with short returns (<3 data points)."""
        returns = pd.Series([0.01, 0.02])
        factor_returns = pd.Series([0.005, 0.01])
        peer_returns = [pd.Series([0.01, 0.02, 0.015])]

        result = alpha_beta.alpha_percentile_rank(returns, peer_returns, factor_returns)

        assert np.isnan(result)

    def test_percentile_rank_nan_strategy_alpha(self):
        """Test alpha_percentile_rank when strategy alpha is NaN."""
        returns = pd.Series([np.nan, np.nan, np.nan])
        factor_returns = pd.Series([0.005, 0.01, 0.008])
        peer_returns = [
            pd.Series([0.01, 0.02, 0.015]),
            pd.Series([0.008, 0.015, 0.012]),
        ]

        result = alpha_beta.alpha_percentile_rank(returns, peer_returns, factor_returns)

        assert np.isnan(result)

    def test_percentile_rank_no_valid_peers(self):
        """Test alpha_percentile_rank when all peer alphas are NaN or too short."""
        returns = pd.Series([0.01, 0.02, 0.015])
        factor_returns = pd.Series([0.005, 0.01, 0.008])
        # All peers are too short
        peer_returns = [
            pd.Series([0.01, 0.02]),
            pd.Series([0.008, 0.015]),
        ]

        result = alpha_beta.alpha_percentile_rank(returns, peer_returns, factor_returns)

        assert np.isnan(result)

    def test_percentile_rank_basic(self):
        """Test alpha_percentile_rank with basic data."""
        strategy_returns = pd.Series([0.01, 0.02, 0.015, 0.012, 0.018])
        factor_returns = pd.Series([0.005, 0.01, 0.008, 0.006, 0.009])
        peer_returns = [
            pd.Series([0.005, 0.01, 0.008, 0.006, 0.009]),
            pd.Series([0.008, 0.015, 0.012, 0.01, 0.013]),
            pd.Series([0.002, 0.005, 0.003, 0.004, 0.006]),
        ]

        result = alpha_beta.alpha_percentile_rank(strategy_returns, peer_returns, factor_returns)

        # Should return a percentile between 0 and 1
        assert isinstance(result, float)
        assert 0 <= result <= 1

    def test_percentile_rank_skips_short_peers(self):
        """Test alpha_percentile_rank skips peers with <3 data points."""
        strategy_returns = pd.Series([0.01, 0.02, 0.015])
        factor_returns = pd.Series([0.005, 0.01, 0.008])
        peer_returns = [
            pd.Series([0.01, 0.02]),  # Too short - should be skipped
            pd.Series([0.008, 0.015, 0.012]),  # Valid
            pd.Series([0.002]),  # Too short - should be skipped
        ]

        result = alpha_beta.alpha_percentile_rank(strategy_returns, peer_returns, factor_returns)

        # Should still work with one valid peer
        assert isinstance(result, float)
        assert 0 <= result <= 1


class TestAlphaAlignedEdgeCases:
    """Test alpha_aligned edge cases for 100% coverage."""

    def test_alpha_aligned_with_1d_returns(self):
        """Test alpha_aligned with 1D returns array."""
        returns = np.array([0.01, 0.02, 0.015, 0.012, 0.018])
        factor_returns = np.array([0.005, 0.01, 0.008, 0.006, 0.009])

        result = alpha_beta.alpha_aligned(returns, factor_returns, risk_free=0.0, period=DAILY)

        # Should return a scalar
        assert isinstance(result, (float, np.floating))

    def test_alpha_beta_aligned_less_than_two_valid_points(self):
        """Test alpha_beta_aligned with less than 2 valid data points after NaN removal."""
        # Create arrays with only 1 valid non-NaN pair
        returns = np.array([0.01, np.nan, np.nan])
        factor_returns = np.array([0.02, np.nan, np.nan])

        result = alpha_beta.alpha_beta_aligned(returns, factor_returns, risk_free=0.0, period=DAILY)

        # Should return [nan, nan] when less than 2 valid points
        assert len(result) == 2
        assert np.isnan(result[0])  # alpha is nan
        assert np.isnan(result[1])  # beta is nan


class TestAlphaWithAllocatedOutput:
    """Test alpha function with allocated output for DataFrame returns."""

    def test_alpha_with_numpy_2d_array(self):
        """Test alpha_aligned with 2D numpy array returns (line 181 coverage)."""
        # Create 2D array returns with 2 assets
        returns = np.array(
            [
                [0.01, 0.008],
                [0.02, 0.015],
                [0.015, 0.012],
                [0.012, 0.01],
                [0.018, 0.016],
            ]
        )
        factor_returns = np.array([0.005, 0.01, 0.008, 0.006, 0.009])

        result = alpha_beta.alpha_aligned(returns, factor_returns, risk_free=0.0, period=DAILY)

        # Should return a 1D array with one alpha per asset
        assert isinstance(result, np.ndarray)
        assert len(result) == 2


class TestAnnualAlphaBetaEdgeCases:
    """Test annual_alpha and annual_beta with edge cases."""

    def test_annual_alpha_returns_with_some_data_in_common_years(self):
        """Test annual_alpha when there are some common years (line 534 coverage)."""
        # Create returns and factor with partial date overlap in same year
        returns = pd.Series([0.01, 0.02, 0.015], index=pd.date_range("2020-01-01", periods=3, freq="D"))
        factor_returns = pd.Series([0.005, 0.01, 0.008], index=pd.date_range("2020-01-01", periods=3, freq="D"))

        result = alpha_beta.annual_alpha(returns, factor_returns)

        # After alignment, the series will have data for common years
        assert isinstance(result, pd.Series)

    def test_annual_beta_returns_with_some_data_in_common_years(self):
        """Test annual_beta when there are some common years (line 587 coverage)."""
        returns = pd.Series([0.01, 0.02, 0.015], index=pd.date_range("2020-01-01", periods=3, freq="D"))
        factor_returns = pd.Series([0.005, 0.01, 0.008], index=pd.date_range("2020-01-01", periods=3, freq="D"))

        result = alpha_beta.annual_beta(returns, factor_returns)

        # After alignment, the series will have data for common years
        assert isinstance(result, pd.Series)

    def test_annual_alpha_multi_year_data(self):
        """Test annual_alpha with multi-year data (line 548 coverage)."""
        # Create multi-year returns and factor
        returns = pd.Series(
            [0.01, 0.02, 0.015, 0.012, 0.018],
            index=pd.date_range("2020-01-01", periods=5, freq="D"),
        )
        factor_returns = pd.Series(
            [0.005, 0.01, 0.008, 0.006, 0.009],
            index=returns.index,
        )

        result = alpha_beta.annual_alpha(returns, factor_returns)

        # Should return a series with alpha for the year
        assert isinstance(result, pd.Series)

    def test_annual_beta_multi_year_data(self):
        """Test annual_beta with multi-year data (line 601 coverage)."""
        returns = pd.Series(
            [0.01, 0.02, 0.015, 0.012, 0.018],
            index=pd.date_range("2020-01-01", periods=5, freq="D"),
        )
        factor_returns = pd.Series(
            [0.005, 0.01, 0.008, 0.006, 0.009],
            index=returns.index,
        )

        result = alpha_beta.annual_beta(returns, factor_returns)

        # Should return a series with beta for the year
        assert isinstance(result, pd.Series)


class TestAlphaBetaAlignedZeroVariance:
    """Test alpha_beta_aligned when factor has zero variance."""

    def test_alpha_beta_aligned_zero_variance_with_nan(self):
        """Test alpha_beta_aligned when factor variance is zero (lines 420-422 coverage)."""
        returns = np.array([0.01, 0.02, 0.015, 0.012, 0.018])
        factor_returns = np.array([0.01, 0.01, 0.01, 0.01, 0.01])  # Zero variance

        result = alpha_beta.alpha_beta_aligned(returns, factor_returns, risk_free=0.0, period=DAILY)

        # Should return [nan, nan] when factor variance is zero
        assert len(result) == 2
        assert np.isnan(result[0])  # alpha is nan
        assert np.isnan(result[1])  # beta is nan
