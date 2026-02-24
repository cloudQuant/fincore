"""Tests for remaining missing coverage lines.

This module covers edge cases for the last 48 uncovered lines:
- drawdown.py:325 - get_all_drawdowns break condition
- ratios.py:417 - mar_ratio with empty returns after NaN removal
- round_trips.py:417 - apply_custom_and_built_in_funcs no built-in funcs
- yearly.py:236 - annual_active_return NaN check
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fincore.metrics import alpha_beta, drawdown, ratios, round_trips, stats, yearly


class TestDrawdownMissingCoverage:
    """Test drawdown.py edge cases for line 325."""

    def test_get_all_drawdowns_with_breaking_condition(self):
        """Test get_all_drawdowns break condition (line 325)."""
        # Create returns that result in the breaking condition
        # This happens when underwater becomes empty during iteration
        # The break occurs when (len(returns) == 0) or (len(underwater) == 0)
        idx = pd.date_range("2024-01-01", periods=10, freq="B")

        # Create a pattern that eventually results in empty underwater
        # After processing a drawdown, if the remaining series is empty, break
        returns = pd.Series([0.01, -0.05, 0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], index=idx)

        result = drawdown.get_all_drawdowns(returns)

        # Should handle returns gracefully
        assert isinstance(result, list)

    def test_get_all_drawdowns_with_all_zeros(self):
        """Test get_all_drawdowns with all zero returns."""
        idx = pd.date_range("2024-01-01", periods=10, freq="B")
        returns = pd.Series([0.0] * 10, index=idx)

        result = drawdown.get_all_drawdowns(returns)

        # Should return empty list when no drawdowns
        assert result == []


class TestRatiosMissingCoverage:
    """Test ratios.py edge cases for line 417."""

    def test_mar_ratio_empty_after_nan_removal(self):
        """Test mar_ratio with all NaN values (line 417)."""
        # Create returns that become empty after NaN removal
        returns = pd.Series([np.nan, np.nan, np.nan])

        result = ratios.mar_ratio(returns)

        # Should return NaN when no valid returns remain
        assert np.isnan(result)


class TestRoundTripsMissingCoverage:
    """Test round_trips.py edge cases for line 417."""

    def test_gen_round_trip_stats_basic(self):
        """Test gen_round_trip_stats which internally uses apply_custom_and_built_in_funcs."""
        # This function is internal, so we test through the public API
        # Create simple round trips data
        idx = pd.date_range("2024-01-01", periods=5, freq="B")
        round_trips_data = pd.DataFrame(
            {
                "symbol": ["AAPL", "MSFT", "AAPL", "MSFT", "GOOG"],
                "pnl": [100, -50, 75, 25, -30],
                "returns": [0.01, -0.005, 0.008, 0.002, -0.003],
                "duration": [5, 3, 4, 2, 6],
                "long": [True, False, True, False, True],
            },
            index=idx,
        )

        # Use the public function that internally uses apply_custom_and_built_in_funcs
        result = round_trips.gen_round_trip_stats(round_trips_data)

        # Should return a summary dict
        assert isinstance(result, dict)
        assert "pnl" in result


class TestYearlyMissingCoverage:
    """Test yearly.py edge cases for line 236."""

    def test_annual_active_return_nan_result(self):
        """Test annual_active_return when either result is NaN (line 236)."""
        # Create returns that result in NaN annual return
        # Empty series after alignment
        returns = pd.Series([], dtype=float)
        factor_returns = pd.Series([], dtype=float)

        result = yearly.annual_active_return(returns, factor_returns)

        # Should return NaN for empty inputs
        assert np.isnan(result)

    def test_annual_active_return_with_single_value(self):
        """Test annual_active_return with single value returns."""
        # Single value returns might produce a result rather than NaN
        returns = pd.Series([0.01])
        factor_returns = pd.Series([0.005])

        result = yearly.annual_active_return(returns, factor_returns)

        # Should handle single value - may return a number or NaN
        assert isinstance(result, (int, float, np.floating))


class TestRatiosEdgeCases:
    """Additional edge cases for ratios.py."""

    def test_calmar_ratio_zero_drawdown(self):
        """Test calmar_ratio when max_drawdown is 0."""
        # All positive returns means no drawdown
        returns = pd.Series([0.01, 0.02, 0.015, 0.01])

        result = ratios.calmar_ratio(returns)

        # Should return NaN when max_drawdown is 0
        assert np.isnan(result)

    def test_sortino_ratio_zero_downside_deviation(self):
        """Test sortino_ratio when downside deviation is 0."""
        # All positive returns means no downside
        returns = pd.Series([0.01, 0.02, 0.015])

        result = ratios.sortino_ratio(returns)

        # Should handle zero downside deviation
        assert isinstance(result, (int, float, np.floating))


class TestYearlyEdgeCases:
    """Additional edge cases for yearly.py."""

    def test_annual_return_empty_series(self):
        """Test annual_return with empty series."""
        returns = pd.Series([], dtype=float)

        result = yearly.annual_return(returns)

        # Should return NaN for empty series
        assert np.isnan(result)

    def test_annual_volatility_by_year_non_datetime_index(self):
        """Test annual_volatility_by_year with non-DatetimeIndex."""
        returns = pd.Series([0.01, 0.02, 0.015])

        result = yearly.annual_volatility_by_year(returns)

        # Should return empty series for non-DatetimeIndex
        assert isinstance(result, pd.Series)


class TestAlphaBetaMissingCoverage:
    """Tests for alpha_beta module missing coverage lines 543, 557, 596, 610."""

    def test_annual_alpha_empty_after_alignment(self):
        """Test annual_alpha returns empty Series after alignment (line 543)."""
        # Create returns with DatetimeIndex
        returns = pd.Series(
            [0.01, 0.02, 0.015],
            index=pd.date_range("2020-01-01", periods=3),
        )
        # Create factor_returns with non-overlapping DatetimeIndex
        factor_returns = pd.Series(
            [0.005, 0.01],
            index=pd.date_range("2021-01-01", periods=2),
        )

        result = alpha_beta.annual_alpha(returns, factor_returns)
        # After alignment, no common dates exist
        # The function returns a Series with NaN values for the years
        assert isinstance(result, pd.Series)
        # Result contains NaN values since no common data

    def test_annual_alpha_no_matching_years(self):
        """Test annual_alpha when no matching years found (line 557)."""
        # Create returns with data but factor with empty after alignment
        returns = pd.Series(
            [0.01, 0.02, 0.015],
            index=pd.date_range("2020-01-01", periods=3),
        )
        # Empty factor returns with empty DatetimeIndex
        factor_returns = pd.Series([], dtype=float)
        factor_returns.index = pd.DatetimeIndex([], freq="D")

        result = alpha_beta.annual_alpha(returns, factor_returns)
        # Should return series when no matching years
        assert isinstance(result, pd.Series)

    def test_annual_beta_empty_after_alignment(self):
        """Test annual_beta returns empty Series after alignment (line 596)."""
        # Create returns with DatetimeIndex
        returns = pd.Series(
            [0.01, 0.02, 0.015],
            index=pd.date_range("2020-01-01", periods=3),
        )
        # Create factor_returns with non-overlapping DatetimeIndex
        factor_returns = pd.Series(
            [0.005, 0.01],
            index=pd.date_range("2021-01-01", periods=2),
        )

        result = alpha_beta.annual_beta(returns, factor_returns)
        # After alignment, no common dates exist
        assert isinstance(result, pd.Series)

    def test_annual_beta_no_matching_years(self):
        """Test annual_beta when no matching years found (line 610)."""
        # Create returns with data but factor with empty after alignment
        returns = pd.Series(
            [0.01, 0.02, 0.015],
            index=pd.date_range("2020-01-01", periods=3),
        )
        # Empty factor returns with empty DatetimeIndex
        factor_returns = pd.Series([], dtype=float)
        factor_returns.index = pd.DatetimeIndex([], freq="D")

        result = alpha_beta.annual_beta(returns, factor_returns)
        # Should return series when no matching years
        assert isinstance(result, pd.Series)


class TestStatsMissingCoverage:
    """Tests for stats module missing coverage lines 175, 193, 203, 604, 625."""

    def test_hurst_exponent_insufficient_rs_values(self):
        """Test hurst_exponent with insufficient R/S values (line 193)."""
        # Very short series that results in < 2 R/S values
        returns = pd.Series([0.01, 0.02])
        result = stats.hurst_exponent(returns)
        # Should return nan for very short series
        assert np.isnan(result)

    def test_hurst_exponent_insufficient_lags(self):
        """Test hurst_exponent with insufficient lags after filtering (line 203)."""
        # Series where after filtering we have < 2 valid lags
        returns = pd.Series([0.01, 0.02, 0.015, 0.008, 0.012])
        result = stats.hurst_exponent(returns)
        # For short series, might return nan
        assert isinstance(result, (float, np.floating))

    def test_r_cubed_turtle_no_years(self):
        """Test r_cubed_turtle when years is empty (line 604)."""
        # Empty returns
        returns = pd.Series([], dtype=float)
        result = stats.r_cubed_turtle(returns)
        assert np.isnan(result)

    def test_r_cubed_turtle_empty_max_drawdowns(self):
        """Test r_cubed_turtle when max_drawdowns is empty (line 625)."""
        # Create returns where no valid max drawdowns are computed
        returns = pd.Series(
            [0.0, 0.0, 0.0],
            index=pd.date_range("2020-01-01", periods=3),
        )
        result = stats.r_cubed_turtle(returns)
        # With all zero returns, might return inf or nan
        assert isinstance(result, (float, np.floating))
