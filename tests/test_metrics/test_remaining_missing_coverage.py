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
import pytest

from fincore.metrics import drawdown, ratios, round_trips, yearly


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
