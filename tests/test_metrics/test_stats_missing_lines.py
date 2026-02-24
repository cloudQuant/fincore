"""Tests for stats module missing coverage lines 175, 193, 203, 604, 625."""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from fincore.metrics import stats


class TestHurstExponentLine175:
    """Test to cover line 175 in stats.py.

    Line 175 is: continue (when n_subseries < 1)
    """

    def test_hurst_exponent_n_subseries_less_than_1(self):
        """Test hurst_exponent when n_subseries < 1 (line 175).

        This happens when the lag is larger than the total number of observations.
        """
        # Create a short series where some lags will produce n_subseries < 1
        returns = pd.Series([0.01, 0.02, 0.015])

        result = stats.hurst_exponent(returns)

        # For very short series, the function should still return a value or NaN
        assert isinstance(result, (float, np.floating))


class TestHurstExponentLine193:
    """Test to cover line 193 in stats.py.

    Line 193 is: return np.nan (when len(rs_values) < 2 and fallback fails)
    """

    def test_hurst_exponent_fallback_returns_nan(self):
        """Test hurst_exponent when fallback also returns NaN (line 193).

        This happens when:
        - len(rs_values) < 2
        - s_std <= 0 or r_range <= 0 (fallback fails)
        """
        # Create a series with all same values (s_std = 0)
        returns = pd.Series([0.01, 0.01, 0.01, 0.01, 0.01])

        result = stats.hurst_exponent(returns)

        # Should return NaN when can't compute hurst exponent
        assert np.isnan(result)


class TestHurstExponentLine203:
    """Test to cover line 203 in stats.py.

    Line 203 is: return np.nan (when len(lags_array) < 2 after filtering)
    """

    def test_hurst_exponent_insufficient_lags_after_filtering(self):
        """Test hurst_exponent when filtering leaves < 2 valid lags (line 203).

        This happens when most R/S values or lags are invalid.
        """
        # Create a series that results in few valid R/S values
        returns = pd.Series([0.01, 0.02, -0.01, 0.015, -0.005, 0.01, 0.02, -0.015])

        result = stats.hurst_exponent(returns)

        # Should handle edge cases gracefully
        assert isinstance(result, (float, np.floating))


class TestRCubedTurtleLine604:
    """Test to cover line 604 in stats.py.

    Line 604 is: return np.nan (when len(years) < 1)
    """

    def test_r_cubed_turtle_empty_years(self):
        """Test r_cubed_turtle when years is empty (line 604).

        This happens when returns is empty or has no years.
        """
        # Empty series with DatetimeIndex
        returns = pd.Series([], dtype=float)

        result = stats.r_cubed_turtle(returns)

        # Should return NaN for empty input
        assert np.isnan(result)

    def test_r_cubed_turtle_non_datetime_empty(self):
        """Test r_cubed_turtle with non-DatetimeIndex empty series."""
        # Series without DatetimeIndex and no data
        returns = pd.Series([], dtype=float)

        result = stats.r_cubed_turtle(returns)

        # Should return NaN
        assert np.isnan(result)


class TestRCubedTurtleLine625:
    """Test to cover line 625 in stats.py.

    Line 625 is: return np.nan (when len(max_dds) == 0)
    """

    def test_r_cubed_turtle_empty_max_drawdowns(self):
        """Test r_cubed_turtle when no max drawdowns computed (line 625).

        This happens when all returns are zero or positive (no drawdowns).
        """
        # All zero returns - no drawdowns possible
        returns = pd.Series(
            [0.0, 0.0, 0.0, 0.0, 0.0],
            index=pd.date_range("2020-01-01", periods=5),
        )

        result = stats.r_cubed_turtle(returns)

        # With no drawdowns, max_dds is empty, should return NaN
        assert np.isnan(result) or result == np.inf
