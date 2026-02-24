"""Tests for stats module hurst_exponent missing coverage lines 175, 193, 203."""

import numpy as np
import pandas as pd
import pytest

from fincore.metrics import stats


class TestHurstExponentLine175:
    """Test to cover line 175 in stats.py.

    Line 175 is: continue (when n_subseries < 1)
    """

    def test_hurst_exponent_large_lag_continues(self):
        """Test hurst_exponent when lag produces n_subseries < 1 (line 175).

        This happens when the lag is larger than the number of observations.
        The function should skip to the next lag.
        """
        # Create a very short series
        returns = pd.Series([0.01, 0.02, 0.015])

        result = stats.hurst_exponent(returns)

        # Should handle short series gracefully
        assert isinstance(result, (float, np.floating))


class TestHurstExponentLine193:
    """Test to cover line 193 in stats.py.

    Line 193 is: return np.nan (when len(rs_values) < 2 and s_std <= 0 or r_range <= 0)
    """

    def test_hurst_exponent_fallback_fails(self):
        """Test hurst_exponent when fallback calculation fails (line 193).

        This happens when:
        - len(rs_values) < 2 (insufficient R/S values)
        - s_std <= 0 (zero standard deviation) OR r_range <= 0 (zero range)

        All same values produce s_std = 0.
        """
        # All same values - s_std = 0
        returns = pd.Series([0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])

        result = stats.hurst_exponent(returns)

        # Should return NaN
        assert np.isnan(result)


class TestHurstExponentLine203:
    """Test to cover line 203 in stats.py.

    Line 203 is: return np.nan (when len(lags_array) < 2 after filtering)
    """

    def test_hurst_exponent_insufficient_filtered_lags(self):
        """Test hurst_exponent when filtering leaves < 2 valid lags (line 203).

        This happens when after filtering for valid (lags > 0 and rs > 0),
        we have less than 2 data points.
        """
        # Try with a series that produces few valid R/S values
        # Short series may not have enough valid lags
        returns = pd.Series([0.01, 0.015, 0.02])

        result = stats.hurst_exponent(returns)

        # Should handle edge case - may return NaN or a value
        assert isinstance(result, (float, np.floating))

    def test_hurst_exponent_edge_case_few_valid_points(self):
        """Test hurst_exponent with very few valid data points."""
        # Series with only 2 data points
        returns = pd.Series([0.01, -0.01])

        result = stats.hurst_exponent(returns)

        # Should handle edge case
        assert isinstance(result, (float, np.floating))
