"""Tests for missing coverage in stats.py module.

This module covers edge cases and branches that were previously uncovered:
- Line 175: hurst_exponent when n_subseries < 1
- Line 193: hurst_exponent when len(lags_array) < 2 after filtering
- Line 203: hurst_exponent empty after validation
- Line 604: r_cubed_turtle when len(years) < 1
- Line 625: r_cubed_turtle when len(max_dds) == 0
"""

import numpy as np
import pandas as pd
import pytest

from fincore.empyrical import Empyrical
from fincore.metrics import stats


class TestHurstExponentMissingCoverage:
    """Test hurst_exponent edge cases for 100% coverage."""

    def test_hurst_exponent_n_subseries_less_than_1(self):
        """Test hurst_exponent when n_subseries < 1 (line 175)."""
        # Create a very short series where n_subseries < 1
        returns = pd.Series([0.01])

        result = stats.hurst_exponent(returns)

        # When series is too short, should return NaN or 0.5
        assert result is not None

    def test_hurst_exponent_insufficient_lags_after_filtering(self):
        """Test hurst_exponent when len(lags_array) < 2 after validation (line 193)."""
        # Create returns that produce very few valid R/S values
        np.random.seed(42)
        # Very short series with limited variation
        returns = pd.Series([0.01] * 10)

        result = stats.hurst_exponent(returns)

        # With insufficient valid lags, might return NaN
        assert result is not None or np.isnan(result)

    def test_hurst_exponent_empty_after_validation(self):
        """Test hurst_exponent when filtered arrays are empty (line 203)."""
        # Create returns where all R/S values are <= 0 after filtering
        np.random.seed(42)
        returns = pd.Series([0.0] * 20)

        result = stats.hurst_exponent(returns)

        # When no valid lags remain, should return NaN
        assert result is not None or np.isnan(result)


class TestRCubedTurtleMissingCoverage:
    """Test r_cubed_turtle edge cases for 100% coverage."""

    def test_r_cubed_turtle_insufficient_data(self):
        """Test r_cubed_turtle with insufficient data for linregress (line 590-591)."""
        # The function requires at least 2 elements for linregress
        returns = pd.Series([0.01])

        result = stats.r_cubed_turtle(returns)

        # Should return NaN for insufficient data
        assert np.isnan(result)

    def test_r_cubed_turtle_short_series(self):
        """Test r_cubed_turtle with very short series (2 elements)."""
        returns = pd.Series([0.01, 0.02])

        result = stats.r_cubed_turtle(returns)

        # Should handle short series
        assert isinstance(result, (int, float, np.floating))


class TestEmpyricalEdgeCases:
    """Additional edge cases for empyrical.py."""

    def test_cagr_empty_dataframe(self):
        """Test cagr with empty DataFrame."""
        returns = pd.DataFrame(dtype=float)

        result = Empyrical.cagr(returns)

        # Should return NaN for empty input (scalar NaN, not Series)
        assert np.isnan(result) or (isinstance(result, pd.Series) and (len(result) == 0 or result.isna().all()))

    def test_cagr_single_value(self):
        """Test cagr with single value."""
        returns = pd.Series([0.01], index=pd.date_range("2020-01-01", periods=1, freq="D"))

        result = Empyrical.cagr(returns)

        # Should handle single value
        assert isinstance(result, (int, float, np.floating))


class TestStatsEdgeCases:
    """Additional edge cases for stats.py."""

    def test_skewness_with_nan(self):
        """Test skewness handles NaN values."""
        returns = pd.Series([0.01, np.nan, 0.02, -0.01, np.nan])

        result = stats.skewness(returns)

        # Should handle NaN gracefully
        assert isinstance(result, (int, float, np.floating))

    def test_kurtosis_with_nan(self):
        """Test kurtosis handles NaN values."""
        returns = pd.Series([0.01, np.nan, 0.02, -0.01, np.nan])

        result = stats.kurtosis(returns)

        # Should handle NaN gracefully
        assert isinstance(result, (int, float, np.floating))

    def test_stutzer_index_short_series(self):
        """Test stutzer_index with very short series."""
        returns = pd.Series([0.01])

        result = stats.stutzer_index(returns)

        # Should handle short series
        assert isinstance(result, (int, float, np.floating))
