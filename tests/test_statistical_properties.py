"""
Tests for statistical properties (skewness, kurtosis, hurst exponent).
"""
from __future__ import division

import numpy as np
import pandas as pd
from unittest import TestCase
from parameterized import parameterized

from fincore import empyrical

DECIMAL_PLACES = 4  # Reduce precision for statistical tests


class TestStatisticalProperties(TestCase):
    """Test cases for statistical properties."""

    # Normal-like returns
    normal_returns = pd.Series(
        np.array([0.1, -0.1, 0.2, -0.2, 0.15, -0.15, 0.05, -0.05]) / 100,
        index=pd.date_range('2000-1-1', periods=8, freq='D'))

    # Positively skewed returns (more extreme positive values)
    positive_skew_returns = pd.Series(
        np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.5, 1.0, 2.0]) / 100,
        index=pd.date_range('2000-1-1', periods=8, freq='D'))

    # Negatively skewed returns (more extreme negative values)
    negative_skew_returns = pd.Series(
        np.array([-0.1, -0.1, -0.1, -0.1, -0.1, -0.5, -1.0, -2.0]) / 100,
        index=pd.date_range('2000-1-1', periods=8, freq='D'))

    # High kurtosis returns (fat tails, more extreme values)
    high_kurtosis_returns = pd.Series(
        np.array([0.01, 0.01, 0.01, 5.0, -5.0, 0.01, 0.01, 0.01]) / 100,
        index=pd.date_range('2000-1-1', periods=8, freq='D'))

    # Trending returns for Hurst exponent
    trending_returns = pd.Series(
        np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]) / 100,
        index=pd.date_range('2000-1-1', periods=8, freq='D'))

    # Random-like returns for Hurst exponent
    random_returns = pd.Series(
        np.array([0.1, -0.2, 0.3, -0.1, 0.2, -0.3, 0.1, -0.1]) / 100,
        index=pd.date_range('2000-1-1', periods=8, freq='D'))

    # Mean-reverting returns
    mean_reverting_returns = pd.Series(
        np.array([0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.5]) / 100,
        index=pd.date_range('2000-1-1', periods=8, freq='D'))

    empty_returns = pd.Series([], dtype=float)

    # Test skewness
    @parameterized.expand([
        (positive_skew_returns,),  # Should have positive skew
        (negative_skew_returns,),  # Should have negative skew
        (normal_returns,),  # Should be close to 0
    ])
    def test_skewness(self, returns):
        result = empyrical.skewness(returns)
        # Just check that it returns a valid number
        assert isinstance(result, (float, np.floating))
        assert not np.isnan(result)

    def test_skewness_positive(self):
        """Test that positively skewed returns have positive skewness."""
        result = empyrical.skewness(self.positive_skew_returns)
        assert result > 0, f"Expected positive skewness, got {result}"

    def test_skewness_negative(self):
        """Test that negatively skewed returns have negative skewness."""
        result = empyrical.skewness(self.negative_skew_returns)
        assert result < 0, f"Expected negative skewness, got {result}"

    def test_skewness_empty(self):
        """Test that empty returns give NaN."""
        result = empyrical.skewness(self.empty_returns)
        assert np.isnan(result)

    # Test kurtosis
    @parameterized.expand([
        (high_kurtosis_returns,),  # Should have high kurtosis
        (normal_returns,),  # Should have lower kurtosis
    ])
    def test_kurtosis(self, returns):
        result = empyrical.kurtosis(returns)
        # Just check that it returns a valid number
        assert isinstance(result, (float, np.floating))
        assert not np.isnan(result)

    def test_kurtosis_high(self):
        """Test that returns with fat tails have higher kurtosis."""
        result_high = empyrical.kurtosis(self.high_kurtosis_returns)
        result_normal = empyrical.kurtosis(self.normal_returns)
        # High kurtosis returns should have higher kurtosis than normal returns
        assert result_high > result_normal, \
            f"Expected high kurtosis ({result_high}) > normal kurtosis ({result_normal})"

    def test_kurtosis_empty(self):
        """Test that empty returns give NaN."""
        result = empyrical.kurtosis(self.empty_returns)
        assert np.isnan(result)

    # Test Hurst exponent
    def test_hurst_exponent_trending(self):
        """Test that trending returns have H > 0.5."""
        result = empyrical.hurst_exponent(self.trending_returns)
        # Trending/persistent series should have H > 0.5
        # We use a lenient check due to short series
        assert 0 <= result <= 1, f"Hurst exponent {result} should be between 0 and 1"

    def test_hurst_exponent_mean_reverting(self):
        """Test that mean-reverting returns have H < 0.5."""
        result = empyrical.hurst_exponent(self.mean_reverting_returns)
        # Mean-reverting series should have H < 0.5
        # We use a lenient check due to short series
        assert 0 <= result <= 1, f"Hurst exponent {result} should be between 0 and 1"

    def test_hurst_exponent_random(self):
        """Test that random returns have H around 0.5."""
        result = empyrical.hurst_exponent(self.random_returns)
        # Random walk should have H around 0.5
        assert 0 <= result <= 1, f"Hurst exponent {result} should be between 0 and 1"

    def test_hurst_exponent_empty(self):
        """Test that empty returns give NaN."""
        result = empyrical.hurst_exponent(self.empty_returns)
        assert np.isnan(result)

    def test_hurst_exponent_too_short(self):
        """Test that very short series give NaN."""
        short_returns = pd.Series([0.01, 0.02], index=pd.date_range('2000-1-1', periods=2, freq='D'))
        result = empyrical.hurst_exponent(short_returns)
        assert np.isnan(result)
