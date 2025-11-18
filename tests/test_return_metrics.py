"""
Tests for additional return metrics.
"""
from __future__ import division

import numpy as np
import pandas as pd
from numpy.testing import assert_almost_equal
from unittest import TestCase

from fincore import empyrical

DECIMAL_PLACES = 4


class TestReturnMetrics(TestCase):
    """Test cases for return metrics."""

    # Simple returns for testing
    simple_returns = pd.Series(
        np.array([1., 2., 1., 1., 1., 1., 1., 1., 1., 1.]) / 100,
        index=pd.date_range('2000-1-1', periods=10, freq='D'))

    # Multi-year returns
    multi_year_returns = pd.Series(
        np.random.randn(500) / 100 + 0.0003,  # Slight positive drift
        index=pd.date_range('2020-1-1', periods=500, freq='D'))

    multi_year_benchmark = pd.Series(
        np.random.randn(500) / 100 + 0.0002,  # Slight positive drift
        index=pd.date_range('2020-1-1', periods=500, freq='D'))

    # Negative returns
    negative_returns = pd.Series(
        np.array([-1., -2., -1., -1., -1., -1., -1., -1., -1., -1.]) / 100,
        index=pd.date_range('2000-1-1', periods=10, freq='D'))

    empty_returns = pd.Series([], dtype=float)

    # Test annualized cumulative return
    def test_annualized_cumulative_return(self):
        """Test annualized cumulative return calculation."""
        result = empyrical.annualized_cumulative_return(self.simple_returns)
        # Should return a valid number
        assert isinstance(result, (float, np.floating))
        if not np.isnan(result):
            # Should be positive for positive returns
            assert result > 0, f"Expected positive return, got {result}"

    def test_annualized_cumulative_return_negative(self):
        """Test with negative returns."""
        result = empyrical.annualized_cumulative_return(self.negative_returns)
        if not np.isnan(result):
            # Should be negative for negative returns
            assert result < 0, f"Expected negative return, got {result}"

    def test_annualized_cumulative_return_empty(self):
        """Test that empty returns give NaN."""
        result = empyrical.annualized_cumulative_return(self.empty_returns)
        assert np.isnan(result)

    def test_annualized_cumulative_return_vs_cagr(self):
        """Test relationship with CAGR/annual_return."""
        ann_cum_ret = empyrical.annualized_cumulative_return(self.multi_year_returns)
        cagr = empyrical.cagr(self.multi_year_returns)
        # Should be very close
        if not (np.isnan(ann_cum_ret) or np.isnan(cagr)):
            assert_almost_equal(ann_cum_ret, cagr, decimal=4)

    # Test annual active return by year
    def test_annual_active_return_by_year(self):
        """Test annual active return by year."""
        result = empyrical.annual_active_return_by_year(
            self.multi_year_returns,
            self.multi_year_benchmark
        )
        # Should return a Series
        assert isinstance(result, pd.Series)
        # Should have at least one year
        if len(result) > 0:
            # Years should be integers
            assert all(isinstance(year, (int, np.integer)) for year in result.index)

    def test_annual_active_return_by_year_values(self):
        """Test that annual active returns are reasonable."""
        result = empyrical.annual_active_return_by_year(
            self.multi_year_returns,
            self.multi_year_benchmark
        )
        # All values should be valid numbers
        if len(result) > 0:
            assert not result.isnull().any()

    def test_annual_active_return_by_year_empty(self):
        """Test that empty returns give empty Series."""
        result = empyrical.annual_active_return_by_year(
            self.empty_returns,
            self.multi_year_benchmark
        )
        assert len(result) == 0

    def test_annual_active_return_by_year_consistency(self):
        """Test consistency with annual_return_by_year."""
        active_by_year = empyrical.annual_active_return_by_year(
            self.multi_year_returns,
            self.multi_year_benchmark
        )

        strategy_by_year = empyrical.annual_return_by_year(self.multi_year_returns)
        benchmark_by_year = empyrical.annual_return_by_year(self.multi_year_benchmark)

        # For common years, active return should equal strategy - benchmark
        for year in active_by_year.index:
            if year in strategy_by_year.index and year in benchmark_by_year.index:
                expected = strategy_by_year[year] - benchmark_by_year[year]
                actual = active_by_year[year]
                assert_almost_equal(actual, expected, decimal=4,
                                  err_msg=f"Year {year}: expected {expected}, got {actual}")
