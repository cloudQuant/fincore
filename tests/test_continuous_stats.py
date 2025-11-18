"""
Tests for continuous rise/fall statistics.
"""
from __future__ import division

import numpy as np
import pandas as pd
from numpy.testing import assert_almost_equal
from unittest import TestCase
from parameterized import parameterized

from fincore import empyrical

DECIMAL_PLACES = 8

# Pandas frequency alias compatibility
try:
    pd.date_range('2000-1-1', periods=1, freq='ME')
    MONTH_FREQ = 'ME'
    YEAR_FREQ = 'YE'
except ValueError:
    MONTH_FREQ = 'M'
    YEAR_FREQ = 'A'


class TestContinuousStats(TestCase):
    """Test cases for continuous rise/fall statistics."""

    # Simple returns with clear patterns
    consecutive_up = pd.Series(
        np.array([1., 2., 3., -1., 1., 2., 1.]) / 100,
        index=pd.date_range('2000-1-1', periods=7, freq='D'))

    consecutive_down = pd.Series(
        np.array([1., -2., -3., -1., 1., -2., -1.]) / 100,
        index=pd.date_range('2000-1-1', periods=7, freq='D'))

    mixed_returns = pd.Series(
        np.array([1., 2., 3., -4., -5., 6., 7., -8.]) / 100,
        index=pd.date_range('2000-1-1', periods=8, freq='D'))

    weekly_returns = pd.Series(
        np.array([1., 2., -1., -2., 3., 4.]) / 100,
        index=pd.date_range('2000-1-1', periods=6, freq='W'))

    monthly_returns = pd.Series(
        np.array([1., 2., -1., -2., 3., 4.]) / 100,
        index=pd.date_range('2000-1-1', periods=6, freq=MONTH_FREQ))

    empty_returns = pd.Series([], dtype=float)

    # Test max_consecutive_up_days
    @parameterized.expand([
        (consecutive_up, 3),  # First 3 days are consecutive up
        (consecutive_down, 1),  # Only single up days
        (mixed_returns, 3),  # First 3 days
        (empty_returns, np.nan),
    ])
    def test_max_consecutive_up_days(self, returns, expected):
        result = empyrical.max_consecutive_up_days(returns)
        if np.isnan(expected):
            assert np.isnan(result)
        else:
            assert result == expected

    # Test max_consecutive_down_days
    @parameterized.expand([
        (consecutive_down, 3),  # Days 2-4 are consecutive down
        (consecutive_up, 1),  # Only single down day
        (mixed_returns, 2),  # Days 4-5
        (empty_returns, np.nan),
    ])
    def test_max_consecutive_down_days(self, returns, expected):
        result = empyrical.max_consecutive_down_days(returns)
        if np.isnan(expected):
            assert np.isnan(result)
        else:
            assert result == expected

    # Test max_consecutive_up_weeks
    def test_max_consecutive_up_weeks(self):
        result = empyrical.max_consecutive_up_weeks(self.weekly_returns)
        assert result == 2  # First 2 weeks

    # Test max_consecutive_down_weeks
    def test_max_consecutive_down_weeks(self):
        result = empyrical.max_consecutive_down_weeks(self.weekly_returns)
        assert result == 2  # Weeks 3-4

    # Test max_consecutive_up_months
    def test_max_consecutive_up_months(self):
        result = empyrical.max_consecutive_up_months(self.monthly_returns)
        assert result == 2  # First 2 months and last 2 months

    # Test max_consecutive_down_months
    def test_max_consecutive_down_months(self):
        result = empyrical.max_consecutive_down_months(self.monthly_returns)
        assert result == 2  # Months 3-4

    # Test max_consecutive_gain
    @parameterized.expand([
        (mixed_returns, 0.13),  # 6% + 7% = 13% (days 6-7 is max)
        (consecutive_up, 0.06),  # 1% + 2% + 3% = 6%
        (empty_returns, np.nan),
    ])
    def test_max_consecutive_gain(self, returns, expected):
        result = empyrical.max_consecutive_gain(returns)
        if np.isnan(expected):
            assert np.isnan(result)
        else:
            assert_almost_equal(result, expected, DECIMAL_PLACES)

    # Test max_consecutive_loss
    @parameterized.expand([
        (mixed_returns, -0.09),  # -4% + -5% = -9%
        (consecutive_down, -0.06),  # -2% + -3% + -1% = -6%
        (empty_returns, np.nan),
    ])
    def test_max_consecutive_loss(self, returns, expected):
        result = empyrical.max_consecutive_loss(returns)
        if np.isnan(expected):
            assert np.isnan(result)
        else:
            assert_almost_equal(result, expected, DECIMAL_PLACES)

    # Test max_single_day_gain
    @parameterized.expand([
        (mixed_returns, 0.07),  # 7% is the max
        (consecutive_up, 0.03),  # 3% is the max
        (empty_returns, np.nan),
    ])
    def test_max_single_day_gain(self, returns, expected):
        result = empyrical.max_single_day_gain(returns)
        if np.isnan(expected):
            assert np.isnan(result)
        else:
            assert_almost_equal(result, expected, DECIMAL_PLACES)

    # Test max_single_day_loss
    @parameterized.expand([
        (mixed_returns, -0.08),  # -8% is the max loss
        (consecutive_down, -0.03),  # -3% is the max loss
        (empty_returns, np.nan),
    ])
    def test_max_single_day_loss(self, returns, expected):
        result = empyrical.max_single_day_loss(returns)
        if np.isnan(expected):
            assert np.isnan(result)
        else:
            assert_almost_equal(result, expected, DECIMAL_PLACES)

    # Test max_consecutive_up_start_date
    def test_max_consecutive_up_start_date(self):
        result = empyrical.max_consecutive_up_start_date(self.consecutive_up)
        expected = pd.Timestamp('2000-1-1')
        assert result == expected

    # Test max_consecutive_up_end_date
    def test_max_consecutive_up_end_date(self):
        result = empyrical.max_consecutive_up_end_date(self.consecutive_up)
        expected = pd.Timestamp('2000-1-3')
        assert result == expected

    # Test max_consecutive_down_start_date
    def test_max_consecutive_down_start_date(self):
        result = empyrical.max_consecutive_down_start_date(self.consecutive_down)
        expected = pd.Timestamp('2000-1-2')
        assert result == expected

    # Test max_consecutive_down_end_date
    def test_max_consecutive_down_end_date(self):
        result = empyrical.max_consecutive_down_end_date(self.consecutive_down)
        expected = pd.Timestamp('2000-1-4')
        assert result == expected

    # Test max_single_day_gain_date
    def test_max_single_day_gain_date(self):
        result = empyrical.max_single_day_gain_date(self.mixed_returns)
        expected = pd.Timestamp('2000-1-7')
        assert result == expected

    # Test max_single_day_loss_date
    def test_max_single_day_loss_date(self):
        result = empyrical.max_single_day_loss_date(self.mixed_returns)
        expected = pd.Timestamp('2000-1-8')
        assert result == expected
