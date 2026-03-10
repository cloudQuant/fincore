"""Tests for tracking risk metrics.

Part of test_tracking_risk.py split - Tracking error, information ratio, treynor, etc. with P1 markers.
"""

from __future__ import annotations

from unittest import TestCase

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_almost_equal
from parameterized import parameterized

from fincore import empyrical
from fincore.constants import DAILY, MONTHLY, WEEKLY
from fincore.empyrical import Empyrical
from fincore.metrics import alpha_beta as alpha_beta_module
from fincore.metrics import ratios as ratios_module
from fincore.metrics import risk as risk_module
from fincore.metrics import stats as stats_module
from fincore.metrics import yearly as yearly_module

DECIMAL_PLACES = 8

# Pandas frequency alias compatibility
try:
    pd.date_range("2000-1-1", periods=1, freq="ME")
    MONTH_FREQ = "ME"
except ValueError:
    MONTH_FREQ = "M"

rand = np.random.RandomState(1337)


@pytest.mark.p1
class TestTrackingRisk(TestCase):
    """Tests for tracking risk and related metrics."""

    simple_benchmark = pd.Series(
        np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]) / 100,
        index=pd.date_range("2000-1-30", periods=9, freq="D"),
    )

    positive_returns = pd.Series(
        np.array([1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]) / 100,
        index=pd.date_range("2000-1-30", periods=9, freq="D"),
    )

    negative_returns = pd.Series(
        np.array([0.0, -6.0, -7.0, -1.0, -9.0, -2.0, -6.0, -8.0, -5.0]) / 100,
        index=pd.date_range("2000-1-30", periods=9, freq="D"),
    )

    mixed_returns = pd.Series(
        np.array([np.nan, 1.0, 10.0, -4.0, 2.0, 3.0, 2.0, 1.0, -10.0]) / 100,
        index=pd.date_range("2000-1-30", periods=9, freq="D"),
    )

    flat_line_1 = pd.Series(
        np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]) / 100,
        index=pd.date_range("2000-1-30", periods=9, freq="D"),
    )

    weekly_returns = pd.Series(
        np.array([0.0, 1.0, 10.0, -4.0, 2.0, 3.0, 2.0, 1.0, -10.0]) / 100,
        index=pd.date_range("2000-1-30", periods=9, freq="W"),
    )

    monthly_returns = pd.Series(
        np.array([0.0, 1.0, 10.0, -4.0, 2.0, 3.0, 2.0, 1.0, -10.0]) / 100,
        index=pd.date_range("2000-1-30", periods=9, freq=MONTH_FREQ),
    )

    empty_returns = pd.Series(np.array([]) / 100, index=pd.date_range("2000-1-30", periods=0, freq="D"))

    one_return = pd.Series(np.array([1.0]) / 100, index=pd.date_range("2000-1-30", periods=1, freq="D"))

    noise = pd.Series(rand.normal(0, 0.001, 1000), index=pd.date_range("2000-1-30", periods=1000, freq="D", tz="UTC"))

    @property
    def empyrical(self):
        """Get empyrical module instance."""
        return empyrical

    @parameterized.expand(
        [
            (empty_returns, empty_returns, DAILY, np.nan),
            (one_return, one_return, DAILY, np.nan),
            (mixed_returns, simple_benchmark, DAILY, 0.9234446382972831),
            (mixed_returns, simple_benchmark, WEEKLY, 0.41948097181431926),
            (mixed_returns, simple_benchmark, MONTHLY, 0.2015121407189722),
        ]
    )
    def test_tracking_error(self, returns, factor_returns, period, expected):
        """Test tracking_error calculation."""
        assert_almost_equal(
            risk_module.tracking_error(returns, factor_returns, period=period),
            expected,
            DECIMAL_PLACES,
        )

    @parameterized.expand(
        [
            (empty_returns, empty_returns, np.nan),
            (one_return, one_return, np.nan),
            (mixed_returns, simple_benchmark, 0.34111411441060574),
        ]
    )
    def test_information_ratio(self, returns, factor_returns, expected):
        """Test information_ratio calculation."""
        assert_almost_equal(
            ratios_module.information_ratio(returns, factor_returns),
            expected,
            DECIMAL_PLACES,
        )

    @parameterized.expand(
        [
            (empty_returns, empty_returns, 0.0, DAILY, np.nan),
            (one_return, one_return, 0.0, DAILY, np.nan),
            (mixed_returns, simple_benchmark, 0.0, DAILY, np.nan),
            (mixed_returns, simple_benchmark, 0.02, DAILY, np.nan),
            (mixed_returns, simple_benchmark, 0.0, WEEKLY, np.nan),
            (mixed_returns, simple_benchmark, 0.0, MONTHLY, np.nan),
        ]
    )
    def test_treynor_ratio(self, returns, factor_returns, risk_free, period, expected):
        """Test treynor_ratio calculation."""
        # treynor_ratio returns NaN when beta is zero or undefined
        result = ratios_module.treynor_ratio(returns, factor_returns, risk_free=risk_free, period=period)
        if np.isnan(expected):
            assert np.isnan(result)
        else:
            assert_almost_equal(result, expected, DECIMAL_PLACES)

    @parameterized.expand(
        [
            (empty_returns, empty_returns, 0.0, DAILY, np.nan),
            (one_return, one_return, 0.0, DAILY, np.nan),
            (mixed_returns, simple_benchmark, 0.0, DAILY, 0.17523476672946897),
        ]
    )
    def test_m_squared(self, returns, factor_returns, risk_free, period, expected):
        """Test m_squared calculation."""
        assert_almost_equal(
            ratios_module.m_squared(returns, factor_returns, risk_free=risk_free, period=period),
            expected,
            DECIMAL_PLACES,
        )

    @parameterized.expand(
        [
            (empty_returns, empty_returns, DAILY, np.nan),
            (one_return, one_return, DAILY, np.nan),
            (mixed_returns, simple_benchmark, DAILY, 0.9234446382972831),
            (mixed_returns, simple_benchmark, WEEKLY, 0.41948097181431926),
        ]
    )
    def test_annual_active_risk(self, returns, factor_returns, period, expected):
        """Test annual_active_risk calculation."""
        assert_almost_equal(
            Empyrical().annual_active_risk(returns, factor_returns, period=period),
            expected,
            DECIMAL_PLACES,
        )

    @parameterized.expand(
        [
            (empty_returns, empty_returns, DAILY, np.nan),
            (one_return, one_return, DAILY, 0.0),
            (mixed_returns, simple_benchmark, DAILY, -0.13425938751982391),
            (mixed_returns, simple_benchmark, WEEKLY, -0.011645391602789212),
            (mixed_returns, simple_benchmark, MONTHLY, -0.0022597421059689093),
        ]
    )
    def test_annual_active_return(self, returns, factor_returns, period, expected):
        """Test annual_active_return calculation."""
        assert_almost_equal(
            yearly_module.annual_active_return(returns, factor_returns, period=period),
            expected,
            DECIMAL_PLACES,
        )

    @parameterized.expand(
        [
            (empty_returns, empty_returns, np.nan),
            (one_return, one_return, 0.0),
            (mixed_returns, simple_benchmark, -0.0016729182995196545),
        ]
    )
    def test_tracking_difference(self, returns, factor_returns, expected):
        """Test tracking_difference calculation."""
        assert_almost_equal(
            stats_module.tracking_difference(returns, factor_returns),
            expected,
            DECIMAL_PLACES,
        )
