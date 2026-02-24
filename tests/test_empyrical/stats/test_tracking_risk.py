"""Tests for tracking risk and related metrics.

This module tests tracking error, information ratio, Treynor ratio,
M-squared, annual active risk/return, tracking difference, and
up/down alpha-beta calculations.

Split from test_alpha_beta.py to improve maintainability.

Priority Markers:
- P0: Core tracking_error, information_ratio tests
- P1: treynor_ratio, up/down alpha-beta tests
- P2: Other metrics
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_almost_equal
from parameterized import parameterized
from unittest import TestCase

from fincore import empyrical
from fincore.constants import DAILY, MONTHLY, WEEKLY
from fincore.empyrical import Empyrical
from fincore.metrics import alpha_beta as alpha_beta_module

DECIMAL_PLACES = 8

# Pandas frequency alias compatibility
try:
    pd.date_range("2000-1-1", periods=1, freq="ME")
    MONTH_FREQ = "ME"
except ValueError:
    MONTH_FREQ = "M"

rand = np.random.RandomState(1337)


class BaseTestCase(TestCase):
    """Base test case with index matching assertion."""

    def assert_indexes_match(self, result, expected):
        """Assert that two pandas objects have the same indices."""
        try:
            from pandas.testing import assert_index_equal
        except ImportError:
            from pandas.util.testing import assert_index_equal

        assert_index_equal(result.index, expected.index)

        if isinstance(result, pd.DataFrame) and isinstance(expected, pd.DataFrame):
            assert_index_equal(result.columns, expected.columns)


class TestTrackingRisk(BaseTestCase):
    """Tests for tracking risk and related metrics."""

    # Test data - common series
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

    # ========================================================================
    # Up/Down Alpha Beta Tests
    # ========================================================================

    @parameterized.expand(
        [
            (empty_returns, empty_returns, (np.nan, np.nan)),
            (one_return, one_return, (np.nan, np.nan)),
            (mixed_returns[1:], negative_returns[1:], (-0.9997853834885004, -0.71296296296296313)),
            (mixed_returns, mixed_returns, (0.0, 1.0)),
            (mixed_returns, -mixed_returns, (0.0, -1.0)),
        ]
    )
    @pytest.mark.p1  # High: important benchmark analysis
    def test_down_alpha_beta(self, returns, benchmark, expected):
        """Test down alpha and beta calculation."""
        # Ensure returns and benchmark have same length to avoid pandas_only skip
        down_alpha, down_beta = Empyrical(
            return_types=np.ndarray,
        ).down_alpha_beta(returns, benchmark)
        assert_almost_equal(down_alpha, expected[0], DECIMAL_PLACES)
        assert_almost_equal(down_beta, expected[1], DECIMAL_PLACES)

    @parameterized.expand(
        [
            (empty_returns, empty_returns, (np.nan, np.nan)),
            (one_return, one_return, (np.nan, np.nan)),
            (mixed_returns[1:], positive_returns[1:], (0.432961242076658, 0.4285714285)),
            (mixed_returns, mixed_returns, (0.0, 1.0)),
            (mixed_returns, -mixed_returns, (0.0, -1.0)),
        ]
    )
    @pytest.mark.p1  # High: important benchmark analysis
    def test_up_alpha_beta(self, returns, benchmark, expected):
        """Test up alpha and beta calculation."""
        # Ensure returns and benchmark have same length to avoid pandas_only skip
        up_alpha, up_beta = Empyrical(
            return_types=np.ndarray,
        ).up_alpha_beta(returns, benchmark)
        assert_almost_equal(up_alpha, expected[0], DECIMAL_PLACES)
        assert_almost_equal(up_beta, expected[1], DECIMAL_PLACES)

    # ========================================================================
    # Tracking Error Tests
    # ========================================================================

    @parameterized.expand(
        [
            (empty_returns, simple_benchmark, DAILY, np.nan),
            (one_return, one_return, DAILY, np.nan),  # Single point has no std
            (mixed_returns, simple_benchmark, DAILY, 0.9234446382972831),
            (mixed_returns, mixed_returns, DAILY, 0.0),
            (negative_returns, negative_returns, DAILY, 0.0),  # Same series = 0 tracking error
            (negative_returns, negative_returns, WEEKLY, 0.0),
            (negative_returns, negative_returns, MONTHLY, 0.0),
        ]
    )
    @pytest.mark.p0  # Critical: core tracking risk metric
    def test_tracking_error(self, returns, factor_returns, period, expected):
        """Test tracking error calculation."""
        assert_almost_equal(
            Empyrical.tracking_error(returns, factor_returns, period=period),
            expected,
            DECIMAL_PLACES,
        )

    # ========================================================================
    # Information Ratio Tests
    # ========================================================================

    @parameterized.expand(
        [
            (empty_returns, simple_benchmark, np.nan),
            (one_return, one_return, np.nan),
            (mixed_returns, simple_benchmark, 0.34111411441060574),
        ]
    )
    @pytest.mark.p0  # Critical: core risk-adjusted return metric
    def test_information_ratio(self, returns, factor_returns, expected):
        """Test information ratio calculation."""
        assert_almost_equal(
            Empyrical.information_ratio(returns, factor_returns),
            expected,
            DECIMAL_PLACES,
        )

    # ========================================================================
    # Treynor Ratio Tests
    # ========================================================================

    @parameterized.expand(
        [
            (empty_returns, simple_benchmark, 0.0, DAILY, np.nan),
            (one_return, one_return, 0.0, DAILY, np.nan),
            (mixed_returns, simple_benchmark, 0.0, DAILY, np.nan),  # Beta is negative
            (mixed_returns, mixed_returns, 0.0, DAILY, 1.913592537319458),  # Beta is 1
            (positive_returns, mixed_returns, 0.0, DAILY, 9382.016570787557),
            (mixed_returns, simple_benchmark, 0.01, DAILY, np.nan),  # Beta is negative
            (weekly_returns, simple_benchmark, 0.0, WEEKLY, 0.13215767793298694),
            (monthly_returns, simple_benchmark, 0.0, MONTHLY, np.nan),  # Beta might be negative or zero
        ]
    )
    @pytest.mark.p1  # High: important risk-adjusted return metric
    def test_treynor_ratio(self, returns, factor_returns, risk_free, period, expected):
        """Test Treynor ratio calculation."""
        result = Empyrical.treynor_ratio(returns, factor_returns, risk_free=risk_free, period=period)
        if np.isnan(expected):
            assert np.isnan(result), f"Expected NaN but got {result}"
        else:
            assert_almost_equal(result, expected, DECIMAL_PLACES)

    # ========================================================================
    # M-Squared Tests
    # ========================================================================

    @parameterized.expand(
        [
            (empty_returns, simple_benchmark, 0.0, DAILY, np.nan),
            (one_return, one_return, 0.0, DAILY, np.nan),
            (mixed_returns, simple_benchmark, 0.0, DAILY, 0.17523476672946897),
            (mixed_returns, mixed_returns, 0.0, DAILY, 1.913592537319458),
            (negative_returns, negative_returns, 0.0, DAILY, -0.9999971141282427),
            (mixed_returns, simple_benchmark, 0.01, DAILY, 0.18431902963647773),
            (negative_returns, negative_returns, 0.0, WEEKLY, -0.9280745543718799),
            (negative_returns, negative_returns, 0.0, MONTHLY, -0.4552419357313745),
        ]
    )
    @pytest.mark.p1  # High: important risk-adjusted return metric
    def test_m_squared(self, returns, factor_returns, risk_free, period, expected):
        """Test M-squared ratio calculation."""
        result = Empyrical.m_squared(returns, factor_returns, risk_free=risk_free, period=period)
        if np.isnan(expected):
            assert np.isnan(result), f"Expected NaN but got {result}"
        else:
            assert_almost_equal(result, expected, DECIMAL_PLACES)

    # ========================================================================
    # Annual Active Risk Tests
    # ========================================================================

    @parameterized.expand(
        [
            (empty_returns, simple_benchmark, DAILY, np.nan),
            (one_return, one_return, DAILY, np.nan),  # Single point has no std
            (mixed_returns, simple_benchmark, DAILY, 0.9234446382972831),
            (mixed_returns, mixed_returns, DAILY, 0.0),
            (negative_returns, negative_returns, DAILY, 0.0),  # Same series = 0 active risk
            (negative_returns, negative_returns, WEEKLY, 0.0),
            (negative_returns, negative_returns, MONTHLY, 0.0),
        ]
    )
    @pytest.mark.p1  # High: important tracking risk metric
    def test_annual_active_risk(self, returns, factor_returns, period, expected):
        """Test annual active risk calculation."""
        assert_almost_equal(
            Empyrical.annual_active_risk(returns, factor_returns, period=period),
            expected,
            DECIMAL_PLACES,
        )

    # ========================================================================
    # Annual Active Return Tests
    # ========================================================================

    @parameterized.expand(
        [
            (empty_returns, simple_benchmark, DAILY, np.nan),
            (one_return, one_return, DAILY, 0.0),
            (mixed_returns, simple_benchmark, DAILY, -0.13425938751982391),
            (mixed_returns, mixed_returns, DAILY, 0.0),
            (positive_returns, mixed_returns, DAILY, 13.259480083900217),
            (weekly_returns, simple_benchmark, WEEKLY, -0.005935602500302561),
            (monthly_returns, simple_benchmark, MONTHLY, -0.001167416484335826),
        ]
    )
    @pytest.mark.p1  # High: important active return metric
    def test_annual_active_return(self, returns, factor_returns, period, expected):
        """Test annual active return calculation."""
        assert_almost_equal(
            Empyrical.annual_active_return(returns, factor_returns, period=period),
            expected,
            DECIMAL_PLACES,
        )

    # ========================================================================
    # Tracking Difference Tests
    # ========================================================================

    @parameterized.expand(
        [
            (empty_returns, simple_benchmark, np.nan),
            (one_return, one_return, 0.0),
            (mixed_returns, simple_benchmark, -0.0016729182995196545),
            (mixed_returns, mixed_returns, 0.0),
            (negative_returns, negative_returns, 0.0),  # Same series = 0 tracking difference
        ]
    )
    @pytest.mark.p1  # High: important tracking metric
    def test_tracking_difference(self, returns, factor_returns, expected):
        """Test tracking difference calculation."""
        assert_almost_equal(
            Empyrical.tracking_difference(returns, factor_returns),
            expected,
            DECIMAL_PLACES,
        )


# ========================================================================
# Module-level reference
# ========================================================================
EMPYRICAL_MODULE = empyrical
