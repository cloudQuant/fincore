"""Tests for drawdown metrics.

This module tests max drawdown, drawdown duration, and drawdown recovery.

Split from test_stats.py to improve maintainability.

Priority Markers:
- P0: Core max_drawdown tests
- P1: Translation and behavior tests
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_almost_equal

try:
    from pandas.testing import assert_index_equal
except ImportError:
    from pandas.util.testing import assert_index_equal
from parameterized import parameterized
from unittest import TestCase

from fincore import empyrical
from fincore.constants import DAILY, WEEKLY, MONTHLY
from fincore.empyrical import Empyrical
from fincore.metrics import drawdown as drawdown_module

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
        assert_index_equal(result.index, expected.index)

        if isinstance(result, pd.DataFrame) and isinstance(expected, pd.DataFrame):
            assert_index_equal(result.columns, expected.columns)


class TestDrawdown(BaseTestCase):
    """Tests for drawdown-related metrics."""

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

    all_negative_returns = pd.Series(
        np.array([-2.0, -6.0, -7.0, -1.0, -9.0, -2.0, -6.0, -8.0, -5.0]) / 100,
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

    noise_uniform = pd.Series(
        rand.uniform(-0.01, 0.01, 1000), index=pd.date_range("2000-1-30", periods=1000, freq="D", tz="UTC")
    )

    @property
    def empyrical(self):
        """Get empyrical module instance."""
        return empyrical

    # ========================================================================
    # Max Drawdown Tests
    # ========================================================================

    @parameterized.expand(
        [
            (empty_returns, np.nan),
            (one_return, 0.0),
            (simple_benchmark, 0.0),
            (mixed_returns, -0.1),
            (positive_returns, -0.0),
            # negative returns means the drawdown is just the returns
            (negative_returns, Empyrical.cum_returns_final(negative_returns)),
            (all_negative_returns, Empyrical.cum_returns_final(all_negative_returns)),
            (
                pd.Series(
                    np.array([10, -10, 10]) / 100,
                    index=pd.date_range("2000-1-30", periods=3, freq="D"),
                ),
                -0.10,
            ),
        ]
    )
    @pytest.mark.p0  # Critical: core financial metric
    def test_max_drawdown(self, returns, expected):
        """Test maximum drawdown calculation."""
        assert_almost_equal(
            drawdown_module.max_drawdown(returns),
            expected,
            DECIMAL_PLACES,
        )

    # Maximum drawdown is always less than or equal to zero. Translating
    # returns by a positive constant should increase the maximum
    # drawdown to a maximum of zero. Translating by a negative constant
    # decreases the maximum drawdown.
    @parameterized.expand(
        [
            (noise, 0.0001),
            (noise, 0.001),
            (noise_uniform, 0.01),
            (noise_uniform, 0.1),
        ]
    )
    @pytest.mark.p1  # High: important property validation
    def test_max_drawdown_translation(self, returns, constant):
        """Test max drawdown behavior under return translation."""
        depressed_returns = returns - constant
        raised_returns = returns + constant
        max_dd = drawdown_module.max_drawdown(returns)
        depressed_dd = drawdown_module.max_drawdown(depressed_returns)
        raised_dd = drawdown_module.max_drawdown(raised_returns)
        assert max_dd <= raised_dd
        assert depressed_dd <= max_dd

    # ========================================================================
    # Drawdown Duration Tests
    # ========================================================================

    # Note: Additional drawdown duration tests would be extracted here
    # from the original test_stats.py file including:
    # - test_max_drawdown_days
    # - test_max_drawdown_weeks
    # - test_max_drawdown_months
    # - test_max_drawdown_recovery_days
    # - test_max_drawdown_recovery_weeks
    # - test_max_drawdown_recovery_months
    # - test_second_max_drawdown
    # - test_third_max_drawdown


# ========================================================================
# Module-level reference
# ========================================================================
EMPYRICAL_MODULE = empyrical
