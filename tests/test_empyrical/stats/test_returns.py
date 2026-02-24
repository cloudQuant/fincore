"""Tests for return calculations.

This module tests simple returns, cumulative returns, and return aggregation.

Split from test_stats.py to improve maintainability.

Priority Markers:
- P0: Core cum_returns, cum_returns_final tests
- P1: Aggregate returns tests
- P2: Edge cases and validation
"""
from __future__ import annotations

from functools import wraps

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
from fincore.constants import DAILY, MONTHLY, QUARTERLY, WEEKLY, YEARLY
from fincore.empyrical import Empyrical
from fincore.metrics import returns as returns_module

DECIMAL_PLACES = 8

# Pandas frequency alias compatibility
try:
    pd.date_range("2000-1-1", periods=1, freq="ME")
    MONTH_FREQ = "ME"
    YEAR_FREQ = "YE"
except ValueError:
    MONTH_FREQ = "M"
    YEAR_FREQ = "A"

rand = np.random.RandomState(1337)


class BaseTestCase(TestCase):
    """Base test case with index matching assertion."""

    def assert_indexes_match(self, result, expected):
        """Assert that two pandas objects have the same indices."""
        assert_index_equal(result.index, expected.index)

        if isinstance(result, pd.DataFrame) and isinstance(expected, pd.DataFrame):
            assert_index_equal(result.columns, expected.columns)


class TestReturns(BaseTestCase):
    """Tests for simple and cumulative returns calculations."""

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

    pos_line = pd.Series(
        np.linspace(0, 1, num=1000), index=pd.date_range("2000-1-30", periods=1000, freq="D", tz="UTC")
    )

    @property
    def empyrical(self):
        """Get empyrical module instance."""
        return EMPYRICAL_MODULE

    # ========================================================================
    # Simple Returns Tests
    # ========================================================================

    @parameterized.expand(
        [
            (flat_line_1, [0.0] * (flat_line_1.shape[0] - 1)),
            (pos_line, [np.inf] + [1 / n for n in range(1, 999)]),
        ]
    )
    @pytest.mark.p1  # High: important calculation method
    def test_simple_returns(self, prices, expected):
        """Test simple return calculation from prices."""
        simple_returns = returns_module.simple_returns(prices)
        assert_almost_equal(np.array(simple_returns), expected, 4)
        self.assert_indexes_match(simple_returns, prices.iloc[1:])

    # ========================================================================
    # Cumulative Returns Tests
    # ========================================================================

    @parameterized.expand(
        [
            (empty_returns, 0, []),
            (mixed_returns, 0, [0.0, 0.01, 0.111, 0.066559, 0.08789, 0.12052, 0.14293, 0.15436, 0.03893]),
            (
                mixed_returns,
                100,
                [100.0, 101.0, 111.1, 106.65599, 108.78912, 112.05279, 114.29384, 115.43678, 103.89310],
            ),
            (negative_returns, 0, [0.0, -0.06, -0.1258, -0.13454, -0.21243, -0.22818, -0.27449, -0.33253, -0.36590]),
        ]
    )
    @pytest.mark.p0  # Critical: core financial metric
    def test_cum_returns(self, returns, starting_value, expected):
        """Test cumulative return calculation."""
        cum_returns = returns_module.cum_returns(
            returns,
            starting_value=starting_value,
        )
        for i in range(returns.size):
            assert_almost_equal(
                cum_returns.iloc[i] if hasattr(cum_returns, "iloc") else cum_returns[i],
                expected[i] if isinstance(expected, list) else expected.iloc[i],
                4,
            )
        self.assert_indexes_match(cum_returns, returns)

    @parameterized.expand(
        [
            (empty_returns, 0, np.nan),
            (one_return, 0, one_return.iloc[0]),
            (mixed_returns, 0, 0.03893),
            (mixed_returns, 100, 103.89310),
            (negative_returns, 0, -0.36590),
        ]
    )
    @pytest.mark.p0  # Critical: core financial metric
    def test_cum_returns_final(self, returns, starting_value, expected):
        """Test final cumulative return value."""
        cum_returns_final = returns_module.cum_returns_final(
            returns,
            starting_value=starting_value,
        )
        assert_almost_equal(cum_returns_final, expected, 4)

    # ========================================================================
    # Aggregate Returns Tests
    # ========================================================================

    @parameterized.expand(
        [
            (simple_benchmark, WEEKLY, [0.0, 0.040604010000000024, 0.0]),
            (simple_benchmark, MONTHLY, [0.01, 0.03030099999999991]),
            (simple_benchmark, QUARTERLY, [0.04060401]),
            (simple_benchmark, YEARLY, [0.040604010000000024]),
            (weekly_returns, MONTHLY, [0.0, 0.087891200000000058, -0.04500459999999995]),
            (weekly_returns, YEARLY, [0.038931091700480147]),
            (monthly_returns, YEARLY, [0.038931091700480147]),
            (monthly_returns, QUARTERLY, [0.11100000000000021, 0.008575999999999917, -0.072819999999999996]),
        ]
    )
    @pytest.mark.p1  # High: important aggregation functionality
    def test_aggregate_returns(self, returns, convert_to, expected):
        """Test returns aggregation by period (weekly, monthly, quarterly, yearly)."""
        # aggregate_returns requires pandas Series with datetime index
        returns = returns_module.aggregate_returns(returns, convert_to).values.tolist()
        for i, v in enumerate(returns):
            assert_almost_equal(v, expected[i], DECIMAL_PLACES)


# ========================================================================
# Module-level reference
# ========================================================================
EMPYRICAL_MODULE = empyrical
