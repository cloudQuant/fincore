"""Tests for rolling metric calculations.

This module tests rolling max drawdown, rolling Sharpe ratio, rolling alpha/beta,
and rolling capture ratios.

Split from test_stats.py to improve maintainability.

Priority Markers:
- P1: All rolling metrics tests (important analytical features)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_almost_equal
from parameterized import parameterized
from unittest import TestCase

from fincore import empyrical
from fincore.empyrical import Empyrical
from fincore.metrics import drawdown as drawdown_module
from fincore.metrics import ratios as ratios_module
from fincore.metrics import rolling as rolling_module

DECIMAL_PLACES = 8


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


class TestRollingMetrics(BaseTestCase):
    """Tests for rolling window metric calculations."""

    # Test data - common series
    simple_benchmark = pd.Series(
        np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]) / 100,
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

    empty_returns = pd.Series(np.array([]) / 100, index=pd.date_range("2000-1-30", periods=0, freq="D"))

    one_return = pd.Series(np.array([1.0]) / 100, index=pd.date_range("2000-1-30", periods=1, freq="D"))

    @property
    def empyrical(self):
        """Get empyrical module instance."""
        return empyrical

    # ========================================================================
    # Rolling Max Drawdown Tests
    # ========================================================================

    @parameterized.expand([(empty_returns, 6, []), (negative_returns, 6, [-0.2282, -0.2745, -0.2899, -0.2747])])
    @pytest.mark.p1  # High: important analytical feature
    def test_roll_max_drawdown(self, returns, window, expected):
        """Test rolling max drawdown calculation."""
        test = rolling_module.roll_max_drawdown(returns, window=window)
        assert_almost_equal(np.asarray(test), np.asarray(expected), 4)

        self.assert_indexes_match(test, returns[-len(expected):])

    # ========================================================================
    # Rolling Sharpe Ratio Tests
    # ========================================================================

    @parameterized.expand(
        [
            (empty_returns, 6, []),
            (negative_returns, 6, [-18.09162052, -26.79897486, -26.69138263, -25.72298838]),
            (mixed_returns, 6, [7.57445259, 8.22784105, 8.22784105, -3.1374751]),
        ]
    )
    @pytest.mark.p1  # High: important analytical feature
    def test_roll_sharpe_ratio(self, returns, window, expected):
        """Test rolling Sharpe ratio calculation."""
        test = rolling_module.roll_sharpe_ratio(returns, window=window)
        assert_almost_equal(np.asarray(test), np.asarray(expected), DECIMAL_PLACES)

        self.assert_indexes_match(test, returns[-len(expected):])

    # ========================================================================
    # Rolling Alpha/Beta Tests
    # ========================================================================

    @parameterized.expand(
        [
            (empty_returns, empty_returns, 1, []),
            (one_return, one_return, 1, [(np.nan, np.nan)]),
            (
                mixed_returns,
                negative_returns,
                6,
                [
                    (-0.97854954, -0.7826087),
                    (-0.9828927, -0.76156584),
                    (-0.93166924, -0.61682243),
                    (-0.99967288, -0.41311475),
                ],
            ),
            (mixed_returns, mixed_returns, 6, [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]),
            (mixed_returns, -mixed_returns, 6, [(0.0, -1.0), (0.0, -1.0), (0.0, -1.0), (0.0, -1.0)]),
        ]
    )
    @pytest.mark.p1  # High: important analytical feature
    def test_roll_alpha_beta(self, returns, benchmark, window, expected):
        """Test rolling alpha/beta calculation."""
        test = Empyrical(
            return_types=(np.ndarray, pd.DataFrame),
        ).roll_alpha_beta(
            returns,
            benchmark,
            window,
        )
        if isinstance(test, pd.DataFrame):
            self.assert_indexes_match(test, benchmark[-len(expected):])
            test = test.values
        else:
            test = np.asarray(test)

        alpha_test = [t[0] for t in test]
        beta_test = [t[1] for t in test]

        alpha_expected = [t[0] for t in expected]
        beta_expected = [t[1] for t in expected]

        assert_almost_equal(
            np.asarray(alpha_test),
            np.asarray(alpha_expected),
            DECIMAL_PLACES,
        )

        assert_almost_equal(
            np.asarray(beta_test),
            np.asarray(beta_expected),
            DECIMAL_PLACES,
        )


# ========================================================================
# Module-level reference
# ========================================================================
EMPYRICAL_MODULE = empyrical
