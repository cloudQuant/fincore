"""Tests for capture ratio calculations.

This module tests capture ratio, up capture, down capture, and up/down capture.

Split from test_stats.py to improve maintainability.

Priority Markers:
- P0: Core capture tests
- P1: Rolling capture tests
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_almost_equal
from parameterized import parameterized
from unittest import TestCase

from fincore import empyrical
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


class TestCaptureRatios(BaseTestCase):
    """Tests for capture ratio calculations."""

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

    empty_returns = pd.Series(np.array([]) / 100, index=pd.date_range("2000-1-30", periods=0, freq="D"))

    one_return = pd.Series(np.array([1.0]) / 100, index=pd.date_range("2000-1-30", periods=1, freq="D"))

    @property
    def empyrical(self):
        """Get empyrical module instance."""
        return empyrical

    # ========================================================================
    # Capture Ratio Tests
    # ========================================================================

    @parameterized.expand(
        [
            (empty_returns, empty_returns, np.nan),
            (one_return, one_return, 1.0),
            (mixed_returns, mixed_returns, 1.0),
            (all_negative_returns, mixed_returns, -0.52257643222960259),
        ]
    )
    @pytest.mark.p1  # High: important benchmark comparison metric
    def test_capture_ratio(self, returns, factor_returns, expected):
        """Test capture ratio calculation."""
        assert_almost_equal(ratios_module.capture(returns, factor_returns), expected, DECIMAL_PLACES)

    # ========================================================================
    # Up Capture Tests
    # ========================================================================

    @parameterized.expand(
        [
            (empty_returns, empty_returns, np.nan),
            (one_return, one_return, 1.0),
            (mixed_returns, mixed_returns, 1.0),
            (positive_returns, mixed_returns, 0.0076167762),
            (all_negative_returns, mixed_returns, -0.0004336328),
        ]
    )
    @pytest.mark.p1  # High: important benchmark comparison metric
    def test_up_capture(self, returns, factor_returns, expected):
        """Test up capture ratio calculation."""
        assert_almost_equal(ratios_module.up_capture(returns, factor_returns), expected, DECIMAL_PLACES)

    # ========================================================================
    # Down Capture Tests
    # ========================================================================

    @parameterized.expand(
        [
            (empty_returns, empty_returns, np.nan),
            (one_return, one_return, np.nan),
            (mixed_returns, mixed_returns, 1.0),
            (all_negative_returns, mixed_returns, 0.99956025703798634),
            (positive_returns, mixed_returns, -11.27400221),
        ]
    )
    @pytest.mark.p1  # High: important benchmark comparison metric
    def test_down_capture(self, returns, factor_returns, expected):
        """Test down capture ratio calculation."""
        assert_almost_equal(ratios_module.down_capture(returns, factor_returns), expected, DECIMAL_PLACES)

    # ========================================================================
    # Up/Down Capture Tests
    # ========================================================================

    @parameterized.expand(
        [
            (empty_returns, empty_returns, np.nan),
            (one_return, one_return, np.nan),
            (mixed_returns, mixed_returns, 1.0),
            (positive_returns, mixed_returns, -0.0006756053495),
            (all_negative_returns, mixed_returns, -0.0004338236),
        ]
    )
    @pytest.mark.p1  # High: important benchmark comparison metric
    def test_up_down_capture(self, returns, factor_returns, expected):
        """Test up/down capture ratio calculation."""
        assert_almost_equal(ratios_module.up_down_capture(returns, factor_returns), expected, DECIMAL_PLACES)

    # ========================================================================
    # Rolling Up Capture Tests
    # ========================================================================

    @parameterized.expand(
        [
            (empty_returns, empty_returns, 1, []),
            (one_return, one_return, 1, [1.0]),
            (mixed_returns, mixed_returns, 6, [1.0, 1.0, 1.0, 1.0]),
            (positive_returns, mixed_returns, 6, [0.00128406, 0.00291564, 0.00171499, 0.0777048]),
            (
                all_negative_returns,
                mixed_returns,
                6,
                [-5.88144154e-05, -1.52119182e-04, -1.52119198e-04, -6.89238735e-03],
            ),
        ]
    )
    @pytest.mark.p1  # High: important benchmark comparison metric
    def test_roll_up_capture(self, returns, factor_returns, window, expected):
        """Test rolling up capture calculation."""
        test = rolling_module.roll_up_capture(returns, factor_returns, window=window)
        assert_almost_equal(np.asarray(test), np.asarray(expected), DECIMAL_PLACES)

        self.assert_indexes_match(test, returns[-len(expected):])

    # ========================================================================
    # Rolling Down Capture Tests
    # ========================================================================

    @parameterized.expand(
        [
            (empty_returns, empty_returns, 1, []),
            (one_return, one_return, 1, [np.nan]),
            (mixed_returns, mixed_returns, 6, [1.0, 1.0, 1.0, 1.0]),
            (positive_returns, mixed_returns, 6, [-11.2743862, -11.2743862, -11.2743862, -11.27400221]),
            (all_negative_returns, mixed_returns, 6, [0.92058591, 0.92058591, 0.92058591, 0.99956026]),
        ]
    )
    @pytest.mark.p1  # High: important benchmark comparison metric
    def test_roll_down_capture(self, returns, factor_returns, window, expected):
        """Test rolling down capture calculation."""
        test = rolling_module.roll_down_capture(returns, factor_returns, window=window)
        assert_almost_equal(np.asarray(test), np.asarray(expected), DECIMAL_PLACES)

        self.assert_indexes_match(test, returns[-len(expected):])

    # ========================================================================
    # Rolling Up/Down Capture Tests
    # ========================================================================

    @parameterized.expand(
        [
            (empty_returns, empty_returns, 1, []),
            (one_return, one_return, 1, np.nan),
            (mixed_returns, mixed_returns, 6, [1.0, 1.0, 1.0, 1.0]),
            (positive_returns, mixed_returns, 6, [-0.00011389, -0.00025861, -0.00015211, -0.00689239]),
            (
                all_negative_returns,
                mixed_returns,
                6,
                [-6.38880246e-05, -1.65241701e-04, -1.65241719e-04, -6.89541957e-03],
            ),
        ]
    )
    @pytest.mark.p1  # High: important benchmark comparison metric
    def test_roll_up_down_capture(self, returns, factor_returns, window, expected):
        """Test rolling up/down capture calculation."""
        test = rolling_module.roll_up_down_capture(returns, factor_returns, window=window)
        assert_almost_equal(np.asarray(test), np.asarray(expected), DECIMAL_PLACES)


# ========================================================================
# Module-level reference
# ========================================================================
EMPYRICAL_MODULE = empyrical
