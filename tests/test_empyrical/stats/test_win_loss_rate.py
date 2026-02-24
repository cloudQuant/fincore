"""Tests for win rate and loss rate calculations.

This module tests win rate and loss rate metrics.

Split from test_stats.py to improve maintainability.

Priority Markers:
- P2: Win/loss rate tests (secondary metrics)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_almost_equal
from parameterized import parameterized
from unittest import TestCase

from fincore import empyrical
from fincore.metrics import stats as stats_module

DECIMAL_PLACES = 8


class BaseTestCase(TestCase):
    """Base test case for win/loss rate tests."""

    pass


class TestWinLossRate(BaseTestCase):
    """Tests for win rate and loss rate calculations."""

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

    empty_returns = pd.Series(np.array([]) / 100, index=pd.date_range("2000-1-30", periods=0, freq="D"))

    one_return = pd.Series(np.array([1.0]) / 100, index=pd.date_range("2000-1-30", periods=1, freq="D"))

    @property
    def empyrical(self):
        """Get empyrical module instance."""
        return empyrical

    # ========================================================================
    # Win Rate Tests
    # ========================================================================

    @parameterized.expand(
        [
            (empty_returns, np.nan),
            (one_return, 1.0 if one_return.iloc[0] > 0 else 0.0),
            (simple_benchmark, 4 / 9),  # 4 positive out of 9: [0., 1., 0., 1., 0., 1., 0., 1., 0.]
            (mixed_returns, 6 / 8),  # 6 positive out of 8 (excluding NaN): [NaN, 1., 10., -4., 2., 3., 2., 1., -10.]
            (positive_returns, 1.0),
            (negative_returns, 0.0),  # 0 positive out of 9: [0., -6., -7., -1., -9., -2., -6., -8., -5.]
        ]
    )
    @pytest.mark.p2  # Medium: secondary metric
    def test_win_rate(self, returns, expected):
        """Test win rate calculation."""
        result = stats_module.win_rate(returns)
        if np.isnan(expected):
            assert np.isnan(result), f"Expected NaN but got {result}"
        else:
            assert_almost_equal(result, expected, DECIMAL_PLACES)

    # ========================================================================
    # Loss Rate Tests
    # ========================================================================

    @parameterized.expand(
        [
            (empty_returns, np.nan),
            (one_return, 0.0 if one_return.iloc[0] > 0 else 1.0),
            (simple_benchmark, 0.0),  # No negative returns, zeros don't count as losses
            (mixed_returns, 2 / 8),  # 2 negative out of 8 (excluding NaN)
            (positive_returns, 0.0),
            (negative_returns, 8 / 9),  # 8 negative out of 9 (zero at index 0 doesn't count)
        ]
    )
    @pytest.mark.p2  # Medium: secondary metric
    def test_loss_rate(self, returns, expected):
        """Test loss rate calculation."""
        result = stats_module.loss_rate(returns)
        if np.isnan(expected):
            assert np.isnan(result), f"Expected NaN but got {result}"
        else:
            assert_almost_equal(result, expected, DECIMAL_PLACES)


# ========================================================================
# Module-level reference
# ========================================================================
EMPYRICAL_MODULE = empyrical
