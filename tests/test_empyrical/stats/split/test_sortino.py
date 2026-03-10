"""Tests for Sortino ratio.

Part of test_sharpe_sortino.py split - Sortino ratio tests with P1 markers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_almost_equal
from parameterized import parameterized
from unittest import TestCase

from fincore.constants import DAILY, MONTHLY, WEEKLY
from fincore.metrics import ratios as ratios_module

DECIMAL_PLACES = 8

# Pandas frequency alias compatibility
try:
    pd.date_range("2000-1-1", periods=1, freq="ME")
    MONTH_FREQ = "ME"
except ValueError:
    MONTH_FREQ = "M"


@pytest.mark.p1
class TestSortinoRatio(TestCase):
    """Tests for Sortino ratio."""

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

    empty_returns = pd.Series(np.array([]) / 100, index=pd.date_range("2000-1-30", periods=0, freq="D"))

    one_return = pd.Series(np.array([1.0]) / 100, index=pd.date_range("2000-1-30", periods=1, freq="D"))

    @parameterized.expand(
        [
            (empty_returns, 0.0, DAILY, np.nan),
            (one_return, 0.0, DAILY, np.nan),
            (mixed_returns, 0.0, DAILY, 2.605531251673693),
            (mixed_returns, 0.0, WEEKLY, 1.1835801911849988),
            (mixed_returns, 0.0, MONTHLY, 0.5685735326841778),
        ]
    )
    def test_sortino(self, returns, required_return, period, expected):
        """Test Sortino ratio calculation."""
        assert_almost_equal(
            ratios_module.sortino_ratio(returns, required_return=required_return, period=period),
            expected,
            DECIMAL_PLACES,
        )

    def test_sortino_add_noise(self):
        """Test Sortino ratio behavior when adding upward noise."""
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.001, 0.02, 100))
        # Add moderate positive noise to the last few values
        noisy_returns = returns.copy()
        noisy_returns.iloc[-5:] += 0.01

        sortino_original = ratios_module.sortino_ratio(returns)
        sortino_noisy = ratios_module.sortino_ratio(noisy_returns)
        sharpe_original = ratios_module.sharpe_ratio(returns)
        sharpe_noisy = ratios_module.sharpe_ratio(noisy_returns)

        # Both ratios should improve with added positive returns
        assert sortino_noisy > sortino_original
        assert sharpe_noisy > sharpe_original

    def test_sortino_sub_noise(self):
        """Test Sortino ratio behavior when adding downward noise."""
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.001, 0.02, 100))
        # Add moderate negative noise to the last few values
        noisy_returns = returns.copy()
        noisy_returns.iloc[-5:] -= 0.01

        sortino_original = ratios_module.sortino_ratio(returns)
        sortino_noisy = ratios_module.sortino_ratio(noisy_returns)
        sharpe_original = ratios_module.sharpe_ratio(returns)
        sharpe_noisy = ratios_module.sharpe_ratio(noisy_returns)

        # Both ratios should decrease with added negative returns
        assert sortino_noisy < sortino_original
        assert sharpe_noisy < sharpe_original

    def test_sortino_translation_same(self):
        """Test translating returns and required_return by same amount doesn't change sortino."""
        np.random.seed(100)
        returns = pd.Series(np.random.normal(0.001, 0.01, 100))
        res1 = ratios_module.sortino_ratio(returns, required_return=0.005)
        res2 = ratios_module.sortino_ratio(returns + 0.005, required_return=0.01)
        assert_almost_equal(res1, res2, DECIMAL_PLACES)

    def test_sortino_translation_diff(self):
        """Test translating returns and required_return by different amounts."""
        np.random.seed(100)
        returns = pd.Series(np.random.normal(0.001, 0.01, 100))
        res1 = ratios_module.sortino_ratio(returns, required_return=0.005)
        res2 = ratios_module.sortino_ratio(
            returns + 0.005, required_return=0.01
        )  # +0.005 to returns, +0.005 to required_return
        assert_almost_equal(res1, res2, DECIMAL_PLACES)
