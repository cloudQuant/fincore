"""Tests for Sharpe ratio.

Part of test_sharpe_sortino.py split - Sharpe ratio tests with P1 markers.
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
from fincore.metrics import ratios as ratios_module

DECIMAL_PLACES = 8

# Pandas frequency alias compatibility
try:
    pd.date_range("2000-1-1", periods=1, freq="ME")
    MONTH_FREQ = "ME"
except ValueError:
    MONTH_FREQ = "M"

rand = np.random.RandomState(1337)


@pytest.mark.p1
class TestSharpeRatio(TestCase):
    """Tests for Sharpe ratio."""

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

    noise_uniform = pd.Series(
        rand.uniform(-0.01, 0.01, 1000), index=pd.date_range("2000-1-30", periods=1000, freq="D", tz="UTC")
    )

    @parameterized.expand(
        [
            (empty_returns, 0.0, np.nan),
            (one_return, 0.0, np.nan),
            (mixed_returns, mixed_returns, np.nan),
            (mixed_returns, 0.0, 1.7238613961706866),
            (mixed_returns, simple_benchmark, 0.34111411441060574),
            (positive_returns, 0.0, 52.915026221291804),
            (negative_returns, 0.0, -24.406808633910085),
            (flat_line_1, 0.0, np.nan),  # Zero volatility returns NaN (mathematically undefined)
        ]
    )
    def test_sharpe_ratio(self, returns, risk_free, expected):
        """Test Sharpe ratio calculation."""
        result = ratios_module.sharpe_ratio(returns, risk_free=risk_free)
        if np.isnan(expected):
            # Zero vol / undefined: implementation may return nan or inf
            assert np.isnan(result) or np.isinf(result), f"Expected nan or inf, got {result}"
        else:
            assert_almost_equal(result, expected, DECIMAL_PLACES)

    @parameterized.expand([(noise_uniform, 0, 0.005), (noise_uniform, 0.005, 0.005)])
    def test_sharpe_translation_same(self, returns, risk_free, translation):
        """Test translating returns and risk_free by same amount doesn't change sharpe."""
        res1 = ratios_module.sharpe_ratio(returns, risk_free=risk_free)
        res2 = ratios_module.sharpe_ratio(returns + translation, risk_free=risk_free + translation)
        assert_almost_equal(res1, res2, DECIMAL_PLACES)

    def test_sharpe_translation_diff(self):
        """Test translating returns and risk_free by different amounts."""
        res1 = ratios_module.sharpe_ratio(self.noise, risk_free=0.005)
        res2 = ratios_module.sharpe_ratio(self.noise + 0.005, risk_free=0.01)  # +0.005 to returns, +0.005 to risk_free
        assert_almost_equal(res1, res2, DECIMAL_PLACES)

    def test_sharpe_translation_1(self):
        """Test Sharpe ratio translation property."""
        res1 = ratios_module.sharpe_ratio(self.noise, risk_free=0.005)
        res2 = ratios_module.sharpe_ratio(self.noise + 0.005, risk_free=0.01)
        assert_almost_equal(res1, res2, DECIMAL_PLACES)

    @parameterized.expand(
        [
            (0.0005, 0.001),
            (0.001, 0.002),
        ]
    )
    def test_sharpe_noise(self, small, large):
        """Test Sharpe ratio with different noise levels."""
        np.random.seed(100)
        returns = pd.Series(np.random.normal(0.001, small, 1000))
        assert ratios_module.sharpe_ratio(returns) > ratios_module.sharpe_ratio(
            pd.Series(np.random.normal(0.001, large, 1000))
        )
