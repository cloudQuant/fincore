"""Tests for Calmar and Omega ratios.

Split from test_other_ratios.py for maintainability.

Priority Markers:
- P1: Calmar, Omega tests (important risk-adjusted return metrics)
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
from fincore.metrics import ratios as ratios_module

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


@pytest.mark.p1  # High: important risk-adjusted return metrics
class TestCalmarRatio(BaseTestCase):
    """Tests for Calmar ratio calculation."""

    # Test data
    empty_returns = pd.Series(
        np.array([]) / 100,
        index=pd.date_range("2000-1-30", periods=0, freq="D")
    )

    one_return = pd.Series(
        np.array([1.0]) / 100,
        index=pd.date_range("2000-1-30", periods=1, freq="D")
    )

    mixed_returns = pd.Series(
        np.array([np.nan, 1.0, 10.0, -4.0, 2.0, 3.0, 2.0, 1.0, -10.0]) / 100,
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

    @property
    def empyrical(self):
        """Get empyrical module instance."""
        return empyrical

    @parameterized.expand([
        (empty_returns, DAILY, np.nan),
        (one_return, DAILY, np.nan),
        (mixed_returns, DAILY, 19.135925373194233),
        (weekly_returns, WEEKLY, 2.4690830513998208),
        (monthly_returns, MONTHLY, 0.52242061386048144),
    ])
    @pytest.mark.p1  # High: important risk-adjusted return metric
    def test_calmar(self, returns, period, expected):
        """Test Calmar ratio calculation."""
        assert_almost_equal(
            ratios_module.calmar_ratio(returns, period=period),
            expected,
            DECIMAL_PLACES,
        )


@pytest.mark.p1  # High: important risk-adjusted return metrics
class TestOmegaRatio(BaseTestCase):
    """Tests for Omega ratio calculation."""

    # Test data
    empty_returns = pd.Series(
        np.array([]) / 100,
        index=pd.date_range("2000-1-30", periods=0, freq="D")
    )

    one_return = pd.Series(
        np.array([1.0]) / 100,
        index=pd.date_range("2000-1-30", periods=1, freq="D")
    )

    mixed_returns = pd.Series(
        np.array([np.nan, 1.0, 10.0, -4.0, 2.0, 3.0, 2.0, 1.0, -10.0]) / 100,
        index=pd.date_range("2000-1-30", periods=9, freq="D"),
    )

    flat_line_1 = pd.Series(
        np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]) / 100,
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

    noise_uniform = pd.Series(
        rand.uniform(-0.01, 0.01, 1000),
        index=pd.date_range("2000-1-30", periods=1000, freq="D", tz="UTC")
    )

    noise = pd.Series(
        rand.normal(0, 0.001, 1000),
        index=pd.date_range("2000-1-30", periods=1000, freq="D", tz="UTC")
    )

    @parameterized.expand([
        (empty_returns, 0.0, 0.0, np.nan),
        (one_return, 0.0, 0.0, np.nan),
        (mixed_returns, 0.0, 10.0, 0.83354263497557934),
        (mixed_returns, 0.0, -10.0, np.nan),
        (mixed_returns, flat_line_1, 0.0, 0.8125),
        (positive_returns, 0.01, 0.0, np.nan),
        (positive_returns, 0.011, 0.0, 1.125),
        (positive_returns, 0.02, 0.0, 0.0),
        (negative_returns, 0.01, 0.0, 0.0),
    ])
    @pytest.mark.p1  # High: important risk-adjusted return metric
    def test_omega(self, returns, risk_free, required_return, expected):
        """Test Omega ratio calculation."""
        assert_almost_equal(
            ratios_module.omega_ratio(
                returns,
                risk_free=risk_free,
                required_return=required_return
            ),
            expected,
            DECIMAL_PLACES,
        )

    @parameterized.expand([
        (noise_uniform, 0.0, 0.001),
        (noise, 0.001, 0.002),
    ])
    def test_omega_returns(self, returns, required_return_less, required_return_more):
        """Test Omega ratio decreases with higher required return."""
        assert ratios_module.omega_ratio(returns, required_return_less) > ratios_module.omega_ratio(
            returns, required_return_more
        )


# Module-level reference
EMPYRICAL_MODULE = empyrical
