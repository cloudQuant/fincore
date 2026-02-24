"""Tests for Downside Risk.

Split from test_other_ratios.py for maintainability.

Priority Markers:
- P0: Core downside_risk tests (critical risk metric)
- P1: Property validation tests
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
from fincore.metrics import risk as risk_module

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


@pytest.mark.p1  # High: important risk metric
class TestDownsideRisk(BaseTestCase):
    """Tests for downside risk calculation."""

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

    df_index_simple = pd.date_range("2000-1-30", periods=8, freq="D")
    df_index_week = pd.date_range("2000-1-30", periods=8, freq="W")
    df_index_month = pd.date_range("2000-1-30", periods=8, freq=MONTH_FREQ)

    one = [-0.00171614, 0.01322056, 0.03063862, -0.01422057, -0.00489779, 0.01268925, -0.03357711, 0.01797036]
    two = [0.01846232, 0.00793951, -0.01448395, 0.00422537, -0.00339611, 0.03756813, 0.0151531, 0.03549769]

    df_simple = pd.DataFrame({
        "one": pd.Series(one, index=df_index_simple),
        "two": pd.Series(two, index=df_index_simple)
    })

    df_week = pd.DataFrame({
        "one": pd.Series(one, index=df_index_week),
        "two": pd.Series(two, index=df_index_week)
    })

    df_month = pd.DataFrame({
        "one": pd.Series(one, index=df_index_month),
        "two": pd.Series(two, index=df_index_month)
    })

    flat_line_0 = pd.Series(
        np.linspace(0, 0, num=1000),
        index=pd.date_range("2000-1-30", periods=1000, freq="D", tz="UTC")
    )

    noise = pd.Series(
        rand.normal(0, 0.001, 1000),
        index=pd.date_range("2000-1-30", periods=1000, freq="D", tz="UTC")
    )

    noise_uniform = pd.Series(
        rand.uniform(-0.01, 0.01, 1000),
        index=pd.date_range("2000-1-30", periods=1000, freq="D", tz="UTC")
    )

    @property
    def empyrical(self):
        """Get empyrical module instance."""
        return empyrical

    @parameterized.expand([
        (empty_returns, 0.0, DAILY, np.nan),
        (one_return, 0.0, DAILY, 0.0),
        (mixed_returns, mixed_returns, DAILY, 0.0),
        (mixed_returns, 0.0, DAILY, 0.60448325038829653),
        (mixed_returns, 0.1, DAILY, 1.7161730681956295),
        (weekly_returns, 0.0, WEEKLY, 0.25888650451930134),
        (weekly_returns, 0.1, WEEKLY, 0.7733045971672482),
        (monthly_returns, 0.0, MONTHLY, 0.1243650540411842),
        (monthly_returns, 0.1, MONTHLY, 0.37148351242013422),
        (df_simple, 0.0, DAILY, pd.Series([0.20671788246185202, 0.083495680595704475], index=["one", "two"])),
        (df_week, 0.0, WEEKLY, pd.Series([0.093902996054410062, 0.037928477556776516], index=["one", "two"])),
        (df_month, 0.0, MONTHLY, pd.Series([0.045109540184877193, 0.018220251263412916], index=["one", "two"])),
    ])
    @pytest.mark.p0  # Critical: core risk metric
    def test_downside_risk(self, returns, required_return, period, expected):
        """Test downside risk calculation."""
        downside_risk = risk_module.downside_risk(returns, required_return=required_return, period=period)
        if isinstance(downside_risk, float):
            assert_almost_equal(downside_risk, expected, DECIMAL_PLACES)
        else:
            for i in range(downside_risk.size):
                assert_almost_equal(
                    downside_risk.iloc[i] if hasattr(downside_risk, "iloc") else downside_risk[i],
                    expected.iloc[i] if hasattr(expected, "iloc") else expected[i],
                    DECIMAL_PLACES,
                )

    @parameterized.expand([(noise, flat_line_0), (noise_uniform, flat_line_0)])
    @pytest.mark.p1  # High: important property validation
    def test_downside_risk_noisy(self, noise, flat_line):
        """Test downside risk increases with more returns below threshold."""
        noisy_returns_1 = noise[0:250].add(flat_line[250:], fill_value=0)
        noisy_returns_2 = noise[0:500].add(flat_line[500:], fill_value=0)
        noisy_returns_3 = noise[0:750].add(flat_line[750:], fill_value=0)
        dr_1 = risk_module.downside_risk(noisy_returns_1, flat_line)
        dr_2 = risk_module.downside_risk(noisy_returns_2, flat_line)
        dr_3 = risk_module.downside_risk(noisy_returns_3, flat_line)
        assert dr_1 <= dr_2
        assert dr_2 <= dr_3

    @parameterized.expand([(noise, 0.005), (noise_uniform, 0.005)])
    @pytest.mark.p1  # High: important property validation
    def test_downside_risk_trans(self, returns, required_return):
        """Test downside risk increases with higher required return."""
        dr_0 = risk_module.downside_risk(returns, -required_return)
        dr_1 = risk_module.downside_risk(returns, 0)
        dr_2 = risk_module.downside_risk(returns, required_return)
        assert dr_0 <= dr_1
        assert dr_1 <= dr_2

    @parameterized.expand([(0.001, 0.002), (0.001, 0.01), (0, 0.001)])
    @pytest.mark.p1  # High: important property validation
    def test_downside_risk_std(self, smaller_std, larger_std):
        """Test downside risk increases with standard deviation."""
        less_noise = pd.Series(
            (rand.normal(0, smaller_std, 1000) if smaller_std != 0 else np.full(1000, 0)),
            index=pd.date_range("2000-1-30", periods=1000, freq="D"),
        )
        more_noise = pd.Series(
            (rand.normal(0, larger_std, 1000) if larger_std != 0 else np.full(1000, 0)),
            index=pd.date_range("2000-1-30", periods=1000, freq="D"),
        )
        assert risk_module.downside_risk(less_noise) < risk_module.downside_risk(more_noise)


# Module-level reference
EMPYRICAL_MODULE = empyrical
