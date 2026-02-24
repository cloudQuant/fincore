"""Tests for Sharpe and Sortino ratios.

This module tests Sharpe ratio and Sortino ratio calculations.

Split from test_ratios.py to improve maintainability.

Priority Markers:
- P0: Core sharpe_ratio and sortino_ratio tests
- P1: Translation and noise behavior tests
- P2: Edge case validation tests
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


class TestSharpeSortino(BaseTestCase):
    """Tests for Sharpe and Sortino ratios."""

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

    noise_uniform = pd.Series(
        rand.uniform(-0.01, 0.01, 1000), index=pd.date_range("2000-1-30", periods=1000, freq="D", tz="UTC")
    )

    df_index_simple = pd.date_range("2000-1-30", periods=8, freq="D")
    df_index_week = pd.date_range("2000-1-30", periods=8, freq="W")
    df_index_month = pd.date_range("2000-1-30", periods=8, freq=MONTH_FREQ)

    one = [-0.00171614, 0.01322056, 0.03063862, -0.01422057, -0.00489779, 0.01268925, -0.03357711, 0.01797036]
    two = [0.01846232, 0.00793951, -0.01448395, 0.00422537, -0.00339611, 0.03756813, 0.0151531, 0.03549769]

    df_simple = pd.DataFrame(
        {"one": pd.Series(one, index=df_index_simple), "two": pd.Series(two, index=df_index_simple)}
    )

    df_week = pd.DataFrame({"one": pd.Series(one, index=df_index_week), "two": pd.Series(two, index=df_index_week)})

    df_month = pd.DataFrame({"one": pd.Series(one, index=df_index_month), "two": pd.Series(two, index=df_index_month)})

    @property
    def empyrical(self):
        """Get empyrical module instance."""
        return empyrical

    # ========================================================================
    # Sharpe Ratio Tests
    # ========================================================================

    @parameterized.expand(
        [
            (empty_returns, 0.0, np.nan),
            (one_return, 0.0, np.nan),
            (mixed_returns, mixed_returns, np.nan),
            (mixed_returns, 0.0, 1.7238613961706866),
            (mixed_returns, simple_benchmark, 0.34111411441060574),
            (positive_returns, 0.0, 52.915026221291804),
            (negative_returns, 0.0, -24.406808633910085),
            (flat_line_1, 0.0, np.inf),
        ]
    )
    @pytest.mark.p0  # Critical: core financial metric
    def test_sharpe_ratio(self, returns, risk_free, expected):
        """Test Sharpe ratio calculation."""
        assert_almost_equal(
            ratios_module.sharpe_ratio(returns, risk_free=risk_free),
            expected,
            DECIMAL_PLACES,
        )

    # Translating the returns and required returns by the same amount
    # does not change the sharpe ratio.
    @parameterized.expand([(noise_uniform, 0, 0.005), (noise_uniform, 0.005, 0.005)])
    @pytest.mark.p1  # High: important property validation
    def test_sharpe_translation_same(self, returns, required_return, translation):
        """Test Sharpe ratio invariance under equal translation."""
        sr = ratios_module.sharpe_ratio(returns, required_return)
        sr_depressed = ratios_module.sharpe_ratio(returns - translation, required_return - translation)
        sr_raised = ratios_module.sharpe_ratio(returns + translation, required_return + translation)
        assert_almost_equal(sr, sr_depressed, DECIMAL_PLACES)
        assert_almost_equal(sr, sr_raised, DECIMAL_PLACES)

    # Translating the returns and required returns by the different amount
    # yields different sharpe ratios
    @parameterized.expand([(noise_uniform, 0, 0.0002, 0.0001), (noise_uniform, 0.005, 0.0001, 0.0002)])
    @pytest.mark.p1  # High: important property validation
    def test_sharpe_translation_diff(self, returns, required_return, translation_returns, translation_required):
        """Test Sharpe ratio changes under unequal translation."""
        sr = ratios_module.sharpe_ratio(returns, required_return)
        sr_depressed = ratios_module.sharpe_ratio(
            returns - translation_returns, required_return - translation_required
        )
        sr_raised = ratios_module.sharpe_ratio(returns + translation_returns, required_return + translation_required)
        assert sr != sr_depressed
        assert sr != sr_raised

    # Translating the required return inversely affects the sharpe ratio.
    @parameterized.expand([(noise_uniform, 0, 0.005), (noise, 0, 0.005)])
    @pytest.mark.p1  # High: important property validation
    def test_sharpe_translation_1(self, returns, required_return, translation):
        """Test Sharpe ratio inverse relationship with required return translation."""
        sr = ratios_module.sharpe_ratio(returns, required_return)
        sr_depressed = ratios_module.sharpe_ratio(returns, required_return - translation)
        sr_raised = ratios_module.sharpe_ratio(returns, required_return + translation)
        assert sr_depressed > sr
        assert sr > sr_raised

    # Returns of a wider range or larger standard deviation decreases the
    # sharpe ratio
    @parameterized.expand([(0.001, 0.002), (0.01, 0.02)])
    @pytest.mark.p1  # High: important property validation
    def test_sharpe_noise(self, small, large):
        """Test Sharpe ratio decreases with increased noise."""
        index = pd.date_range("2000-1-30", periods=1000, freq="D")
        smaller_normal = pd.Series(
            rand.normal(0.01, small, 1000),
            index=index,
        )
        larger_normal = pd.Series(
            rand.normal(0.01, large, 1000),
            index=index,
        )
        assert ratios_module.sharpe_ratio(smaller_normal, 0.001) > ratios_module.sharpe_ratio(larger_normal, 0.001)

    # ========================================================================
    # Sortino Ratio Tests
    # ========================================================================

    @parameterized.expand(
        [
            (empty_returns, 0.0, DAILY, np.nan),
            (one_return, 0.0, DAILY, np.nan),
            (mixed_returns, mixed_returns, DAILY, np.nan),
            (mixed_returns, 0.0, DAILY, 2.605531251673693),
            (mixed_returns, flat_line_1, DAILY, -1.3934779588919977),
            (positive_returns, 0.0, DAILY, np.inf),
            (negative_returns, 0.0, DAILY, -13.532743075043401),
            (simple_benchmark, 0.0, DAILY, np.inf),
            (weekly_returns, 0.0, WEEKLY, 1.1158901056866439),
            (monthly_returns, 0.0, MONTHLY, 0.53605626741889756),
            (df_simple, 0.0, DAILY, pd.Series([3.0639640966566306, 38.090963117002495], index=["one", "two"])),
            (df_week, 0.0, WEEKLY, pd.Series([1.3918264112070571, 17.303077589064618], index=["one", "two"])),
            (df_month, 0.0, MONTHLY, pd.Series([0.6686117809312383, 8.3121296084492844], index=["one", "two"])),
        ]
    )
    @pytest.mark.p0  # Critical: core financial metric
    def test_sortino(self, returns, required_return, period, expected):
        """Test Sortino ratio calculation."""
        sortino_ratio = ratios_module.sortino_ratio(returns, required_return=required_return, period=period)
        if isinstance(sortino_ratio, float):
            assert_almost_equal(sortino_ratio, expected, DECIMAL_PLACES)
        else:
            for i in range(sortino_ratio.size):
                assert_almost_equal(
                    sortino_ratio.iloc[i] if hasattr(sortino_ratio, "iloc") else sortino_ratio[i],
                    expected.iloc[i] if hasattr(expected, "iloc") else expected[i],
                    DECIMAL_PLACES,
                )

    # A large Sortino ratio indicates there is a low probability of a large
    # loss, therefore randomly changing values larger than required return to a
    # loss of 25 percent decreases the ratio.
    @parameterized.expand(
        [
            (noise_uniform, 0),
            (noise, 0),
        ]
    )
    @pytest.mark.p1  # High: important property validation
    def test_sortino_add_noise(self, returns, required_return):
        """Test Sortino ratio decreases when adding losses."""
        # Don't mutate global test state
        returns = returns.copy()
        sr_1 = ratios_module.sortino_ratio(returns, required_return)
        upside_values = returns[returns > required_return].index.tolist()
        # Add large losses at random upside locations
        loss_loc = rand.choice(upside_values, 2)
        returns[loss_loc[0]] = -0.01
        sr_2 = ratios_module.sortino_ratio(returns, required_return)
        returns[loss_loc[1]] = -0.01
        sr_3 = ratios_module.sortino_ratio(returns, required_return)
        assert sr_1 > sr_2
        assert sr_2 > sr_3

    # Similarly, randomly increasing some values below the required return to
    # the required return increases the ratio.
    @parameterized.expand([(noise_uniform, 0), (noise, 0)])
    @pytest.mark.p1  # High: important property validation
    def test_sortino_sub_noise(self, returns, required_return):
        """Test Sortino ratio increases when removing losses."""
        # Don't mutate global test state
        returns = returns.copy()
        sr_1 = ratios_module.sortino_ratio(returns, required_return)
        downside_values = returns[returns < required_return].index.tolist()
        # Replace some values below the required return to the required return
        loss_loc = rand.choice(downside_values, 2)
        returns[loss_loc[0]] = required_return
        sr_2 = ratios_module.sortino_ratio(returns, required_return)
        returns[loss_loc[1]] = required_return
        sr_3 = ratios_module.sortino_ratio(returns, required_return)
        assert sr_1 <= sr_2
        assert sr_2 <= sr_3

    # Translating the returns and required returns by the same amount
    # should not change the sortino ratio.
    @parameterized.expand([(noise_uniform, 0, 0.005), (noise_uniform, 0.005, 0.005)])
    @pytest.mark.p1  # High: important property validation
    def test_sortino_translation_same(self, returns, required_return, translation):
        """Test Sortino ratio invariance under equal translation."""
        sr = ratios_module.sortino_ratio(returns, required_return)
        sr_depressed = ratios_module.sortino_ratio(returns - translation, required_return - translation)
        sr_raised = ratios_module.sortino_ratio(returns + translation, required_return + translation)
        assert_almost_equal(sr, sr_depressed, DECIMAL_PLACES)
        assert_almost_equal(sr, sr_raised, DECIMAL_PLACES)

    # Translating the returns and required returns by the same amount
    # should not change the sortino ratio.
    @parameterized.expand([(noise_uniform, 0, 0, 0.001), (noise_uniform, 0.005, 0.001, 0)])
    @pytest.mark.p1  # High: important property validation
    def test_sortino_translation_diff(self, returns, required_return, translation_returns, translation_required):
        """Test Sortino ratio changes under unequal translation."""
        sr = ratios_module.sortino_ratio(returns, required_return)
        sr_depressed = ratios_module.sortino_ratio(
            returns - translation_returns, required_return - translation_required
        )
        sr_raised = ratios_module.sortino_ratio(returns + translation_returns, required_return + translation_required)
        assert sr != sr_depressed
        assert sr != sr_raised


# ========================================================================
# Module-level reference
# ========================================================================
EMPYRICAL_MODULE = empyrical
