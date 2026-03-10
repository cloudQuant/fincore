"""Tests for 2D stats functions.

Tests functions that output DataFrames with both DataFrame and array inputs.

Split from test_2d_stats.py for maintainability.
"""

from __future__ import annotations

from unittest import TestCase

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_almost_equal
from parameterized import parameterized

try:
    from pandas.testing import assert_index_equal
except ImportError:
    from pandas.util.testing import assert_index_equal

from fincore.empyrical import Empyrical
from fincore.metrics import returns as returns_module
from tests.test_empyrical.stats.helpers.test_proxies import (
    PassArraysEmpyricalProxy,
    ReturnTypeEmpyricalProxy,
)

DECIMAL_PLACES = 8


class BaseTestCase(TestCase):
    """Base test case with index matching assertion."""

    def assert_indexes_match(self, result, expected):
        """Assert that two pandas objects have the same indices."""
        assert_index_equal(result.index, expected.index)

        if isinstance(result, pd.DataFrame) and isinstance(expected, pd.DataFrame):
            assert_index_equal(result.columns, expected.columns)


@pytest.mark.p1
class Test2DStatsDataFrames(BaseTestCase):
    """Tests for functions that are capable of outputting a DataFrame."""

    input_one = [np.nan, 0.01322056, 0.03063862, -0.01422057, -0.00489779, 0.01268925, -0.03357711, 0.01797036]
    input_two = [0.01846232, 0.00793951, -0.01448395, 0.00422537, -0.00339611, 0.03756813, 0.0151531, np.nan]

    expected_0_one = [0.000000, 0.013221, 0.044264, 0.029414, 0.024372, 0.037371, 0.002539, 0.020555]
    expected_0_two = [0.018462, 0.026548, 0.011680, 0.015955, 0.012504, 0.050542, 0.066461, 0.066461]

    expected_100_one = [100.000000, 101.322056, 104.426424, 102.941421, 102.437235, 103.737087, 100.253895, 102.055494]
    expected_100_two = [101.846232, 102.654841, 101.167994, 101.595466, 101.250436, 105.054226, 106.646123, 106.646123]

    df_index = pd.date_range("2000-1-30", periods=8, freq="D")

    df_input = pd.DataFrame({"one": pd.Series(input_one, index=df_index), "two": pd.Series(input_two, index=df_index)})

    df_empty = pd.DataFrame()

    df_0_expected = pd.DataFrame(
        {"one": pd.Series(expected_0_one, index=df_index), "two": pd.Series(expected_0_two, index=df_index)}
    )

    df_100_expected = pd.DataFrame(
        {"one": pd.Series(expected_100_one, index=df_index), "two": pd.Series(expected_100_two, index=df_index)}
    )

    @parameterized.expand(
        [
            (df_input, 0, df_0_expected),
            (df_input, 100, df_100_expected),
            (df_empty, 0, pd.DataFrame()),
        ]
    )
    def test_cum_returns_df(self, returns, starting_value, expected):
        """Test cumulative returns with DataFrame input."""
        cum_returns = returns_module.cum_returns(
            returns,
            starting_value=starting_value,
        )

        assert_almost_equal(
            np.asarray(cum_returns),
            np.asarray(expected),
            4,
        )

        self.assert_indexes_match(cum_returns, returns)

    @parameterized.expand(
        [
            (df_input, 0, df_0_expected.iloc[-1]),
            (df_input, 100, df_100_expected.iloc[-1]),
        ]
    )
    def test_cum_returns_final_df(self, returns, starting_value, expected):
        """Test final cumulative returns with DataFrame input."""
        return_types = (pd.Series, np.ndarray)
        result = self.empyrical(return_types=return_types).cum_returns_final(
            returns,
            starting_value=starting_value,
        )
        assert_almost_equal(np.array(result), expected, 5)
        self.assert_indexes_match(result, expected)

    @property
    def empyrical(self):
        """
        Returns a wrapper around the empyrical module so tests can
        perform input conversions or return type checks on each call to an
        empyrical function.

        Returns
        -------
        empyrical
        """
        return ReturnTypeEmpyricalProxy(self, pd.DataFrame)


@pytest.mark.p1
class Test2DStatsArrays(Test2DStatsDataFrames):
    """
    Tests pass np.ndarray inputs to empyrical and assert that outputs are of
    type np.ndarray.
    """

    @property
    def empyrical(self):
        return PassArraysEmpyricalProxy(self, np.ndarray)

    def assert_indexes_match(self, result, expected):
        pass
