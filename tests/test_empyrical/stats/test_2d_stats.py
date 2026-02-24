"""Tests for 2D stats functions and proxy classes.

This module tests functions that output DataFrames and proxy classes
for type checking in tests.

Split from test_helpers.py for maintainability.
"""
from __future__ import annotations

import pytest
from copy import copy
from functools import wraps
from operator import attrgetter
from unittest import SkipTest, TestCase

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose, assert_almost_equal
from pandas.core.generic import NDFrame
from parameterized import parameterized

try:
    from pandas.testing import assert_index_equal
except ImportError:
    from pandas.util.testing import assert_index_equal

from fincore import empyrical
from fincore.empyrical import Empyrical
from fincore.metrics import returns as returns_module

DECIMAL_PLACES = 8


class BaseTestCase(TestCase):
    """Base test case with index matching assertion."""

    def assert_indexes_match(self, result, expected):
        """Assert that two pandas objects have the same indices."""
        assert_index_equal(result.index, expected.index)

        if isinstance(result, pd.DataFrame) and isinstance(expected, pd.DataFrame):
            assert_index_equal(result.columns, expected.columns)


@pytest.mark.p1  # High: important DataFrame-based metrics
class Test2DStats(BaseTestCase):
    """Tests for functions that are capable of outputting a DataFrame."""

    input_one = [np.nan, 0.01322056, 0.03063862, -0.01422057, -0.00489779, 0.01268925, -0.03357711, 0.01797036]
    input_two = [0.01846232, 0.00793951, -0.01448395, 0.00422537, -0.00339611, 0.03756813, 0.0151531, np.nan]

    expected_0_one = [0.000000, 0.013221, 0.044264, 0.029414, 0.024372, 0.037371, 0.002539, 0.020555]
    expected_0_two = [0.018462, 0.026548, 0.011680, 0.015955, 0.012504, 0.050542, 0.066461, 0.066461]

    expected_100_one = [100.000000, 101.322056, 104.426424, 102.941421, 102.437235, 103.737087, 100.253895, 102.055494]
    expected_100_two = [101.846232, 102.654841, 101.167994, 101.595466, 101.250436, 105.054226, 106.646123, 106.646123]

    df_index = pd.date_range("2000-1-30", periods=8, freq="D")

    df_input = pd.DataFrame({
        "one": pd.Series(input_one, index=df_index),
        "two": pd.Series(input_two, index=df_index)
    })

    df_empty = pd.DataFrame()

    df_0_expected = pd.DataFrame({
        "one": pd.Series(expected_0_one, index=df_index),
        "two": pd.Series(expected_0_two, index=df_index)
    })

    df_100_expected = pd.DataFrame({
        "one": pd.Series(expected_100_one, index=df_index),
        "two": pd.Series(expected_100_two, index=df_index)
    })

    @parameterized.expand([
        (df_input, 0, df_0_expected),
        (df_input, 100, df_100_expected),
        (df_empty, 0, pd.DataFrame()),
    ])
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

    @parameterized.expand([
        (df_input, 0, df_0_expected.iloc[-1]),
        (df_input, 100, df_100_expected.iloc[-1]),
    ])
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


@pytest.mark.p1  # High: important array-based metrics
class Test2DStatsArrays(Test2DStats):
    """
    Tests pass np.ndarray inputs to empyrical and assert that outputs are of
    type np.ndarray.
    """

    @property
    def empyrical(self):
        return PassArraysEmpyricalProxy(self, np.ndarray)

    def assert_indexes_match(self, result, expected):
        pass


# ========================================================================
# Proxy Classes for Type Checking
# ========================================================================

class ReturnTypeEmpyricalProxy:
    """
    A wrapper around the empyrical module which, on each function call, asserts
    that the type of the return value is in a given set.

    Also asserts that inputs were not modified by the empyrical function call.

    Calling an instance with kwargs will return a new copy with those
    attributes overridden.
    """

    def __init__(self, test_case, return_types):
        self._test_case = test_case
        self._return_types = return_types

    def __call__(self, **kwargs):
        dupe = copy(self)

        for k, v in kwargs.items():
            attr = "_" + k
            if hasattr(dupe, attr):
                setattr(dupe, attr, v)

        return dupe

    def __copy__(self):
        newone = type(self).__new__(type(self))
        newone.__dict__.update(self.__dict__)
        return newone

    def __getattr__(self, item):
        emp = Empyrical()
        func = getattr(emp, item)
        return self._check_input_not_mutated(self._check_return_type(func))

    def _check_return_type(self, func):
        @wraps(func)
        def check_return_type(*args, **kwargs):
            result = func(*args, **kwargs)
            if isinstance(result, tuple):
                tuple_result = result
            else:
                tuple_result = (result,)

            for r in tuple_result:
                self._test_case.assertIsInstance(r, self._return_types)
            return result

        return check_return_type

    def _check_input_not_mutated(self, func):
        @wraps(func)
        def check_not_mutated(*args, **kwargs):
            # Copy inputs to compare them to originals later.
            arg_copies = [
                (i, arg.copy())
                for i, arg in enumerate(args)
                if isinstance(arg, (NDFrame, np.ndarray))
            ]
            kwarg_copies = {
                k: v.copy()
                for k, v in kwargs.items()
                if isinstance(v, (NDFrame, np.ndarray))
            }

            result = func(*args, **kwargs)

            # Check that inputs weren't mutated by func.
            for i, arg_copy in arg_copies:
                assert_allclose(
                    args[i],
                    arg_copy,
                    atol=0.5 * 10 ** (-DECIMAL_PLACES),
                    err_msg=f"Input 'arg {i}' mutated by {func.__name__}",
                )
            for kwarg_name, kwarg_copy in kwarg_copies.items():
                assert_allclose(
                    kwargs[kwarg_name],
                    kwarg_copy,
                    atol=0.5 * 10 ** (-DECIMAL_PLACES),
                    err_msg=f"Input '{kwarg_name}' mutated by {func.__name__}",
                )

            return result

        return check_not_mutated


class ConvertPandasEmpyricalProxy(ReturnTypeEmpyricalProxy):
    """
    A ReturnTypeEmpyricalProxy which also converts pandas NDFrame inputs to
    empyrical functions according to the given conversion method.

    Calling an instance with a truthy pandas_only will return a new instance
    which will skip the test when an empyrical function is called.
    """

    def __init__(self, test_case, return_types, convert, pandas_only=False):
        super().__init__(test_case, return_types)
        self._convert = convert
        self._pandas_only = pandas_only

    def __getattr__(self, item):
        if self._pandas_only:
            raise SkipTest(
                "empyrical.%s expects pandas-only inputs that have dt indices/labels" % item
            )

        func = super().__getattr__(item)

        @wraps(func)
        def convert_args(*args, **kwargs):
            args = [
                self._convert(arg) if isinstance(arg, NDFrame) else arg
                for arg in args
            ]
            kwargs = {
                k: self._convert(v) if isinstance(v, NDFrame) else v
                for k, v in kwargs.items()
            }
            return func(*args, **kwargs)

        return convert_args


class PassArraysEmpyricalProxy(ConvertPandasEmpyricalProxy):
    """
    A ConvertPandasEmpyricalProxy which converts NDFrame inputs to empyrical
    functions to numpy arrays.

    Calls the underlying
    empyrical.[alpha|beta|alpha_beta]_aligned functions directly, instead of
    the wrappers which align Series first.
    """

    def __init__(self, test_case, return_types):
        super().__init__(
            test_case,
            return_types,
            attrgetter("values"),
        )

    def __getattr__(self, item):
        if item in (
            "alpha",
            "beta",
            "alpha_beta",
            "beta_fragility_heuristic",
            "gpd_risk_estimates",
        ):
            item += "_aligned"

        return super().__getattr__(item)


# Module-level reference
EMPYRICAL_MODULE = empyrical
