"""Tests for alpha and beta basic calculations.

This module tests basic alpha, beta, and alpha_beta combined calculation.
Split from test_alpha_beta_core.py for maintainability.

Priority Markers:
- P0: Core alpha, beta calculations
- P1: Important property tests
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_almost_equal
from parameterized import parameterized
from scipy import stats
from unittest import TestCase

from fincore import empyrical
from fincore.empyrical import Empyrical

DECIMAL_PLACES = 8

# Pandas frequency alias compatibility
try:
    pd.date_range("2000-1-1", periods=1, freq="ME")
    MONTH_FREQ = "ME"
except ValueError:
    MONTH_FREQ = "M"


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


@pytest.mark.p1  # High: important alpha/beta tests
class TestAlphaBetaBasic(BaseTestCase):
    """Tests for basic alpha and beta calculations."""

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

    flat_line_1 = pd.Series(
        np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]) / 100,
        index=pd.date_range("2000-1-30", periods=9, freq="D"),
    )

    empty_returns = pd.Series(
        np.array([]) / 100,
        index=pd.date_range("2000-1-30", periods=0, freq="D")
    )

    one_return = pd.Series(
        np.array([1.0]) / 100,
        index=pd.date_range("2000-1-30", periods=1, freq="D")
    )

    # ========================================================================
    # Alpha and Beta Tests
    # ========================================================================

    @parameterized.expand([
        (empty_returns, empty_returns, (np.nan, np.nan)),
        (one_return, one_return, (np.nan, np.nan)),
        (mixed_returns[1:], negative_returns[1:], (-0.9997853834885004, -0.71296296296296313)),
        (mixed_returns, mixed_returns, (0.0, 1.0)),
        (mixed_returns, -mixed_returns, (0.0, -1.0)),
    ])
    @pytest.mark.p0  # Critical: core financial metric
    def test_alpha_beta(self, returns, benchmark, expected):
        """Test combined alpha and beta calculation."""
        alpha, beta = Empyrical(return_types=np.ndarray).alpha_beta(returns, benchmark)
        assert_almost_equal(alpha, expected[0], DECIMAL_PLACES)
        assert_almost_equal(beta, expected[1], DECIMAL_PLACES)

    # ========================================================================
    # Alpha Only Tests
    # ========================================================================

    @parameterized.expand([
        (empty_returns, simple_benchmark, np.nan),
        (one_return, one_return, np.nan),
        (mixed_returns, flat_line_1, np.nan),
        (mixed_returns, mixed_returns, 0.0),
        (mixed_returns, -mixed_returns, 0.0),
    ])
    @pytest.mark.p0  # Critical: core financial metric
    def test_alpha(self, returns, benchmark, expected):
        """Test alpha calculation."""
        observed = Empyrical.alpha(returns, benchmark)
        assert_almost_equal(observed, expected, DECIMAL_PLACES)

        if len(returns) == len(benchmark):
            # Compare to scipy linregress
            returns_arr = returns.values
            benchmark_arr = benchmark.values
            # Skip comparison with scipy when benchmark values are identical
            if np.all(benchmark_arr == benchmark_arr[0]):
                return
            mask = ~np.isnan(returns_arr) & ~np.isnan(benchmark_arr)
            slope, intercept, _, _, _ = stats.linregress(benchmark_arr[mask], returns_arr[mask])

            assert_almost_equal(observed, intercept * 252, DECIMAL_PLACES)

    # ========================================================================
    # Beta Only Tests
    # ========================================================================

    @parameterized.expand([
        (empty_returns, simple_benchmark, np.nan),
        (one_return, one_return, np.nan),
        (mixed_returns, flat_line_1, np.nan),
    ])
    @pytest.mark.p0  # Critical: core financial metric
    def test_beta(self, returns, benchmark, expected, decimal_places=DECIMAL_PLACES):
        """Test beta calculation with edge cases."""
        observed = Empyrical.beta(returns, benchmark)
        assert_almost_equal(observed, expected, decimal_places)

    @parameterized.expand([
        (empty_returns, empty_returns),
        (one_return, one_return),
        (mixed_returns[1:], simple_benchmark[1:]),
        (mixed_returns[1:], negative_returns[1:]),
        (mixed_returns, mixed_returns),
        (mixed_returns, -mixed_returns),
    ])
    def test_alpha_beta_equality(self, returns, benchmark):
        """Test alpha_beta returns same as individual alpha/beta calls."""
        alpha, beta = Empyrical(return_types=np.ndarray).alpha_beta(returns, benchmark)
        assert_almost_equal(alpha, Empyrical.alpha(returns, benchmark), DECIMAL_PLACES)
        assert_almost_equal(beta, Empyrical.beta(returns, benchmark), DECIMAL_PLACES)
