"""Tests for yearly breakdown metrics.

This module tests annual return by year, Sharpe ratio by year, information ratio by year,
annual volatility by year, and max drawdown by year.

Split from test_stats.py to improve maintainability.

Priority Markers:
- P1: All yearly breakdown tests (important reporting features)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_almost_equal
from parameterized import parameterized
from unittest import TestCase

from fincore import empyrical
from fincore.empyrical import Empyrical

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


class TestYearlyBreakdown(BaseTestCase):
    """Tests for yearly breakdown calculations."""

    # Test data - common series
    simple_benchmark = pd.Series(
        np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]) / 100,
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
    # Annual Return By Year Tests
    # ========================================================================

    @parameterized.expand(
        [
            (empty_returns, {}),
            (one_return, {2000: 11.274002099240244}),  # Single return gets annualized
            (mixed_returns, {2000: 1.913592537319458}),
        ]
    )
    @pytest.mark.p1  # High: important reporting feature
    def test_annual_return_by_year(self, returns, expected):
        """Test annual return breakdown by year."""
        result = Empyrical.annual_return_by_year(returns)
        if len(expected) == 0:
            assert len(result) == 0 or all(
                np.isnan(v) for v in (result.values if hasattr(result, "values") else result)
            )
        else:
            # For numpy arrays, we can only compare values at position 0 since all data is in one year
            if isinstance(result, np.ndarray):
                for i, (year, exp_val) in enumerate(expected.items()):
                    if i < len(result):
                        assert_almost_equal(result[i], exp_val, DECIMAL_PLACES)
            else:
                for year, exp_val in expected.items():
                    if year in result.index:
                        assert_almost_equal(result[year], exp_val, DECIMAL_PLACES)

    # ========================================================================
    # Sharpe Ratio By Year Tests
    # ========================================================================

    @parameterized.expand(
        [
            (empty_returns, 0.0, {}),
            (one_return, 0.0, {}),
            (mixed_returns, 0.0, {2000: 1.7238613961706866}),
        ]
    )
    @pytest.mark.p1  # High: important reporting feature
    def test_sharpe_ratio_by_year(self, returns, risk_free, expected):
        """Test Sharpe ratio breakdown by year."""
        result = Empyrical.sharpe_ratio_by_year(returns, risk_free=risk_free)
        if len(expected) == 0:
            assert len(result) == 0 or all(
                np.isnan(v) for v in (result.values if hasattr(result, "values") else result)
            )
        else:
            # For numpy arrays, we can only compare values at position 0 since all data is in one year
            if isinstance(result, np.ndarray):
                for i, (year, exp_val) in enumerate(expected.items()):
                    if i < len(result):
                        assert_almost_equal(result[i], exp_val, DECIMAL_PLACES)
            else:
                for year, exp_val in expected.items():
                    if year in result.index:
                        assert_almost_equal(result[year], exp_val, DECIMAL_PLACES)

    # ========================================================================
    # Information Ratio By Year Tests
    # ========================================================================

    @parameterized.expand(
        [
            (empty_returns, simple_benchmark, {}),
            (one_return, one_return, {}),
            (mixed_returns, simple_benchmark, {2000: 0.34111411441060574}),
        ]
    )
    @pytest.mark.p1  # High: important reporting feature
    def test_information_ratio_by_year(self, returns, factor_returns, expected):
        """Test information ratio breakdown by year."""
        result = Empyrical.information_ratio_by_year(returns, factor_returns)
        if len(expected) == 0:
            assert len(result) == 0 or all(
                np.isnan(v) for v in (result.values if hasattr(result, "values") else result)
            )
        else:
            # For numpy arrays, we can only compare values at position 0 since all data is in one year
            if isinstance(result, np.ndarray):
                for i, (year, exp_val) in enumerate(expected.items()):
                    if i < len(result):
                        assert_almost_equal(result[i], exp_val, DECIMAL_PLACES)
            else:
                for year, exp_val in expected.items():
                    if year in result.index:
                        assert_almost_equal(result[year], exp_val, DECIMAL_PLACES)

    # ========================================================================
    # Annual Volatility By Year Tests
    # ========================================================================

    @parameterized.expand(
        [
            (empty_returns, {}),
            (one_return, {}),  # Single return has NaN volatility (no variance)
            (mixed_returns, {2000: 0.9136465399704637}),
        ]
    )
    @pytest.mark.p1  # High: important reporting feature
    def test_annual_volatility_by_year(self, returns, expected):
        """Test annual volatility breakdown by year."""
        result = Empyrical.annual_volatility_by_year(returns)
        if len(expected) == 0:
            assert len(result) == 0 or all(
                np.isnan(v) for v in (result.values if hasattr(result, "values") else result)
            )
        else:
            # For numpy arrays, we can only compare values at position 0 since all data is in one year
            if isinstance(result, np.ndarray):
                for i, (year, exp_val) in enumerate(expected.items()):
                    if i < len(result):
                        assert_almost_equal(result[i], exp_val, DECIMAL_PLACES)
            else:
                for year, exp_val in expected.items():
                    if year in result.index:
                        assert_almost_equal(result[year], exp_val, DECIMAL_PLACES)

    # ========================================================================
    # Max Drawdown By Year Tests
    # ========================================================================

    @parameterized.expand(
        [
            (empty_returns, {}),
            (one_return, {2000: 0.0}),  # Single return has no drawdown
            (mixed_returns, {2000: -0.1}),
        ]
    )
    @pytest.mark.p1  # High: important reporting feature
    def test_max_drawdown_by_year(self, returns, expected):
        """Test max drawdown breakdown by year."""
        result = Empyrical.max_drawdown_by_year(returns)
        if len(expected) == 0:
            assert len(result) == 0 or all(
                np.isnan(v) for v in (result.values if hasattr(result, "values") else result)
            )
        else:
            # For numpy arrays, we can only compare values at position 0 since all data is in one year
            if isinstance(result, np.ndarray):
                for i, (year, exp_val) in enumerate(expected.items()):
                    if i < len(result):
                        assert_almost_equal(result[i], exp_val, DECIMAL_PLACES)
            else:
                for year, exp_val in expected.items():
                    if year in result.index:
                        assert_almost_equal(result[year], exp_val, DECIMAL_PLACES)


# ========================================================================
# Module-level reference
# ========================================================================
EMPYRICAL_MODULE = empyrical
