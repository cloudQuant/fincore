"""Tests for missing coverage in yearly module.

Test cases for previously uncovered code paths in yearly.py.
Split from test_missing_coverage.py for maintainability.
"""

import unittest

import numpy as np
import pandas as pd

from fincore.metrics import yearly


class YearlyReturnsMissingCoverageTestCase(unittest.TestCase):
    """Test cases for previously uncovered code paths in returns.py/yearly.py."""

    def test_annual_return_empty(self):
        """Test annual_return with empty returns (line 236)."""
        returns = pd.Series([], dtype=float)

        result = yearly.annual_return(returns)
        self.assertTrue(np.isnan(result))

    def test_annual_volatility_by_year_with_datetime(self):
        """Test annual_volatility_by_year with proper datetime index."""
        index = pd.date_range("2020-01-01", periods=400, freq="D")
        returns = pd.Series([0.01] * 400, index=index)

        result = yearly.annual_volatility_by_year(returns)
        self.assertGreater(len(result), 0)


class YearlyModuleAdditionalCoverageTestCase(unittest.TestCase):
    """Test cases for yearly.py module coverage gaps."""

    def test_annual_return_with_series_negative_ending_value(self):
        """Test annual_return with Series containing negative ending values (lines 71-77)."""
        index = pd.date_range("2020-01-01", periods=10, freq="D")
        returns = pd.Series([-0.5] * 10, index=index)

        result = yearly.annual_return(returns)
        self.assertEqual(result, -1.0)

    def test_annual_return_with_dataframe(self):
        """Test annual_return with DataFrame input (lines 71-77)."""
        index = pd.date_range("2020-01-01", periods=100, freq="D")
        returns = pd.DataFrame(
            {
                "col1": [0.01] * 50 + [-0.01] * 50,
                "col2": [0.005] * 100,
            },
            index=index,
        )

        result = yearly.annual_return(returns)
        self.assertIsInstance(result, pd.Series)
        self.assertEqual(len(result), 2)

    def test_annual_active_return_with_nan_result(self):
        """Test annual_active_return when either result is NaN (line 236)."""
        returns = pd.Series([], dtype=float, index=pd.DatetimeIndex([]))
        factor_returns = pd.Series([], dtype=float, index=pd.DatetimeIndex([]))

        result = yearly.annual_active_return(returns, factor_returns)
        self.assertTrue(np.isnan(result))

    def test_annual_active_return_by_year_non_datetime_index(self):
        """Test annual_active_return_by_year with non-DatetimeIndex (line 264)."""
        returns = pd.Series([0.01, 0.02, 0.015])
        factor_returns = pd.Series([0.005, 0.01, 0.01])

        result = yearly.annual_active_return_by_year(returns, factor_returns)
        self.assertEqual(len(result), 0)

    def test_annual_active_return_by_year_empty_result(self):
        """Test annual_active_return_by_year with no matching years (line 278)."""
        returns_idx = pd.date_range("2020-01-01", periods=10, freq="D")
        returns = pd.Series([0.01] * 10, index=returns_idx)

        factor_idx = pd.date_range("2021-01-01", periods=10, freq="D")
        factor_returns = pd.Series([0.005] * 10, index=factor_idx)

        result = yearly.annual_active_return_by_year(returns, factor_returns)
        self.assertEqual(len(result), 0)

    def test_annual_return_with_numpy_array_negative_ending(self):
        """Test annual_return with numpy array and negative ending value (line 77)."""
        returns = np.array([[-0.5] * 10, [0.01] * 10])

        result = yearly.annual_return(returns)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result[0], -1.0)

    def test_annual_active_return_with_nan_annual_values(self):
        """Test annual_active_return when annual_return produces NaN (line 236)."""
        returns = pd.Series([0.01], index=pd.date_range("2020-01-01", periods=1))
        factor_returns = pd.Series([0.01], index=pd.date_range("2020-01-02", periods=1))

        result = yearly.annual_active_return(returns, factor_returns)
        self.assertIsNotNone(result)
