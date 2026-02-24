"""Tests for missing coverage in returns module.

Test cases for previously uncovered code paths in returns.py.
Split from test_missing_coverage.py for maintainability.
"""

import unittest

import numpy as np
import pandas as pd

from fincore.metrics import returns as returns_mod


class ReturnsMissingCoverageTestCase(unittest.TestCase):
    """Test cases for previously uncovered code paths in returns.py."""

    def test_aggregate_returns_empty(self):
        """Test aggregate_returns with empty returns (line 246 in returns.py)."""
        returns = pd.Series([], dtype=float, index=pd.DatetimeIndex([]))

        result = returns_mod.aggregate_returns(returns, "yearly")
        self.assertEqual(len(result), 0)

    def test_aggregate_returns_weekly(self):
        """Test aggregate_returns with weekly frequency."""
        index = pd.date_range("2020-01-01", periods=14, freq="D")
        returns = pd.Series([0.01] * 14, index=index)

        result = returns_mod.aggregate_returns(returns, "weekly")
        self.assertGreater(len(result), 0)

    def test_aggregate_returns_monthly(self):
        """Test aggregate_returns with monthly frequency."""
        index = pd.date_range("2020-01-01", periods=60, freq="D")
        returns = pd.Series([0.01] * 60, index=index)

        result = returns_mod.aggregate_returns(returns, "monthly")
        self.assertEqual(len(result), 2)  # 2 months

    def test_aggregate_returns_quarterly(self):
        """Test aggregate_returns with quarterly frequency."""
        index = pd.date_range("2020-01-01", periods=180, freq="D")
        returns = pd.Series([0.01] * 180, index=index)

        result = returns_mod.aggregate_returns(returns, "quarterly")
        self.assertEqual(len(result), 2)  # 2 quarters

    def test_aggregate_returns_yearly(self):
        """Test aggregate_returns with yearly frequency."""
        index = pd.date_range("2020-01-01", periods=400, freq="D")
        returns = pd.Series([0.01] * 400, index=index)

        result = returns_mod.aggregate_returns(returns, "yearly")
        self.assertEqual(len(result), 2)  # 2 years

    def test_cum_returns_final_with_start_value(self):
        """Test cum_returns_final with non-1 starting value (line 298)."""
        returns = pd.Series([0.01, 0.02, -0.01, 0.03])

        result = returns_mod.cum_returns_final(returns, starting_value=100)
        expected = 100 * 1.01 * 1.02 * 0.99 * 1.03
        self.assertAlmostEqual(result, expected, places=5)

    def test_normalize_returns(self):
        """Test normalize function."""
        returns = pd.Series([0.01, 0.02, -0.01, 0.03])

        result = returns_mod.normalize(returns)
        self.assertAlmostEqual(result.iloc[0], 1.0, places=5)

    def test_aggregate_returns_non_datetime_index(self):
        """Test aggregate_returns with non-DatetimeIndex (line 246)."""
        returns = pd.Series([0.01, 0.02, 0.015])

        with self.assertRaises(ValueError) as cm:
            returns_mod.aggregate_returns(returns, "yearly")
        self.assertIn("DatetimeIndex", str(cm.exception))

    def test_aggregate_returns_invalid_convert_to(self):
        """Test aggregate_returns with invalid convert_to value (line 259)."""
        index = pd.date_range("2020-01-01", periods=10, freq="D")
        returns = pd.Series([0.01] * 10, index=index)

        with self.assertRaises(ValueError) as cm:
            returns_mod.aggregate_returns(returns, "invalid")
        self.assertIn("convert_to must be", str(cm.exception))

    def test_normalize_with_zero_first_value(self):
        """Test normalize with first value equal to 0 (lines 302-305)."""
        returns = pd.Series([0.0, 0.01, 0.02, 0.015])

        import warnings

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = returns_mod.normalize(returns)

        self.assertTrue(result.isna().all())

    def test_normalize_with_empty_returns(self):
        """Test normalize with empty returns (line 298)."""
        returns = pd.Series([], dtype=float)

        result = returns_mod.normalize(returns)
        self.assertEqual(len(result), 0)
