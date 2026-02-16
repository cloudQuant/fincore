"""Tests for missing coverage in metrics modules.

This module covers edge cases and branches that were previously uncovered:
- alpha_beta.py: Various edge cases
- transactions.py: Edge cases for transaction analysis
- returns.py/yearly.py: Edge cases for return metrics
"""

import unittest

import numpy as np
import pandas as pd

from fincore.metrics import alpha_beta, transactions, yearly
from fincore.metrics import returns as returns_mod


class AlphaBetaMissingCoverageTestCase(unittest.TestCase):
    """Test cases for previously uncovered code paths in alpha_beta.py."""

    def test_alpha_beta_aligned_insufficient_data(self):
        """Test alpha_beta_aligned with insufficient data (line 408-411)."""
        returns = np.array([0.01])
        factor_returns = np.array([0.01])

        result = alpha_beta.alpha_beta_aligned(returns, factor_returns)
        self.assertTrue(np.isnan(result).all())

    def test_alpha_beta_aligned_zero_factor_variance(self):
        """Test alpha_beta_aligned with zero factor variance (line 419-422)."""
        returns = np.array([0.01, 0.02, 0.015])
        factor_returns = np.array([0.01, 0.01, 0.01])  # Zero variance

        result = alpha_beta.alpha_beta_aligned(returns, factor_returns)
        self.assertTrue(np.isnan(result).all())

    def test_annual_alpha_empty_returns(self):
        """Test annual_alpha with empty returns (line 526-527)."""
        returns = pd.Series([], dtype=float)
        factor_returns = pd.Series([], dtype=float)

        result = alpha_beta.annual_alpha(returns, factor_returns)
        self.assertEqual(len(result), 0)

    def test_annual_alpha_non_datetime_index(self):
        """Test annual_alpha with non-DatetimeIndex (line 529-530)."""
        returns = pd.Series([0.01, 0.02, 0.015])
        factor_returns = pd.Series([0.01, 0.015, 0.01])

        result = alpha_beta.annual_alpha(returns, factor_returns)
        self.assertEqual(len(result), 0)

    def test_annual_beta_empty_returns(self):
        """Test annual_beta with empty returns (line 583)."""
        returns = pd.Series([], dtype=float)
        factor_returns = pd.Series([], dtype=float)

        result = alpha_beta.annual_beta(returns, factor_returns)
        self.assertEqual(len(result), 0)

    def test_up_alpha_beta_no_up_periods(self):
        """Test up_alpha_beta when there are no up periods."""
        returns = pd.Series([0.01, 0.02, 0.015])
        factor_returns = pd.Series([-0.01, -0.02, -0.015])  # All negative

        result = alpha_beta.up_alpha_beta(returns, factor_returns)
        self.assertTrue(np.isnan(result).all())

    def test_down_alpha_beta_no_down_periods(self):
        """Test down_alpha_beta when there are no down periods."""
        returns = pd.Series([0.01, 0.02, 0.015])
        factor_returns = pd.Series([0.01, 0.02, 0.015])  # All positive

        result = alpha_beta.down_alpha_beta(returns, factor_returns)
        self.assertTrue(np.isnan(result).all())


class TransactionsMissingCoverageTestCase(unittest.TestCase):
    """Test cases for previously uncovered code paths in transactions.py."""

    def test_map_transaction_with_extra_fields(self):
        """Test map_transaction with additional fields."""
        txn = {
            "sid": 1,
            "amount": 100,
            "price": 10.0,
            "dt": pd.Timestamp("2020-01-01"),
            "commission": 1.0,
        }

        result = transactions.map_transaction(txn)
        self.assertIsNotNone(result)
        self.assertEqual(result["sid"], 1)
        self.assertEqual(result["amount"], 100)

    def test_make_transaction_frame_from_dict(self):
        """Test make_transaction_frame from list of dicts."""
        txns = [
            {"sid": 1, "amount": 100, "price": 10.0, "dt": pd.Timestamp("2020-01-01")},
            {"sid": 2, "amount": -50, "price": 20.0, "dt": pd.Timestamp("2020-01-02")},
        ]

        result = transactions.make_transaction_frame(txns)
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result.index, pd.DatetimeIndex)

    def test_adjust_returns_for_slippage_with_positions(self):
        """Test adjust_returns_for_slippage with proper inputs."""
        returns = pd.Series([0.01, 0.02, 0.015], index=pd.date_range("2020-01-01", periods=3))
        positions = pd.DataFrame(
            {
                "AAPL": [10000, 11000, 10500],
                "MSFT": [5000, 5500, 5250],
            },
            index=pd.date_range("2020-01-01", periods=3),
        )

        txn_df = pd.DataFrame(
            {
                "sid": [1, 1],
                "amount": [100, -50],
                "price": [10.0, 11.0],
                "dt": pd.date_range("2020-01-01", periods=2),
            }
        )
        txn_df = txn_df.set_index("dt")

        result = transactions.adjust_returns_for_slippage(returns, positions, txn_df, slippage_bps=1)
        self.assertIsNotNone(result)


class YearlyReturnsMissingCoverageTestCase(unittest.TestCase):
    """Test cases for previously uncovered code paths in returns.py/yearly.py."""

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
        # First value should be 1, last should be cumulative return
        self.assertAlmostEqual(result.iloc[0], 1.0, places=5)

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

        # When first value is 0, result should be all NaN
        self.assertTrue(result.isna().all())

    def test_normalize_with_empty_returns(self):
        """Test normalize with empty returns (line 298)."""
        returns = pd.Series([], dtype=float)

        result = returns_mod.normalize(returns)
        # Should return a copy of the empty series
        self.assertEqual(len(result), 0)


if __name__ == "__main__":
    unittest.main()
