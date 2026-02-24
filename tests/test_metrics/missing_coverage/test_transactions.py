"""Tests for missing coverage in transactions module.

Test cases for previously uncovered code paths in transactions.py.
Split from test_missing_coverage.py for maintainability.
"""

import unittest

import pandas as pd

from fincore.metrics import transactions


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
