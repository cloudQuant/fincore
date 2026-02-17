"""Tests for transactions module - edge cases for full coverage."""

import pandas as pd
import pytest

from fincore.metrics import transactions


class TestTransactionsEdgeCases:
    """Test transactions module edge cases."""

    def test_get_max_days_to_liquidate_by_ticker_with_last_n_days(self):
        """Test get_max_days_to_liquidate_by_ticker with last_n_days parameter (line 138)."""
        # Create sample positions and market data
        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        positions = pd.DataFrame(
            [
                [100, 200],
                [110, 210],
                [105, 215],
                [108, 218],
                [102, 212],
                [107, 217],
                [103, 213],
                [109, 219],
                [104, 214],
                [106, 216],
            ],
            index=dates,
            columns=["A", "B"],
        )
        positions["cash"] = 1000

        # Create market data with price and volume
        volume = pd.DataFrame(
            {"A": [10000] * 10, "B": [20000] * 10},
            index=dates,
        )
        volume.index.name = "dt"
        price = pd.DataFrame(
            {"A": [100] * 10, "B": [200] * 10},
            index=dates,
        )
        price.index.name = "dt"
        market_data = {"price": price, "volume": volume}

        # This should trigger the last_n_days slicing
        result = transactions.get_max_days_to_liquidate_by_ticker(
            positions,
            market_data,
            last_n_days=5,
        )
        assert isinstance(result, pd.DataFrame)

    def test_make_transaction_frame_with_dataframe(self):
        """Test make_transaction_frame with DataFrame input (line 287)."""
        # Create a DataFrame directly with datetime index
        df_txn = pd.DataFrame(
            {
                "sid": [1, 2, 3],
                "amount": [100, 200, 150],
                "price": [10.0, 20.0, 15.0],
                "symbol": ["A", "B", "C"],
            },
            index=pd.date_range("2020-01-01", periods=3),
        )

        result = transactions.make_transaction_frame(df_txn)
        # Should return the same DataFrame when input is DataFrame
        assert isinstance(result, pd.DataFrame)

    def test_get_turnover_invalid_denominator(self):
        """Test get_turnover with invalid denominator parameter (line 391)."""
        # Create sample positions and transactions
        dates = pd.date_range("2020-01-01", periods=5, freq="D")
        positions = pd.DataFrame(
            [[100, 200], [110, 210], [105, 215], [108, 218], [102, 212]],
            index=dates,
            columns=["A", "B"],
        )
        positions["cash"] = 1000

        # Create proper transactions DataFrame with datetime index
        transactions_df = pd.DataFrame(
            {
                "sid": [1] * 5,
                "amount": [10, -10, 20, -20, 15],
                "price": [100] * 5,
                "symbol": ["A"] * 5,
            },
            index=dates,
        )

        # Test with invalid denominator
        with pytest.raises(ValueError, match="Unexpected value for denominator"):
            transactions.get_turnover(
                positions,
                transactions_df,
                denominator="invalid_value",
            )
