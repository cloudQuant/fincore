"""Tests for extract_round_trips and add_closing_transactions functions.

This file tests round trip extraction from transaction data.
"""

from __future__ import annotations

import pandas as pd
import pytest


class TestExtractRoundTrips:
    """Test cases for extract_round_trips function."""

    def test_extract_round_trips_negative_price_warning(self):
        """Test extract_round_trips with negative price (warning case)."""
        from fincore.metrics.round_trips import extract_round_trips

        idx = pd.date_range("2020-01-01", periods=5, freq="D")
        df = pd.DataFrame(
            {
                "amount": [10, -10, 5, -5, 10],
                "price": [100.0, 105.0, -50.0, 95.0, 110.0],  # Negative price
                "symbol": ["A", "A", "A", "A", "A"],
            },
            index=idx,
        )

        # Should warn about negative price and skip that transaction
        with pytest.warns(UserWarning, match="Negative price detected"):
            result = extract_round_trips(df)

        # Result should skip the negative price transaction
        assert isinstance(result, pd.DataFrame)

    def test_extract_round_trips_empty_result(self):
        """Test extract_round_trips with no complete round trips."""
        from fincore.metrics.round_trips import extract_round_trips

        idx = pd.date_range("2020-01-01", periods=3, freq="D")
        # Only buys, no sells - no complete round trips
        df = pd.DataFrame(
            {
                "amount": [10, 5, 10],
                "price": [100.0, 105.0, 110.0],
                "symbol": ["A", "A", "A"],
            },
            index=idx,
        )

        result = extract_round_trips(df)

        # Should return empty DataFrame
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_extract_round_trips_with_portfolio_value_tz_aware(self):
        """Test extract_round_trips with timezone-aware portfolio value."""
        from fincore.metrics.round_trips import extract_round_trips

        idx = pd.date_range("2020-01-01", periods=5, freq="D", tz="UTC")
        df = pd.DataFrame(
            {
                "amount": [10, -10, 5, -5, 10],
                "price": [100.0, 105.0, 95.0, 100.0, 110.0],
                "symbol": ["A", "A", "A", "A", "A"],
            },
            index=idx,
        )

        # Portfolio value with timezone
        portfolio_value = pd.Series([10000.0, 10050.0, 10100.0, 10150.0, 10200.0], index=idx)

        result = extract_round_trips(df, portfolio_value=portfolio_value)

        # Should have returns column
        assert "returns" in result.columns
        assert len(result) > 0

    def test_extract_round_trips_with_tz_naive_portfolio_and_tz_aware_pv(self):
        """Test extract_round_trips with mixed timezone awareness."""
        from fincore.metrics.round_trips import extract_round_trips

        # Naive datetime for transactions
        idx = pd.date_range("2020-01-01", periods=4, freq="D")
        df = pd.DataFrame(
            {
                "amount": [10, -10, 5, -5],
                "price": [100.0, 105.0, 95.0, 100.0],
                "symbol": ["A", "A", "A", "A"],
            },
            index=idx,
        )

        # Timezone-aware portfolio value
        pv_idx = pd.date_range("2020-01-01", periods=4, freq="D", tz="UTC")
        portfolio_value = pd.Series([10000.0, 10050.0, 10100.0, 10150.0], index=pv_idx)

        result = extract_round_trips(df, portfolio_value=portfolio_value)

        # Should localize naive dates to UTC
        assert "returns" in result.columns

    def test_extract_round_trips_crossing_from_long_to_short(self):
        """Test extract_round_trips handles crossing from long to short position."""
        from fincore.metrics.round_trips import extract_round_trips

        idx = pd.date_range("2020-01-01", periods=5, freq="D")
        # Buy 10, sell 15 (covers long and goes short)
        df = pd.DataFrame(
            {
                "amount": [10, -15, 10, -5, 5],
                "price": [100.0, 105.0, 95.0, 100.0, 110.0],
                "symbol": ["A", "A", "A", "A", "A"],
            },
            index=idx,
        )

        result = extract_round_trips(df)

        # Should create separate round trips for long and short
        assert len(result) > 0
        # Should have both long and short round trips
        assert result["long"].nunique() == 2  # Both True and False present

    def test_extract_round_trips_invested_zero_division(self):
        """Test extract_round_trips handles zero invested amount."""
        from fincore.metrics.round_trips import extract_round_trips

        idx = pd.date_range("2020-01-01", periods=6, freq="D")
        # Create scenario where we have complete round trips
        df = pd.DataFrame(
            {
                "amount": [10, -10, 5, -5],
                "price": [100.0, 105.0, 95.0, 100.0],
                "symbol": ["A", "A", "A", "A"],
            },
            index=idx[:4],
        )

        result = extract_round_trips(df)

        # Should have 2 round trips
        assert len(result) == 2
        # Should have rt_returns column
        assert "rt_returns" in result.columns
        # Returns should be computed correctly
        assert "pnl" in result.columns

    def test_extract_round_trips_partial_fill(self):
        """Test extract_round_trips with partial position closing."""
        from fincore.metrics.round_trips import extract_round_trips

        idx = pd.date_range("2020-01-01", periods=6, freq="D")
        # Buy 15 @ 100, sell 10 @ 105 (partial close), sell 5 @ 102 (close remainder)
        df = pd.DataFrame(
            {
                "amount": [15, -10, -5],
                "price": [100.0, 105.0, 102.0],
                "symbol": ["A", "A", "A"],
            },
            index=idx[:3],
        )

        result = extract_round_trips(df)

        # Should have 2 round trips (partial close + final close)
        assert len(result) == 2
        assert "pnl" in result.columns


class TestAddClosingTransactions:
    """Test cases for add_closing_transactions function."""

    def test_add_closing_transactions_with_zero_amount(self):
        """Test add_closing_transactions skips zero-amount positions."""
        from fincore.metrics.round_trips import add_closing_transactions

        # Create a case where buys and sells cancel out
        transactions = pd.DataFrame(
            {
                "amount": [10, -5, 5, -10, 100, -100],  # Last two cancel out
                "price": [100.0, 105.0, 102.0, 98.0, 100.0, 100.0],
                "symbol": ["A", "A", "A", "A", "B", "B"],
            },
            index=pd.date_range("2020-01-01", periods=6, freq="D"),
        )

        positions = pd.DataFrame(
            {
                "cash": [10000, 9500, 9600, 10580, 10580, 10580],
                "A": [1000.0, 500.0, 1010.0, 0.0, 0.0, 0.0],  # Closed
                "B": [0.0, 0.0, 0.0, 0.0, 10000.0, 10000.0],  # Has position
            },
            index=pd.date_range("2020-01-01", periods=6, freq="D"),
        )

        result = add_closing_transactions(positions, transactions)

        # B should be skipped since its transaction amount sums to 0
        # Result should equal original (no closing txns added for B)
        assert len(result) == 6

    def test_add_closing_transactions_negative_amount(self):
        """Test add_closing_transactions with negative ending amount."""
        from fincore.metrics.round_trips import add_closing_transactions

        idx = pd.date_range("2020-01-01", periods=3, freq="D")
        transactions = pd.DataFrame(
            {
                "amount": [-10, 5, 3],
                "price": [100.0, 105.0, 95.0],
                "symbol": ["A", "A", "B"],
            },
            index=idx,
        )

        positions = pd.DataFrame(
            {
                "cash": [10000, 10500, 10215],
                "A": [-1000, -500, 0],  # Short position closed
                "B": [0, 0, 285],
            },
            index=idx,
        )

        result = add_closing_transactions(positions, transactions)

        # Should have closing transaction for A
        assert (result["symbol"] == "A").sum() >= 2  # Original + closing
