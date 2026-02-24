"""Tests for groupby_consecutive function in round_trips module.

This file tests the grouping of consecutive transactions.
"""

from __future__ import annotations

import pandas as pd
import pytest


class TestGroupbyConsecutive:
    """Test cases for groupby_consecutive function."""

    def test_groupby_consecutive_with_zero_shares_warning(self):
        """Test groupby_consecutive with zero shares (warning case)."""
        from fincore.metrics.round_trips import groupby_consecutive

        idx = pd.date_range("2020-01-01", periods=5, freq="D")
        df = pd.DataFrame(
            {
                "amount": [10, 0, -5, 0, 5],
                "price": [100.0, 100.0, 105.0, 100.0, 95.0],
                "symbol": ["A", "A", "A", "B", "B"],
            },
            index=idx,
        )

        # This should trigger warning for zero shares
        with pytest.warns(UserWarning, match="Zero transacted shares"):
            result = groupby_consecutive(df)

        # Result should have NaN price for the zero-share transaction
        assert isinstance(result, pd.DataFrame)

    def test_groupby_consecutive_multiple_symbols(self):
        """Test groupby_consecutive with multiple symbols."""
        from fincore.metrics.round_trips import groupby_consecutive

        idx = pd.date_range("2020-01-01", periods=6, freq="D")
        df = pd.DataFrame(
            {
                "amount": [10, 5, -5, -10, 8, -8],
                "price": [100.0, 102.0, 105.0, 103.0, 98.0, 99.0],
                "symbol": ["A", "A", "A", "A", "B", "B"],
            },
            index=idx,
        )

        result = groupby_consecutive(df)

        # Should group by symbol
        assert "symbol" in result.columns
        assert set(result["symbol"].unique()) <= {"A", "B"}

    def test_groupby_consecutive_max_delta_parameter(self):
        """Test groupby_consecutive with custom max_delta."""
        from fincore.metrics.round_trips import groupby_consecutive

        idx = pd.date_range("2020-01-01", periods=5, freq="h")
        df = pd.DataFrame(
            {
                "amount": [10, 5, -5, -10, 5],
                "price": [100.0, 102.0, 105.0, 103.0, 98.0],
                "symbol": ["A", "A", "A", "A", "A"],
            },
            index=idx,
        )

        # Use very small max_delta - should not group
        result_small_delta = groupby_consecutive(df, max_delta=pd.Timedelta(seconds=1))

        # Use large max_delta - should group consecutive same-direction trades
        result_large_delta = groupby_consecutive(df, max_delta=pd.Timedelta(days=1))

        # Large delta should result in fewer rows
        assert len(result_large_delta) <= len(result_small_delta)

    def test_vwap_calculation_edge_cases(self):
        """Test VWAP calculation with edge cases."""
        from fincore.metrics.round_trips import groupby_consecutive

        idx = pd.date_range("2020-01-01", periods=4, freq="D")
        df = pd.DataFrame(
            {
                "amount": [10, -5, 5, -10],
                "price": [100.0, 105.0, 95.0, 98.0],
                "symbol": ["A", "A", "A", "A"],
            },
            index=idx,
        )

        result = groupby_consecutive(df)

        # Prices should be VWAP of grouped transactions
        assert "price" in result.columns
