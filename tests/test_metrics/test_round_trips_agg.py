"""Tests for agg_all_long_short function in round_trips module.

This file tests the aggregation function that computes statistics for
long/short trade categories.
"""

from __future__ import annotations

import pandas as pd
import pytest


class TestAggAllLongShort:
    """Test cases for agg_all_long_short function."""

    def test_agg_all_long_short_with_exception_handling(self):
        """Test agg_all_long_short with stats that raise exceptions."""
        from fincore.metrics.round_trips import agg_all_long_short

        # Create round trips dataframe
        df = pd.DataFrame(
            {
                "long": [True, False, True],
                "pnl": [10.0, -5.0, 15.0],
                "duration": [pd.Timedelta(days=1), pd.Timedelta(days=2), pd.Timedelta(days=3)],
                "rt_returns": [0.1, -0.05, 0.15],
            }
        )

        # Test with callable that raises exception
        stats_dict = {
            "mean": lambda x: x.mean(),
            "bad_stat": lambda x: 1 / 0,  # Will raise ZeroDivisionError
        }

        result = agg_all_long_short(df, "pnl", stats_dict)

        # Should contain NaN for bad_stat but still compute mean
        assert "bad_stat" in result.columns
        assert "mean" in result.columns
        assert result.loc["All trades", "mean"] == df["pnl"].mean()
        assert pd.isna(result.loc["All trades", "bad_stat"])

    def test_agg_all_long_short_with_string_methods(self):
        """Test agg_all_long_short with string method names."""
        from fincore.metrics.round_trips import agg_all_long_short

        df = pd.DataFrame(
            {
                "long": [True, False, True],
                "pnl": [10.0, -5.0, 15.0],
                "duration": [pd.Timedelta(days=1), pd.Timedelta(days=2), pd.Timedelta(days=3)],
                "rt_returns": [0.1, -0.05, 0.15],
            }
        )

        # Test with string method names
        stats_dict = {
            "mean": "mean",
            "sum": "sum",
            "count": "count",
        }

        result = agg_all_long_short(df, "pnl", stats_dict)

        # Verify results
        assert result.loc["All trades", "mean"] == df["pnl"].mean()
        assert result.loc["All trades", "sum"] == df["pnl"].sum()
        assert result.loc["All trades", "count"] == len(df)
        assert "long" in result.index
        assert "short" in result.index

    def test_agg_all_long_short_with_invalid_stat_type(self):
        """Test agg_all_long_short with non-callable, non-string stat type."""
        from fincore.metrics.round_trips import agg_all_long_short

        df = pd.DataFrame(
            {
                "long": [True, False],
                "pnl": [10.0, -5.0],
                "rt_returns": [0.1, -0.05],
            }
        )

        # Test with invalid stat type (int instead of callable or string)
        stats_dict = {
            "mean": "mean",
            "invalid": 42,  # Invalid type
        }

        result = agg_all_long_short(df, "pnl", stats_dict)

        # Should have NaN for invalid stat
        assert "invalid" in result.columns
        assert pd.isna(result.loc["All trades", "invalid"])

    def test_agg_all_long_short_exception_in_all_trades(self):
        """Test exception handling when computing 'All trades' statistics."""
        from fincore.metrics.round_trips import agg_all_long_short

        df = pd.DataFrame(
            {
                "long": [True, False],
                "pnl": [10.0, -5.0],
                "rt_returns": [0.1, -0.05],
            }
        )

        # Create a function that works for long/short but fails for all
        class SelectiveFailure:
            def __call__(self, data):
                if len(data) > 1:  # Will fail for 'All' but work for individual groups
                    raise ValueError("Too many items")
                return data.iloc[0] if len(data) > 0 else 0

        stats_dict = {
            "selective": SelectiveFailure(),
            "mean": "mean",
        }

        result = agg_all_long_short(df, "pnl", stats_dict)

        # 'All trades' should have NaN for selective stat
        assert pd.isna(result.loc["All trades", "selective"])
        # But mean should work
        assert not pd.isna(result.loc["All trades", "mean"])

    def test_agg_all_long_short_empty_dataframe(self):
        """Test agg_all_long_short with empty round trips."""
        from fincore.metrics.round_trips import agg_all_long_short

        df = pd.DataFrame({"long": [], "pnl": [], "rt_returns": []})

        stats_dict = {"mean": "mean", "sum": "sum"}

        result = agg_all_long_short(df, "pnl", stats_dict)

        # Should return empty DataFrame with correct structure
        assert isinstance(result, pd.DataFrame)
        assert "All trades" in result.index
        assert "mean" in result.columns
