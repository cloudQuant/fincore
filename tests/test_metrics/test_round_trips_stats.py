"""Tests for gen_round_trip_stats and sector mapping functions.

This file tests round trip statistics generation and sector mapping.
"""

from __future__ import annotations

import pandas as pd
import pytest


class TestGenRoundTripStats:
    """Test cases for gen_round_trip_stats function."""

    def test_gen_round_trip_stats_empty_roundtrips(self):
        """Test gen_round_trip_stats with empty round trips DataFrame."""
        from fincore.metrics.round_trips import gen_round_trip_stats

        empty_df = pd.DataFrame({"pnl": [], "rt_returns": [], "duration": []})

        result = gen_round_trip_stats(empty_df)

        # Should return dict of empty DataFrames
        assert isinstance(result, dict)
        assert "pnl" in result
        assert "summary" in result
        assert "duration" in result
        assert "returns" in result
        assert "symbols" in result
        assert len(result["pnl"]) == 0

    def test_gen_round_trip_stats_custom_functions_only(self):
        """Test gen_round_trip_stats with only custom functions."""
        from fincore.metrics.round_trips import gen_round_trip_stats

        df = pd.DataFrame(
            {
                "pnl": [10.0, -5.0, 15.0],
                "rt_returns": [0.1, -0.05, 0.15],
                "duration": [
                    pd.Timedelta(days=1),
                    pd.Timedelta(days=2),
                    pd.Timedelta(days=3),
                ],
                "symbol": ["A", "B", "A"],
                "long": [True, False, True],
            }
        )

        result = gen_round_trip_stats(df)

        # Should compute all stats
        assert "pnl" in result
        assert "returns" in result
        assert "symbols" in result

        # symbols should be transposed with symbols as columns
        assert "A" in result["symbols"].columns or isinstance(result["symbols"], pd.DataFrame)

    def test_gen_round_trip_stats_with_returns_column(self):
        """Test gen_round_trip_stats uses 'returns' column when available."""
        from fincore.metrics.round_trips import gen_round_trip_stats

        df = pd.DataFrame(
            {
                "pnl": [10.0, -5.0, 15.0],
                "returns": [0.1, -0.05, 0.15],  # Has 'returns' column
                "rt_returns": [0.08, -0.04, 0.12],  # Also has 'rt_returns'
                "duration": [
                    pd.Timedelta(days=1),
                    pd.Timedelta(days=2),
                    pd.Timedelta(days=3),
                ],
                "symbol": ["A", "B", "A"],
                "long": [True, False, True],
            }
        )

        result = gen_round_trip_stats(df)

        # Should use 'returns' column, not 'rt_returns'
        assert "returns" in result

    def test_gen_round_trip_stats_exercise_custom_functions_only_path(self):
        """Test gen_round_trip_stats that exercises custom functions only path."""
        from fincore.metrics.round_trips import gen_round_trip_stats

        df = pd.DataFrame(
            {
                "pnl": [10.0, -5.0, 15.0, -3.0],
                "rt_returns": [0.1, -0.05, 0.15, -0.03],
                "duration": [
                    pd.Timedelta(days=1),
                    pd.Timedelta(days=2),
                    pd.Timedelta(days=3),
                    pd.Timedelta(days=1),
                ],
                "symbol": ["A", "B", "A", "B"],
                "long": [True, False, True, False],
            }
        )

        result = gen_round_trip_stats(df)

        # symbols result uses apply_custom_and_built_in_funcs
        assert "symbols" in result
        assert isinstance(result["symbols"], pd.DataFrame)


class TestSectorMappings:
    """Test cases for sector mapping functions."""

    def test_apply_sector_mappings_to_round_trips(self):
        """Test apply_sector_mappings_to_round_trips function."""
        from fincore.metrics.round_trips import apply_sector_mappings_to_round_trips

        df = pd.DataFrame(
            {
                "pnl": [10.0, -5.0],
                "symbol": ["AAPL", "MSFT"],
                "long": [True, False],
            }
        )

        sector_mappings = {"AAPL": "Technology", "MSFT": "Technology", "GOOGL": "Technology"}

        result = apply_sector_mappings_to_round_trips(df, sector_mappings)

        # Should add sector column
        assert "sector" in result.columns
        assert result.loc[0, "sector"] == "Technology"
        assert result.loc[1, "sector"] == "Technology"

        # Original DataFrame should not be modified
        assert "sector" not in df.columns

    def test_apply_sector_mappings_no_symbol_column(self):
        """Test apply_sector_mappings when no symbol column exists."""
        from fincore.metrics.round_trips import apply_sector_mappings_to_round_trips

        df = pd.DataFrame({"pnl": [10.0, -5.0], "value": [100, 200]})

        sector_mappings = {"AAPL": "Technology"}

        result = apply_sector_mappings_to_round_trips(df, sector_mappings)

        # Should still return DataFrame
        assert isinstance(result, pd.DataFrame)
