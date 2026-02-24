"""Tests for metrics/round_trips.py edge cases.

Targets:
- metrics/round_trips.py: 417 - gen_round_trip_stats return path
"""

import pandas as pd


class TestRoundTripsReturnPath:
    """Test round_trips.py line 417."""

    def test_gen_round_trip_stats_return_path(self):
        """Line 417: return without built_in_funcs."""
        from fincore.metrics.round_trips import gen_round_trip_stats

        idx = pd.date_range("2024-01-01", periods=5, freq="B")
        round_trips = pd.DataFrame({
            "symbol": ["AAPL", "MSFT", "AAPL", "MSFT", "GOOG"],
            "pnl": [100, -50, 75, 25, -30],
            "returns": [0.01, -0.005, 0.008, 0.002, -0.003],
            "duration": [5, 3, 4, 2, 6],
            "long": [True, False, True, False, True],
        }, index=idx)

        result = gen_round_trip_stats(round_trips)
        assert isinstance(result, dict)
        assert "pnl" in result
