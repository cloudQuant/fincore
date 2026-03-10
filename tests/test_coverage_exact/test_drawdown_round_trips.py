"""Tests for drawdown and round_trips line coverage.

Part of test_exact_line_coverage.py split - Drawdown and round trips tests with P2 markers.
"""
from __future__ import annotations

import pandas as pd
import pytest

from fincore.metrics.drawdown import get_all_drawdowns
from fincore.metrics.round_trips import gen_round_trip_stats


@pytest.mark.p2
class TestDrawdownRoundTripsLineCoverage:
    """Test drawdown and round_trips edge cases for exact line coverage."""

    def test_drawdown_line_325(self):
        """drawdown.py line 325: break when returns or underwater is empty."""
        # Create returns that will result in empty underwater during iteration
        # After processing a drawdown that covers all remaining data
        idx = pd.date_range("2024-01-01", periods=10, freq="B")
        returns = pd.Series([0.01, -0.1, 0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], index=idx)

        result = get_all_drawdowns(returns)
        assert isinstance(result, list)

    def test_round_trips_line_417(self):
        """round_trips.py line 417: return without built_in_funcs."""
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
