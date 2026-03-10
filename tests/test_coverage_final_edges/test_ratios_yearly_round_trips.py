"""Final edge case tests for ratios, yearly, round_trips coverage.

Part of test_final_coverage_edges.py split - Ratios and yearly tests with P2 markers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fincore.metrics import ratios, round_trips, yearly


@pytest.mark.p2
class TestMarRatioFinalEdgeCase:
    """Test mar_ratio edge case for line 417."""

    def test_mar_ratio_all_nan_after_cleaning(self):
        """Line 417: len(returns_clean) < 1."""
        # All NaN values become empty after cleaning
        returns = pd.Series([np.nan, np.nan, np.nan])
        result = ratios.mar_ratio(returns)
        assert np.isnan(result)


@pytest.mark.p2
class TestAnnualActiveReturnFinalEdgeCase:
    """Test annual_active_return edge case for line 236."""

    def test_annual_active_return_nan_benchmark_annual(self):
        """Line 236: either annual return is NaN."""
        # Empty aligned series produces NaN annual returns
        returns = pd.Series([], dtype=float, index=pd.DatetimeIndex([]))
        factor_returns = pd.Series([], dtype=float, index=pd.DatetimeIndex([]))
        result = yearly.annual_active_return(returns, factor_returns)
        assert np.isnan(result)


@pytest.mark.p2
class TestRoundTripsFinalEdgeCase:
    """Test gen_round_trip_stats edge case for line 417."""

    def test_gen_round_trip_stats_without_built_in_funcs(self):
        """Line 417: return without built_in_funcs concat path."""
        # Create round trips that will hit the return without built_in_funcs
        idx = pd.date_range("2024-01-01", periods=5, freq="B")
        round_trips_data = pd.DataFrame(
            {
                "symbol": ["AAPL", "MSFT", "AAPL", "MSFT", "GOOG"],
                "pnl": [100, -50, 75, 25, -30],
                "returns": [0.01, -0.005, 0.008, 0.002, -0.003],
                "duration": [5, 3, 4, 2, 6],
                "long": [True, False, True, False, True],
            },
            index=idx,
        )

        result = round_trips.gen_round_trip_stats(round_trips_data)
        # Should return dict with stats
        assert isinstance(result, dict)
        assert "pnl" in result
        assert "summary" in result
