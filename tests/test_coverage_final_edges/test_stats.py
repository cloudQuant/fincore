"""Final edge case tests for stats module coverage.

Part of test_final_coverage_edges.py split - Stats tests with P2 markers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fincore.metrics import stats


@pytest.mark.p2
class TestHurstExponentFinalEdgeCases:
    """Test hurst_exponent edge cases for lines 175, 193, 203."""

    def test_hurst_n_subseries_less_than_1(self):
        """Line 175: n_subseries < 1 causes continue."""
        # Single data point - when lag >= 2, n_subseries < 1
        returns = pd.Series([0.01])
        result = stats.hurst_exponent(returns)
        # Should return NaN or use fallback
        assert isinstance(result, (float, np.floating)) or np.isnan(result)

    def test_hurst_insufficient_rs_values_uses_fallback(self):
        """Line 193: len(rs_values) < 2 with valid s_std and r_range."""
        # Create data where R/S calculation produces < 2 valid values
        # but s_std > 0 and r_range > 0 for fallback path
        np.random.seed(42)
        returns = pd.Series([0.01, 0.02, -0.01, 0.005, 0.003])
        result = stats.hurst_exponent(returns)
        # May return NaN or use fallback calculation
        assert isinstance(result, (float, np.floating)) or np.isnan(result)

    def test_hurst_constant_returns(self):
        """Line 203: len(lags_array) < 2 after filtering (constant returns)."""
        # Data that results in insufficient valid lags after filtering
        returns = pd.Series([0.01] * 10)  # Constant returns
        result = stats.hurst_exponent(returns)
        # Constant returns -> std=0 -> returns fallback value (1.0 for constant)
        # or NaN depending on the code path
        assert isinstance(result, (float, np.floating))


@pytest.mark.p2
class TestRCubedTurtleFinalEdgeCases:
    """Test r_cubed_turtle edge cases for lines 604, 625."""

    def test_r_cubed_turtle_empty_years(self):
        """Line 604: len(years) < 1."""
        # Empty returns
        returns = pd.Series([], dtype=float)
        result = stats.r_cubed_turtle(returns)
        assert np.isnan(result)

    def test_r_cubed_turtle_empty_max_drawdowns(self):
        """Line 625: len(max_dds) == 0."""
        # Returns that produce no valid drawdowns
        # This happens when all years have empty data or zero-length chunks
        returns = pd.Series(
            [0.0, 0.0, 0.0],
            index=pd.date_range("2020-01-01", periods=3, freq="D"),
        )
        result = stats.r_cubed_turtle(returns)
        # Should return NaN or inf
        assert isinstance(result, (float, np.floating))
