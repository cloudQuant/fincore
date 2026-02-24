"""Tests for metrics/stats.py edge cases.

Targets:
- metrics/stats.py: 175, 193, 203, 604, 625 - hurst_exponent, r_cubed_turtle
"""

import numpy as np
import pandas as pd


class TestStatsEdgeCases:
    """Test stats.py edge cases."""

    def test_hurst_exponent_n_subseries_less_than_1(self):
        """Line 175: n_subseries < 1."""
        from fincore.metrics.stats import hurst_exponent

        returns = pd.Series([0.01])
        result = hurst_exponent(returns)
        assert isinstance(result, (float, np.floating))

    def test_hurst_exponent_insufficient_rs_uses_fallback(self):
        """Line 193: len(rs_values) < 2 with valid s_std and r_range."""
        from fincore.metrics.stats import hurst_exponent

        np.random.seed(42)
        returns = pd.Series([0.01, 0.02, -0.01, 0.005, 0.003])
        result = hurst_exponent(returns)
        assert isinstance(result, (float, np.floating))

    def test_hurst_exponent_constant_returns(self):
        """Line 203: constant returns -> insufficient lags."""
        from fincore.metrics.stats import hurst_exponent

        returns = pd.Series([0.01] * 10)
        result = hurst_exponent(returns)
        # May return fallback value or NaN
        assert isinstance(result, (float, np.floating))

    def test_r_cubed_turtle_empty_years(self):
        """Line 604: len(years) < 1."""
        from fincore.metrics.stats import r_cubed_turtle

        returns = pd.Series([], dtype=float)
        result = r_cubed_turtle(returns)
        assert np.isnan(result)

    def test_r_cubed_turtle_empty_max_drawdowns(self):
        """Line 625: len(max_dds) == 0."""
        from fincore.metrics.stats import r_cubed_turtle

        returns = pd.Series(
            [0.0, 0.0, 0.0],
            index=pd.date_range("2020-01-01", periods=3),
        )
        result = r_cubed_turtle(returns)
        assert isinstance(result, (float, np.floating))
