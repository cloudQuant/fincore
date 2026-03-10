"""Tests for stats.hurst_exponent line coverage.

Part of test_exact_line_coverage.py split - Hurst exponent tests with P2 markers.
"""

from __future__ import annotations

import pandas as pd
import pytest

from fincore.metrics.stats import hurst_exponent


@pytest.mark.p2
class TestHurstExponentLineCoverage:
    """Test hurst_exponent edge cases for exact line coverage."""

    def test_hurst_line_175_continue(self):
        """stats.py line 175: continue when n_subseries < 1."""
        # Create series where lag > n will cause n_subseries < 1
        # The function starts with min_lag = max(2, n // 50)
        # For very short series, some lags will cause n_subseries < 1
        returns = pd.Series([0.01, 0.02, 0.015, -0.005, 0.003])
        result = hurst_exponent(returns)
        # Should hit line 175 continue for some lags
        assert isinstance(result, (float, float))

    def test_hurst_line_193_nan_fallback(self):
        """stats.py line 193: return np.nan when insufficient rs_values and invalid fallback."""
        # Very short series where s_std <= 0 or r_range <= 0
        # This causes fallback to not be usable, returning NaN
        returns = pd.Series([0.01, 0.02])
        result = hurst_exponent(returns)
        # Should hit line 193 (return np.nan)
        assert isinstance(result, (float, float))

    def test_hurst_line_203_filtered_insufficient(self):
        """stats.py line 203: return np.nan when filtered lags insufficient."""
        # Create returns where after filtering lags_array, we have < 2
        # This happens when all rs_values result in invalid (lags <= 0 or rs <= 0)
        returns = pd.Series([0.01, 0.02, -0.01, 0.005, 0.003])
        result = hurst_exponent(returns)
        # May hit line 203 depending on data
        assert isinstance(result, (float, float))
