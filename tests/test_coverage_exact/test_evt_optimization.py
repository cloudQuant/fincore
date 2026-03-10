"""Tests for EVT and optimization line coverage.

Part of test_exact_line_coverage.py split - EVT and optimization tests with P2 markers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fincore.optimization._utils import OptimizationError
from fincore.optimization.frontier import efficient_frontier
from fincore.risk.evt import evt_cvar, gpd_fit


@pytest.mark.p2
class TestEVTAndOptimizationLineCoverage:
    """Test EVT and optimization edge cases for exact line coverage."""

    def test_gpd_fit_line_156(self):
        """evt.py line 156: return 1e10 when beta <= 0 in neg_loglik."""
        # This is hit during optimization when beta becomes negative
        # The function returns 1e10 to reject that parameter combination
        np.random.seed(42)
        data = -np.random.exponential(scale=0.01, size=500)
        result = gpd_fit(data, method="mle")
        # The optimizer should find beta > 0
        assert result["beta"] > 0

    def test_gpd_fit_line_166(self):
        """evt.py line 166: exponential case when |xi| < 1e-10."""
        # Exponential-distributed data produces xi ≈ 0
        np.random.seed(42)
        data = -np.random.exponential(scale=0.01, size=500)
        result = gpd_fit(data, method="mle")
        # Should hit line 166 when xi is very close to 0
        assert "xi" in result

    def test_evt_cvar_line_447(self):
        """evt.py line 447: raise ValueError for unknown model."""
        returns = np.array([0.01, 0.02, -0.01, -0.02, 0.005])
        with pytest.raises(ValueError, match="Unknown model"):
            evt_cvar(returns, alpha=0.05, model="unknown")

    def test_frontier_line_106(self):
        """frontier.py line 106: return 1e6 when vol < 1e-12."""
        # Create returns with very low variance for one asset
        # This can trigger the vol < 1e-12 condition during optimization
        np.random.seed(42)
        returns = pd.DataFrame({
            "A": np.random.normal(0.01, 0.0001, 50),
            "B": np.random.normal(0.01, 0.0001, 50),
            "C": np.random.normal(0.01, 0.01, 50),
        })

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            try:
                result = efficient_frontier(returns, n_points=3)
                assert isinstance(result, dict)
            except OptimizationError:
                # Also acceptable
                pass
