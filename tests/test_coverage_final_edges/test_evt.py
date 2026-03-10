"""Final edge case tests for EVT coverage.

Part of test_final_coverage_edges.py split - EVT tests with P2 markers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fincore.risk.evt import evt_cvar, gpd_fit


@pytest.mark.p2
class TestEVTFinalEdgeCases:
    """Test EVT edge cases for lines 156, 166, 447."""

    def test_gpd_fit_beta_le_zero_in_neg_loglik(self):
        """Line 156: beta <= 0 in neg_loglik returns 1e10."""
        # Negative returns from exponential distribution
        np.random.seed(42)
        data = -np.random.exponential(scale=0.01, size=500)
        result = gpd_fit(data, method="mle")
        # Should successfully fit with beta > 0
        assert result["beta"] > 0

    def test_gpd_fit_exponential_case_mle(self):
        """Line 166: |xi| < 1e-10 uses exponential case."""
        # Exponential-distributed losses (xi ≈ 0)
        np.random.seed(42)
        data = -np.random.exponential(scale=0.01, size=500)
        result = gpd_fit(data, method="mle")
        # Should return valid parameters
        assert "xi" in result
        assert "beta" in result

    def test_evt_cvar_unknown_model(self):
        """Line 447: unknown model raises ValueError."""
        np.random.seed(42)
        returns = np.random.normal(0, 0.01, 500)
        with pytest.raises(ValueError, match="Unknown model"):
            evt_cvar(returns, alpha=0.05, model="unknown_model")
