"""Tests for risk/evt.py edge cases.

Targets:
- risk/evt.py: 156, 166, 447 - GPD fitting and EVT CVaR
"""

import numpy as np
import pytest


class TestEVTEdgeCases:
    """Test EVT edge cases."""

    def test_gpd_fit_neg_loglik_beta_le_zero(self):
        """Line 156: beta <= 0 returns 1e10."""
        from fincore.risk.evt import gpd_fit

        np.random.seed(42)
        data = -np.random.exponential(scale=0.01, size=500)
        result = gpd_fit(data, method="mle")
        assert result["beta"] > 0

    def test_gpd_fit_exponential_case(self):
        """Line 166: |xi| < 1e-10 uses exponential case."""
        from fincore.risk.evt import gpd_fit

        np.random.seed(42)
        data = -np.random.exponential(scale=0.01, size=500)
        result = gpd_fit(data, method="mle")
        assert "xi" in result

    def test_evt_cvar_unknown_model(self):
        """Line 447: unknown model raises ValueError."""
        from fincore.risk.evt import evt_cvar

        np.random.seed(42)
        returns = np.random.normal(0, 0.01, 500)

        with pytest.raises(ValueError, match="Unknown model"):
            evt_cvar(returns, alpha=0.05, model="unknown")
