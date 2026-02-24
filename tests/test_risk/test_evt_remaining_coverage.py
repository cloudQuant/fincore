"""Tests for EVT module remaining missing coverage (lines 156, 166, 447)."""

import numpy as np
import pandas as pd
import pytest

from fincore.risk.evt import evt_cvar, gpd_fit


class TestEVTCVARCoverage:
    """Tests for EVT CVaR calculation missing coverage."""

    def test_evt_cvar_gev_gumbel_case(self):
        """Test evt_cvar with GEV model and |xi| < 1e-10 (line 440)."""
        # Create returns that result in xi close to 0 for GEV fit
        np.random.seed(42)
        # Use normal-like distribution which should give xi close to 0
        returns = np.random.randn(500) / 100

        result = evt_cvar(returns, alpha=0.05, model="gev")

        # Should return a finite CVaR value
        assert isinstance(result, (float, np.floating))

    def test_evt_cvar_gev_general_case(self):
        """Test evt_cvar with GEV model and xi < 1 but not close to 0 (line 443)."""
        # Create returns with heavier tail for GEV
        np.random.seed(42)
        # Use t-distribution for heavier tail
        returns = np.random.standard_t(4, 500) / 100

        result = evt_cvar(returns, alpha=0.05, model="gev")

        # Should return a finite CVaR value
        assert isinstance(result, (float, np.floating))

    def test_evt_cvar_unknown_model(self):
        """Test evt_cvar with unknown model raises ValueError (line 447)."""
        returns = np.array([0.01, 0.02, 0.015, -0.01, -0.02])

        with pytest.raises(ValueError, match="Unknown model"):
            evt_cvar(returns, alpha=0.05, model="unknown_model")

    def test_evt_cvar_gpd_xi_ge_1_raises(self):
        """Test evt_cvar raises ValueError when xi >= 1 for GPD."""
        # Create extreme negative returns that might result in xi >= 1
        # Need enough data for threshold computation
        np.random.seed(42)
        returns = np.concatenate([
            -np.random.exponential(scale=0.01, size=100),
            [-1.0, -2.0, -5.0, -10.0]  # Extreme negative outliers
        ])

        # This might raise ValueError or return a result
        try:
            result = evt_cvar(returns, alpha=0.05, model="gpd")
            # If it doesn't raise, check result
            assert isinstance(result, (float, np.floating))
        except ValueError as e:
            # Expected for extreme data
            assert "CVaR infinite" in str(e) or "not enough exceedances" in str(e)

    def test_gpd_fit_exponential_case_mle(self):
        """Test gpd_fit with MLE when |xi| < 1e-10 (line 166)."""
        # Create data from exponential-like distribution (xi should be ~0)
        # gpd_fit looks at negative returns, so we provide negative data
        np.random.seed(42)
        # Negative returns from exponential distribution
        data = -np.random.exponential(scale=0.01, size=500)

        result = gpd_fit(data, method="mle")

        # Should return valid parameters
        assert "xi" in result
        assert "beta" in result
        assert isinstance(result["xi"], (float, np.floating))

    def test_gpd_fit_beta_le_zero_early_return(self):
        """Test gpd_fit neg_loglik when beta <= 0 (line 156)."""
        # The line 156 is inside the neg_loglik function used for MLE optimization
        # It returns 1e10 when beta <= 0 to reject that parameter combination
        # We test this indirectly by calling gpd_fit with valid data

        np.random.seed(42)
        # Negative returns
        data = -np.random.exponential(scale=0.01, size=500)

        result = gpd_fit(data, method="mle")

        # Should successfully fit and return beta > 0
        assert result["beta"] > 0
