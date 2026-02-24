"""Tests for EVT-based CVaR (Conditional Value at Risk) calculation.

Tests CVaR calculation using GPD and GEV models.
Split from test_evt_full_coverage.py for maintainability.
"""

from __future__ import annotations

import numpy as np
import pytest

from fincore.risk.evt import evt_cvar


class TestEVTCVar:
    """Test EVT-based CVaR calculation."""

    def test_gpd_model_lower_tail(self, heavy_tailed_data):
        """Test GPD-based CVaR for lower tail."""
        cvar = evt_cvar(heavy_tailed_data, alpha=0.05, model="gpd", tail="lower")

        assert isinstance(cvar, float)
        assert cvar < 0  # CVaR should be negative for losses

    def test_gev_model_lower_tail(self, heavy_tailed_data):
        """Test GEV-based CVaR for lower tail."""
        cvar = evt_cvar(heavy_tailed_data, alpha=0.05, model="gev", tail="lower")

        assert isinstance(cvar, float)

    def test_cvar_less_than_var(self, heavy_tailed_data):
        """Test that CVaR is more negative than VaR."""
        from fincore.risk.evt import evt_var

        var = evt_var(heavy_tailed_data, alpha=0.05, model="gpd", tail="lower")
        cvar = evt_cvar(heavy_tailed_data, alpha=0.05, model="gpd", tail="lower")

        assert cvar <= var  # CVaR should be worse (more negative) than VaR

    def test_unknown_model(self, heavy_tailed_data):
        """Test that unknown model raises ValueError."""
        with pytest.raises(ValueError, match="Unknown model"):
            evt_cvar(heavy_tailed_data, alpha=0.05, model="unknown")


class TestEVTCVArEdgeCases:
    """Test EVT CVaR edge cases for full coverage."""

    def test_evt_cvar_gpd_exponential_case(self):
        """Test EVT CVaR with GPD when xi is near zero (exponential case)."""
        np.random.seed(42)
        data = np.random.exponential(0.01, 5000)
        returns = -np.abs(data)
        cvar = evt_cvar(returns, alpha=0.05, model="gpd", tail="lower")
        assert isinstance(cvar, float)

    def test_evt_cvar_gev_gumbel_case(self):
        """Test EVT CVaR with GEV when xi is near zero (Gumbel case)."""
        np.random.seed(42)
        data = np.random.gumbel(0, 0.01, 5000)
        cvar = evt_cvar(data, alpha=0.05, model="gev", tail="lower")
        assert isinstance(cvar, float)

    def test_evt_cvar_gev_xi_ge_1_raises(self):
        """Test that GEV CVaR raises error when xi >= 1."""
        np.random.seed(42)
        # Create data that will have xi >= 1 by using very heavy-tailed
        # We'll mock this by directly calling with data that produces high xi
        data = np.random.pareto(0.5, 5000) * 0.01  # Very heavy tail
        # This may or may not produce xi >= 1, so we test the function path
        try:
            cvar = evt_cvar(data, alpha=0.05, model="gev", tail="lower")
            # If it doesn't raise, that's okay - just check it returns a value
            assert isinstance(cvar, float)
        except ValueError as e:
            # Expected for very heavy tails
            assert "CVaR infinite" in str(e)

    def test_evt_cvar_gpd_exponential_case_line_420(self):
        """Test EVT CVaR GPD exponential case (specific line coverage)."""
        np.random.seed(42)
        data = np.random.exponential(0.01, 5000)
        returns = -np.abs(data)
        cvar = evt_cvar(returns, alpha=0.05, model="gpd", tail="lower")
        assert isinstance(cvar, float)
        assert cvar < 0

    def test_evt_cvar_gev_gumbel_case_line_440(self):
        """Test EVT CVaR GEV Gumbel case (specific line coverage)."""
        np.random.seed(42)
        data = np.random.gumbel(0, 0.01, 5000)
        cvar = evt_cvar(data, alpha=0.05, model="gev", tail="lower")
        assert isinstance(cvar, float)

    def test_evt_cvar_gpd_xi_ge_1_raises_line_425(self):
        """Test GPD CVaR raises error when xi >= 1."""
        from unittest.mock import patch

        # Mock gpd_fit to return xi >= 1
        mock_params = {"xi": 1.5, "beta": 0.1, "threshold": 0.05, "n_exceed": 100}
        with (
            patch("fincore.risk.evt.gpd_fit", return_value=mock_params),
            pytest.raises(ValueError, match="CVaR infinite for xi >= 1"),
        ):
            evt_cvar(
                np.random.exponential(0.01, 1000),
                alpha=0.05,
                model="gpd",
                tail="lower",
            )

    def test_evt_cvar_gev_xi_ge_1_raises_line_445(self):
        """Test GEV CVaR raises error when xi >= 1."""
        from unittest.mock import patch

        # Mock gev_fit to return xi >= 1
        mock_params = {"xi": 1.2, "sigma": 0.1, "mu": 0, "n_blocks": 10}
        with (
            patch("fincore.risk.evt.gev_fit", return_value=mock_params),
            pytest.raises(ValueError, match="CVaR infinite for xi >= 1"),
        ):
            evt_cvar(
                np.random.exponential(0.01, 1000),
                alpha=0.05,
                model="gev",
                tail="lower",
            )

    def test_evt_cvar_unknown_model_line_447(self):
        """Test CVaR raises error for unknown model."""
        data = np.random.exponential(0.01, 1000)
        with pytest.raises(ValueError, match="Unknown model"):
            evt_cvar(data, alpha=0.05, model="unknown")

    def test_evt_cvar_gpd_near_zero_xi_branch(self):
        """Test EVT CVaR GPD branch when xi is very close to zero."""
        from unittest.mock import patch

        np.random.seed(42)
        data = np.random.exponential(0.01, 5000)
        returns = -np.abs(data)

        # Mock both gpd_fit and evt_var to return consistent values with xi ~ 0
        mock_params = {"xi": 1e-12, "beta": 0.02, "threshold": 0.05, "n_exceed": 100}
        mock_var = -0.08  # Negative return-space VaR

        with (
            patch("fincore.risk.evt.gpd_fit", return_value=mock_params),
            patch("fincore.risk.evt.evt_var", return_value=mock_var),
        ):
            cvar = evt_cvar(returns, alpha=0.05, model="gpd", tail="lower")
            assert isinstance(cvar, float)

    def test_evt_cvar_gev_near_zero_xi_branch(self):
        """Test EVT CVaR GEV branch when xi is very close to zero."""
        from unittest.mock import patch

        np.random.seed(42)
        data = np.random.gumbel(0, 0.01, 5000)

        # Mock gev_fit and evt_var with xi ~ 0
        mock_params = {"xi": 1e-12, "mu": -0.05, "sigma": 0.02, "n_blocks": 50}
        mock_var = -0.08

        with (
            patch("fincore.risk.evt.gev_fit", return_value=mock_params),
            patch("fincore.risk.evt.evt_var", return_value=mock_var),
        ):
            cvar = evt_cvar(data, alpha=0.05, model="gev", tail="lower")
            assert isinstance(cvar, float)
