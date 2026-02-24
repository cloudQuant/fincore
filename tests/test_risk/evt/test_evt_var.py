"""Tests for EVT-based VaR (Value at Risk) calculation.

Tests VaR calculation using GPD and GEV models.
Split from test_evt_full_coverage.py for maintainability.
"""

from __future__ import annotations

import numpy as np
import pytest

from fincore.risk.evt import evt_var


class TestEVTVar:
    """Test EVT-based VaR calculation."""

    def test_gpd_model_lower_tail(self, heavy_tailed_data):
        """Test GPD-based VaR for lower tail."""
        var = evt_var(heavy_tailed_data, alpha=0.05, model="gpd", tail="lower")

        assert isinstance(var, float)
        assert var < 0  # VaR should be negative for losses

    def test_gpd_model_upper_tail(self, heavy_tailed_data):
        """Test GPD-based VaR for upper tail."""
        var = evt_var(heavy_tailed_data, alpha=0.05, model="gpd", tail="upper")

        assert isinstance(var, float)

    def test_gev_model_lower_tail(self, heavy_tailed_data):
        """Test GEV-based VaR for lower tail."""
        var = evt_var(heavy_tailed_data, alpha=0.05, model="gev", tail="lower")

        assert isinstance(var, float)

    def test_custom_threshold(self, heavy_tailed_data):
        """Test VaR with custom threshold."""
        var = evt_var(heavy_tailed_data, alpha=0.05, model="gpd", threshold=0.05)

        assert isinstance(var, float)

    def test_unknown_model(self, heavy_tailed_data):
        """Test that unknown model raises ValueError."""
        with pytest.raises(ValueError, match="Unknown model"):
            evt_var(heavy_tailed_data, alpha=0.05, model="unknown")


class TestEVTVarEdgeCases:
    """Test EVT VaR edge cases for full coverage."""

    def test_evt_var_gpd_exponential_case(self):
        """Test EVT VaR with GPD when xi is near zero (exponential case)."""
        np.random.seed(42)
        # Light-tailed data for near-zero xi
        data = np.random.exponential(0.01, 5000)
        returns = -np.abs(data)
        var = evt_var(returns, alpha=0.05, model="gpd", tail="lower")
        assert isinstance(var, float)

    def test_evt_var_gev_gumbel_case(self):
        """Test EVT VaR with GEV when xi is near zero (Gumbel case)."""
        np.random.seed(42)
        # Gumbel-like distribution (light-tailed block maxima)
        data = np.random.gumbel(0, 0.01, 5000)
        var = evt_var(data, alpha=0.05, model="gev", tail="lower")
        assert isinstance(var, float)

    def test_evt_var_gpd_exponential_case_line_335(self):
        """Test EVT VaR GPD exponential case (specific line coverage)."""
        np.random.seed(42)
        # Exponential-like data produces xi ~ 0
        data = np.random.exponential(0.01, 5000)
        returns = -np.abs(data)
        var = evt_var(returns, alpha=0.05, model="gpd", tail="lower")
        assert isinstance(var, float)
        assert var < 0  # VaR should be negative

    def test_evt_var_gev_gumbel_case_line_354(self):
        """Test EVT VaR GEV Gumbel case (specific line coverage)."""
        np.random.seed(42)
        # Gumbel data produces xi ~ 0
        data = np.random.gumbel(0, 0.01, 5000)
        var = evt_var(data, alpha=0.05, model="gev", tail="lower")
        assert isinstance(var, float)

    def test_evt_var_gpd_near_zero_xi_branch(self):
        """Test EVT VaR GPD branch when xi is very close to zero."""
        from unittest.mock import patch

        np.random.seed(42)
        data = np.random.exponential(0.01, 5000)
        _returns = -np.abs(data)  # Not used, just for setup

        # Mock gpd_fit to return xi very close to 0
        mock_params = {"xi": 1e-12, "beta": 0.02, "threshold": 0.05, "n_exceed": 100}

        with patch("fincore.risk.evt.gpd_fit", return_value=mock_params):
            var = evt_var(data, alpha=0.05, model="gpd", tail="lower")
            assert isinstance(var, float)

    def test_evt_var_gev_near_zero_xi_branch(self):
        """Test EVT VaR GEV branch when xi is very close to zero."""
        from unittest.mock import patch

        np.random.seed(42)
        data = np.random.gumbel(0, 0.01, 5000)

        # Mock gev_fit to return xi very close to 0
        mock_params = {"xi": 1e-12, "mu": -0.05, "sigma": 0.02, "n_blocks": 50}

        with patch("fincore.risk.evt.gev_fit", return_value=mock_params):
            var = evt_var(data, alpha=0.05, model="gev", tail="lower")
            assert isinstance(var, float)
