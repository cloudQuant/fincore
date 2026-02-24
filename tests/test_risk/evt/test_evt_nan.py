"""Tests for EVT functions with NaN data.

Tests handling of NaN values in EVT calculations.
Split from test_evt_full_coverage.py for maintainability.
"""

from __future__ import annotations

import numpy as np
import pytest

from fincore.risk.evt import (
    evt_cvar,
    evt_var,
    gev_fit,
    gpd_fit,
    hill_estimator,
)


class TestEVTWithNanData:
    """Test EVT functions with NaN data."""

    def test_hill_estimator_with_nans(self):
        """Test Hill estimator handles NaN values."""
        np.random.seed(42)
        data = np.concatenate([np.random.standard_t(3, 1000), [np.nan, np.nan]])
        xi, excesses = hill_estimator(data)

        assert isinstance(xi, float)

    def test_gpd_fit_with_nans(self):
        """Test GPD fit handles NaN values."""
        data = np.concatenate([np.random.standard_t(3, 1000), [np.nan, np.nan]])
        params = gpd_fit(data)

        assert "xi" in params
        assert "beta" in params

    def test_gev_fit_with_nans(self):
        """Test GEV fit handles NaN values."""
        data = np.concatenate([np.random.standard_t(3, 1000), [np.nan, np.nan]])
        params = gev_fit(data)

        assert "xi" in params
        assert "mu" in params
        assert "sigma" in params

    def test_evt_var_with_nans(self):
        """Test EVT VaR handles NaN values."""
        data = np.concatenate([np.random.standard_t(3, 1000), [np.nan, np.nan]])
        var = evt_var(data, alpha=0.05, model="gpd")

        assert isinstance(var, float)

    def test_evt_cvar_with_nans(self):
        """Test EVT CVaR handles NaN values."""
        data = np.concatenate([np.random.standard_t(3, 1000), [np.nan, np.nan]])
        cvar = evt_cvar(data, alpha=0.05, model="gpd")

        assert isinstance(cvar, float)
