"""Tests for comprehensive extreme risk function.

Tests the extreme_risk function that combines VaR and CVaR.
Split from test_evt_full_coverage.py for maintainability.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fincore.risk.evt import extreme_risk


class TestExtremeRisk:
    """Test comprehensive extreme risk function."""

    def test_gpd_model(self, heavy_tailed_data):
        """Test extreme_risk with GPD model."""
        returns = pd.Series(heavy_tailed_data)
        risk = extreme_risk(returns, alpha=0.05, model="gpd")

        assert isinstance(risk, pd.DataFrame)
        assert "VaR" in risk.columns
        assert "CVaR" in risk.columns
        assert "tail_index" in risk.columns
        assert "threshold" in risk.columns
        assert "n_exceedances" in risk.columns

    def test_gev_model(self, heavy_tailed_data):
        """Test extreme_risk with GEV model."""
        returns = pd.Series(heavy_tailed_data)
        risk = extreme_risk(returns, alpha=0.05, model="gev")

        assert isinstance(risk, pd.DataFrame)
        assert "VaR" in risk.columns
        assert "CVaR" in risk.columns
        assert "tail_index" in risk.columns
        assert "location" in risk.columns
        assert "scale" in risk.columns

    def test_unknown_model(self, heavy_tailed_data):
        """Test that unknown model raises ValueError."""
        returns = pd.Series(heavy_tailed_data)
        with pytest.raises(ValueError, match="Unknown model"):
            extreme_risk(returns, alpha=0.05, model="unknown")
