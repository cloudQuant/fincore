"""Tests for GARCH volatility models.

Tests GARCH, EGARCH, GJR-GARCH models and related functions.
Split from test_risk_models.py for maintainability.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


class TestGARCH:
    """Tests for GARCH model."""

    def test_garch_fit(self):
        """Test GARCH model fitting."""
        from fincore.risk.garch import GARCH

        np.random.seed(42)
        returns = pd.Series(np.random.randn(1000) * 0.02)

        model = GARCH(p=1, q=1)
        result = model.fit(returns)

        assert result.params is not None
        assert "omega" in result.params
        assert "alpha" in result.params
        assert "beta" in result.params
        assert 0 < result.params["alpha"] < 1
        assert 0 < result.params["beta"] < 1

    def test_garch_conditional_var(self):
        """Test conditional variance computation."""
        from fincore.risk.garch import GARCH

        returns = pd.Series(np.random.randn(1000) * 0.02)

        model = GARCH(p=1, q=1)
        result = model.fit(returns)

        cond_var = result.conditional_var

        assert len(cond_var) == len(returns)
        assert np.all(cond_var > 0)  # Variance must be positive

    def test_garch_forecast(self):
        """Test GARCH forecasting."""
        from fincore.risk.garch import GARCH

        returns = pd.Series(np.random.randn(1000) * 0.02)

        model = GARCH(p=1, q=1)
        result = model.fit(returns)

        forecasts = result.forecast(horizon=5)

        assert len(forecasts) == 5
        assert np.all(forecasts > 0)  # Variances must be positive

    def test_garch_insufficient_data_raises(self):
        """Test GARCH with insufficient data."""
        from fincore.risk.garch import GARCH

        short_returns = pd.Series(np.random.randn(5) * 0.02)

        model = GARCH(p=1, q=1)
        with pytest.raises(ValueError, match="Insufficient data"):
            model.fit(short_returns)


class TestEGARCH:
    """Tests for EGARCH model."""

    def test_egarch_fit(self):
        """Test EGARCH model fitting."""
        from fincore.risk.garch import EGARCH

        returns = pd.Series(np.random.randn(1000) * 0.02)

        model = EGARCH()
        result = model.fit(returns)

        assert result.params is not None
        assert "omega" in result.params
        assert "alpha" in result.params
        assert "gamma" in result.params  # Asymmetry parameter
        assert "beta" in result.params


class TestGJRGARCH:
    """Tests for GJR-GARCH model."""

    def test_gjr_garch_fit(self):
        """Test GJR-GARCH model fitting."""
        from fincore.risk.garch import GJRGARCH

        returns = pd.Series(np.random.randn(1000) * 0.02)

        model = GJRGARCH()
        result = model.fit(returns)

        assert result.params is not None
        assert "gamma" in result.params  # Leverage parameter
        assert result.params["gamma"] >= 0  # Leverage should be non-negative

    def test_gjr_garch_leverage_effect(self):
        """Test that GJR-GARCH captures leverage."""
        from fincore.risk.garch import GJRGARCH

        # Create asymmetric returns
        np.random.seed(42)
        pos_shocks = np.random.exponential(0.01, 500) * 0.3
        neg_shocks = -np.random.exponential(0.015, 500) * 0.7  # More volatility
        returns = pd.Series(np.concatenate([pos_shocks, neg_shocks]))

        gjr_model = GJRGARCH()
        gjr_result = gjr_model.fit(returns)

        # Gamma should capture asymmetry
        assert gjr_result.params["gamma"] >= 0


class TestForecastVolatility:
    """Tests for forecast_volatility convenience function."""

    def test_forecast_garch(self):
        """Test volatility forecasting with GARCH."""
        from fincore.risk.garch import forecast_volatility

        returns = pd.Series(np.random.randn(1000) * 0.02)

        forecasts, result = forecast_volatility(returns, model="GARCH", horizon=5)

        assert len(forecasts) == 5
        assert result is not None

    def test_forecast_egarch(self):
        """Test volatility forecasting with EGARCH."""
        from fincore.risk.garch import forecast_volatility

        returns = pd.Series(np.random.randn(1000) * 0.02)

        forecasts, result = forecast_volatility(returns, model="EGARCH", horizon=3)

        assert len(forecasts) == 3

    def test_forecast_invalid_model_raises(self):
        """Test invalid model name raises error."""
        from fincore.risk.garch import forecast_volatility

        returns = pd.Series(np.random.randn(1000) * 0.02)

        with pytest.raises(ValueError, match="Unknown model"):
            forecast_volatility(returns, model="INVALID")


class TestConditionalVar:
    """Tests for conditional VaR calculation."""

    def test_conditional_var_garch(self):
        """Test conditional VaR with GARCH."""
        from fincore.risk.garch import conditional_var

        returns = pd.Series(np.random.randn(1000) * 0.02)

        risk = conditional_var(returns, model="GARCH", alpha=0.05)

        assert "var" in risk
        assert "cond_var" in risk
        assert risk["var"] < 0  # VaR should be negative for 5% level

    def test_conditional_var_gjr(self):
        """Test conditional VaR with GJR-GARCH."""
        from fincore.risk.garch import conditional_var

        returns = pd.Series(np.random.randn(1000) * 0.02)

        risk = conditional_var(returns, model="GJRGARCH", alpha=0.01)

        assert "var" in risk
        # 1% VaR should be more extreme than 5%
        risk_5 = conditional_var(returns, model="GJRGARCH", alpha=0.05)
        assert risk["var"] < risk_5["var"]
