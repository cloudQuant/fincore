"""Tests for risk.garch module — GARCH forecast mean-reversion."""

from __future__ import annotations

import numpy as np
import pytest

from fincore.risk.garch import GARCH, GARCHResult


class TestGARCHForecastMeanReversion:
    """Verify that GARCH forecast converges to long-run variance."""

    @pytest.fixture
    def fitted_garch(self):
        np.random.seed(42)
        returns = np.random.randn(500) * 0.01
        model = GARCH(p=1, q=1)
        result = model.fit(returns)
        return result

    def test_forecast_returns_array(self, fitted_garch):
        fc = fitted_garch.forecast(horizon=10)
        assert isinstance(fc, np.ndarray)
        assert len(fc) == 10

    def test_forecast_converges_to_long_run(self, fitted_garch):
        """Long horizon forecast should converge toward unconditional variance."""
        fc = fitted_garch.forecast(horizon=500)
        omega = fitted_garch.params["omega"]
        alpha = fitted_garch.params.get("alpha", 0.0)
        beta = fitted_garch.params.get("beta", 0.0)
        persistence = alpha + beta

        if 0 < persistence < 1:
            long_run_var = omega / (1 - persistence)
            # Last forecast element should be close to long-run variance
            assert abs(fc[-1] - long_run_var) / long_run_var < 0.01, (
                f"Forecast didn't converge: fc[-1]={fc[-1]:.6f}, long_run_var={long_run_var:.6f}"
            )

    def test_forecast_monotone_convergence(self, fitted_garch):
        """Forecast should monotonically approach long_run_var from either direction."""
        fc = fitted_garch.forecast(horizon=100)
        # Check that differences shrink (convergence)
        diffs = np.abs(np.diff(fc))
        # After burn-in, differences should generally decrease
        assert diffs[-1] < diffs[0] or np.isclose(diffs[-1], diffs[0], atol=1e-15)

    def test_forecast_all_positive(self, fitted_garch):
        fc = fitted_garch.forecast(horizon=50)
        assert (fc > 0).all(), "Variance forecast should be positive"


class TestGARCHFit:
    def test_fit_returns_result(self):
        np.random.seed(42)
        returns = np.random.randn(300) * 0.01
        model = GARCH(p=1, q=1)
        result = model.fit(returns)
        assert isinstance(result, GARCHResult)
        assert "omega" in result.params
        assert "alpha" in result.params
        assert "beta" in result.params

    def test_fit_short_data(self):
        np.random.seed(42)
        returns = np.random.randn(30) * 0.01
        model = GARCH(p=1, q=1)
        result = model.fit(returns)
        assert isinstance(result, GARCHResult)

    def test_persistence_less_than_one(self):
        np.random.seed(42)
        returns = np.random.randn(500) * 0.01
        model = GARCH(p=1, q=1)
        result = model.fit(returns)
        persistence = result.params.get("alpha", 0) + result.params.get("beta", 0)
        # Stationarity condition
        assert persistence < 1.0, f"Persistence {persistence} >= 1"
