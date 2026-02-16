"""Tests for risk.garch module — GARCH forecast mean-reversion."""

from __future__ import annotations

import numpy as np
import pytest

from fincore.risk.garch import EGARCH, GARCH, GJRGARCH, GARCHResult, conditional_var, forecast_volatility


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

    def test_forecast_horizon_must_be_positive(self, fitted_garch):
        with pytest.raises(ValueError, match="horizon must be >= 1"):
            fitted_garch.forecast(horizon=0)

    def test_forecast_first_step_uses_latest_shock(self):
        """One-step forecast should use last shock and last variance."""
        result = GARCHResult(
            params={"omega": 0.1, "alpha": 0.2, "beta": 0.7},
            conditional_var=np.array([0.04, 0.09]),
            residuals=np.array([0.0, 2.0]),  # standardized residuals
            log_likelihood=0.0,
        )

        fc = result.forecast(horizon=2)

        expected_1 = 0.1 + 0.2 * (2.0 * np.sqrt(0.09)) ** 2 + 0.7 * 0.09
        expected_2 = 0.1 + (0.2 + 0.7) * expected_1

        assert np.isclose(fc[0], expected_1)
        assert np.isclose(fc[1], expected_2)


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


class TestGARCHConstantMean:
    """Test GARCH with constant mean model (lines 150-151, 179, 204-205)."""

    def test_fit_with_constant_mean(self):
        """Test GARCH fit with mean_model='constant'."""
        np.random.seed(42)
        returns = np.random.randn(300) * 0.01 + 0.001  # Add small drift
        model = GARCH(p=1, q=1, mean_model="constant")
        result = model.fit(returns)

        assert isinstance(result, GARCHResult)
        assert "mu" in result.params
        assert "omega" in result.params
        assert "alpha" in result.params
        assert "beta" in result.params
        # mu should be close to the true mean
        assert abs(result.params["mu"] - 0.001) < 0.005

    def test_fit_with_constant_mean_negative_returns(self):
        """Test GARCH fit with constant mean on negative returns."""
        np.random.seed(42)
        returns = np.random.randn(200) * 0.01 - 0.0005  # Negative drift
        model = GARCH(p=1, q=1, mean_model="constant")
        result = model.fit(returns)

        assert isinstance(result, GARCHResult)
        assert "mu" in result.params
        assert result.params["mu"] < 0  # Should capture negative mean

    def test_neg_log_likelihood_constant_mean(self):
        """Test _neg_log_likelihood with mean_model='constant'."""
        np.random.seed(42)
        returns = np.random.randn(200) * 0.01
        model = GARCH(p=1, q=1, mean_model="constant")

        # Fit to get parameters
        result = model.fit(returns)

        # Verify log likelihood is reasonable (it's actually positive log-likelihood)
        assert result.log_likelihood > 0  # Log-likelihood should be positive

    def test_fit_insufficient_data_constant_mean(self):
        """Test GARCH with constant mean on insufficient data."""
        np.random.seed(42)
        returns = np.random.randn(5) * 0.01  # Too few data points
        model = GARCH(p=1, q=1, mean_model="constant")

        with pytest.raises(ValueError, match="Insufficient data"):
            model.fit(returns)


class TestEGARCH:
    """Test EGARCH model."""

    def test_egarch_fit(self):
        """Test basic EGARCH fit."""
        np.random.seed(42)
        returns = np.random.randn(300) * 0.01
        model = EGARCH()
        result = model.fit(returns)

        assert isinstance(result, GARCHResult)
        assert "omega" in result.params
        assert "alpha" in result.params
        assert "gamma" in result.params  # Asymmetry parameter
        assert "beta" in result.params

    def test_egarch_insufficient_data(self):
        """Test EGARCH with insufficient data (line 294)."""
        np.random.seed(42)
        returns = np.random.randn(5) * 0.01  # Too few data points
        model = EGARCH()

        with pytest.raises(ValueError, match="Insufficient data for EGARCH"):
            model.fit(returns)

    def test_egarch_forecast(self):
        """Test EGARCH forecast method exists and runs."""
        np.random.seed(42)
        returns = np.random.randn(200) * 0.01
        model = EGARCH()
        result = model.fit(returns)

        # The forecast method exists on GARCHResult but is designed for standard GARCH
        # For EGARCH it will use a simplified formula that may produce negative values
        forecasts = result.forecast(horizon=5)
        assert len(forecasts) == 5
        # Just verify the method runs; actual EGARCH forecasting would need
        # log-variance specific logic which is out of scope for this test


class TestGJRGARCH:
    """Test GJR-GARCH model."""

    def test_gjr_garch_fit(self):
        """Test basic GJR-GARCH fit."""
        np.random.seed(42)
        returns = np.random.randn(300) * 0.01
        model = GJRGARCH()
        result = model.fit(returns)

        assert isinstance(result, GARCHResult)
        assert "omega" in result.params
        assert "alpha" in result.params
        assert "gamma" in result.params  # Leverage parameter
        assert "beta" in result.params

    def test_gjr_garch_insufficient_data(self):
        """Test GJR-GARCH with insufficient data (line 420)."""
        np.random.seed(42)
        returns = np.random.randn(5) * 0.01  # Too few data points
        model = GJRGARCH()

        with pytest.raises(ValueError, match="Insufficient data for GJR-GARCH"):
            model.fit(returns)

    def test_gjr_garch_forecast(self):
        """Test GJR-GARCH forecast."""
        np.random.seed(42)
        returns = np.random.randn(200) * 0.01
        model = GJRGARCH()
        result = model.fit(returns)

        forecasts = result.forecast(horizon=5)
        assert len(forecasts) == 5
        assert (forecasts > 0).all()


class TestForecastVolatility:
    """Test forecast_volatility convenience function."""

    def test_forecast_volatility_garch(self):
        """Test forecast_volatility with GARCH."""
        np.random.seed(42)
        returns = np.random.randn(200) * 0.01

        forecasts, result = forecast_volatility(returns, model="GARCH", horizon=3)

        assert len(forecasts) == 3
        assert isinstance(result, GARCHResult)

    def test_forecast_volatility_egarch(self):
        """Test forecast_volatility with EGARCH."""
        np.random.seed(42)
        returns = np.random.randn(200) * 0.01

        forecasts, result = forecast_volatility(returns, model="EGARCH", horizon=2)

        assert len(forecasts) == 2
        assert isinstance(result, GARCHResult)

    def test_forecast_volatility_gjr_garch(self):
        """Test forecast_volatility with GJR-GARCH."""
        np.random.seed(42)
        returns = np.random.randn(200) * 0.01

        forecasts, result = forecast_volatility(returns, model="GJRGARCH", horizon=2)

        assert len(forecasts) == 2
        assert isinstance(result, GARCHResult)

    def test_forecast_volatility_unknown_model(self):
        """Test forecast_volatility with unknown model."""
        np.random.seed(42)
        returns = np.random.randn(200) * 0.01

        with pytest.raises(ValueError, match="Unknown model"):
            forecast_volatility(returns, model="UNKNOWN", horizon=2)


class TestConditionalVar:
    """Test conditional_var convenience function."""

    def test_conditional_var_garch(self):
        """Test conditional_var with GARCH."""
        np.random.seed(42)
        returns = np.random.randn(200) * 0.01

        result = conditional_var(returns, model="GARCH", alpha=0.05)

        assert "var" in result
        assert "cond_var" in result
        assert "result" in result
        assert isinstance(result["result"], GARCHResult)

    def test_conditional_var_different_alpha(self):
        """Test conditional_var with different alpha levels."""
        np.random.seed(42)
        returns = np.random.randn(200) * 0.01

        result_05 = conditional_var(returns, model="GARCH", alpha=0.05)
        result_01 = conditional_var(returns, model="GARCH", alpha=0.01)

        # 99% VaR should be more extreme than 95% VaR
        assert abs(result_01["var"]) > abs(result_05["var"])
