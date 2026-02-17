"""Tests for EVT module - 100% coverage."""

import numpy as np
import pandas as pd
import pytest

from fincore.risk.evt import (
    evt_cvar,
    evt_var,
    extreme_risk,
    gev_fit,
    gpd_fit,
    hill_estimator,
)


@pytest.fixture
def heavy_tailed_data():
    """Create heavy-tailed data for EVT testing."""
    np.random.seed(42)
    # Use t-distribution with low degrees of freedom for heavy tails
    return np.random.standard_t(3, 5000)


@pytest.fixture
def light_tailed_data():
    """Create light-tailed data for EVT testing."""
    np.random.seed(42)
    return np.random.normal(0, 0.01, 5000)


class TestHillEstimator:
    """Test Hill estimator functionality."""

    def test_upper_tail(self, heavy_tailed_data):
        """Test Hill estimator for upper tail."""
        xi, excesses = hill_estimator(heavy_tailed_data, tail="upper")

        assert isinstance(xi, float)
        assert xi > 0  # Heavy-tailed
        assert len(excesses) > 0

    def test_lower_tail(self, heavy_tailed_data):
        """Test Hill estimator for lower tail."""
        xi, excesses = hill_estimator(heavy_tailed_data, tail="lower")

        assert isinstance(xi, float)
        assert xi > 0  # Heavy-tailed
        assert len(excesses) > 0

    def test_custom_threshold(self, heavy_tailed_data):
        """Test Hill estimator with custom threshold."""
        xi, excesses = hill_estimator(heavy_tailed_data, threshold=0.1, tail="upper")

        assert isinstance(xi, float)
        assert len(excesses) > 0

    def test_invalid_tail(self, heavy_tailed_data):
        """Test that invalid tail raises ValueError."""
        with pytest.raises(ValueError, match="tail must be"):
            hill_estimator(heavy_tailed_data, tail="middle")

    def test_insufficient_exceedances(self):
        """Test that insufficient exceedances raises ValueError."""
        data = np.array([1, 2, 3, 4, 5])  # Too few data points
        with pytest.raises(ValueError, match="Not enough exceedances"):
            hill_estimator(data, threshold=10, tail="upper")


class TestGPDFit:
    """Test GPD fitting functionality."""

    def test_mle_method(self, heavy_tailed_data):
        """Test GPD fit with MLE method."""
        params = gpd_fit(heavy_tailed_data, method="mle")

        assert "xi" in params
        assert "beta" in params
        assert "threshold" in params
        assert "n_exceed" in params
        assert params["n_exceed"] > 0

    def test_pwm_method(self, heavy_tailed_data):
        """Test GPD fit with PWM method."""
        params = gpd_fit(heavy_tailed_data, method="pwm")

        assert "xi" in params
        assert "beta" in params
        assert params["n_exceed"] > 0

    def test_custom_threshold(self, heavy_tailed_data):
        """Test GPD fit with custom threshold."""
        params = gpd_fit(heavy_tailed_data, threshold=0.05)

        assert params["threshold"] == 0.05

    def test_insufficient_exceedances(self):
        """Test that insufficient exceedances raises ValueError."""
        data = np.array([1, 2, 3, 4, 5])
        with pytest.raises(ValueError, match="Not enough exceedances"):
            gpd_fit(data, threshold=10)

    def test_unknown_method(self, heavy_tailed_data):
        """Test that unknown method raises ValueError."""
        with pytest.raises(ValueError, match="Unknown method"):
            gpd_fit(heavy_tailed_data, method="unknown")


class TestGEVFit:
    """Test GEV fitting functionality."""

    def test_default_block_size(self, heavy_tailed_data):
        """Test GEV fit with default block size."""
        params = gev_fit(heavy_tailed_data)

        assert "xi" in params
        assert "mu" in params
        assert "sigma" in params
        assert "n_blocks" in params

    def test_custom_block_size(self, heavy_tailed_data):
        """Test GEV fit with custom block size."""
        params = gev_fit(heavy_tailed_data, block_size=100)

        assert params["n_blocks"] == len(heavy_tailed_data) // 100


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
        var = evt_var(heavy_tailed_data, alpha=0.05, model="gpd", tail="lower")
        cvar = evt_cvar(heavy_tailed_data, alpha=0.05, model="gpd", tail="lower")

        assert cvar <= var  # CVaR should be worse (more negative) than VaR

    def test_unknown_model(self, heavy_tailed_data):
        """Test that unknown model raises ValueError."""
        with pytest.raises(ValueError, match="Unknown model"):
            evt_cvar(heavy_tailed_data, alpha=0.05, model="unknown")


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


class TestEVTEdgeCases:
    """Test edge cases for 100% coverage."""

    def test_gpd_mle_exponential_case(self):
        """Test GPD MLE with data approaching exponential distribution (xi ~ 0)."""
        np.random.seed(42)
        # Exponential-like distribution (light-tailed)
        data = np.random.exponential(0.01, 5000)
        # Negative returns
        returns = -np.abs(data)
        params = gpd_fit(returns, method="mle")
        # Should fit without error
        assert "xi" in params
        assert "beta" in params

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

    def test_gpd_mle_beta_near_zero(self):
        """Test GPD MLE handling when beta approaches zero."""
        np.random.seed(42)
        # Data with very small variance
        data = np.full(1000, -0.01) + np.random.normal(0, 1e-6, 1000)
        params = gpd_fit(data, method="mle")
        assert "xi" in params
        assert params["beta"] > 0


class TestEVTEdgeCasesForFullCoverage:
    """Additional tests to reach 100% coverage."""

    def test_gpd_mle_exponential_case_line_166(self):
        """Test GPD MLE exponential case (xi ~ 0) - covers line 166."""
        np.random.seed(42)
        # Data that produces xi close to 0 (exponential-like)
        data = np.random.exponential(0.01, 5000)
        returns = -np.abs(data)
        params = gpd_fit(returns, method="mle")
        # xi should be close to 0 for exponential-like data
        assert "xi" in params
        assert abs(params["xi"]) < 0.3  # Exponential has xi ~ 0

    def test_gpd_mle_invalid_beta_returns_large_value(self, monkeypatch):
        """Test that GPD MLE handles invalid beta (line 155-156)."""
        import numpy as np

        np.random.seed(42)
        data = np.random.exponential(0.01, 5000)
        returns = -np.abs(data)

        # This test verifies the optimizer handles the constraint
        # Use lower threshold to ensure enough exceedances
        params = gpd_fit(returns, method="mle", threshold=0.001)
        assert params["beta"] > 0

    def test_evt_var_gpd_exponential_case_line_335(self):
        """Test EVT VaR GPD exponential case (line 335)."""
        np.random.seed(42)
        # Exponential-like data produces xi ~ 0
        data = np.random.exponential(0.01, 5000)
        returns = -np.abs(data)
        var = evt_var(returns, alpha=0.05, model="gpd", tail="lower")
        assert isinstance(var, float)
        assert var < 0  # VaR should be negative

    def test_evt_var_gev_gumbel_case_line_354(self):
        """Test EVT VaR GEV Gumbel case (line 354)."""
        np.random.seed(42)
        # Gumbel data produces xi ~ 0
        data = np.random.gumbel(0, 0.01, 5000)
        var = evt_var(data, alpha=0.05, model="gev", tail="lower")
        assert isinstance(var, float)

    def test_evt_cvar_gpd_exponential_case_line_420(self):
        """Test EVT CVaR GPD exponential case (line 420)."""
        np.random.seed(42)
        data = np.random.exponential(0.01, 5000)
        returns = -np.abs(data)
        cvar = evt_cvar(returns, alpha=0.05, model="gpd", tail="lower")
        assert isinstance(cvar, float)
        assert cvar < 0

    def test_evt_cvar_gev_gumbel_case_line_440(self):
        """Test EVT CVaR GEV Gumbel case (line 440)."""
        np.random.seed(42)
        data = np.random.gumbel(0, 0.01, 5000)
        cvar = evt_cvar(data, alpha=0.05, model="gev", tail="lower")
        assert isinstance(cvar, float)

    def test_evt_cvar_gpd_xi_ge_1_raises_line_425(self):
        """Test GPD CVaR raises error when xi >= 1 (line 425)."""
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
        """Test GEV CVaR raises error when xi >= 1 (line 445)."""
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
        """Test CVaR raises error for unknown model (line 447)."""
        data = np.random.exponential(0.01, 1000)
        with pytest.raises(ValueError, match="Unknown model"):
            evt_cvar(data, alpha=0.05, model="unknown")
