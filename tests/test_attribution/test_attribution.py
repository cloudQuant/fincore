"""Tests for attribution module.

Moved from __init__.py so pytest can discover them properly.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from fincore.attribution.fama_french import (
    FamaFrenchModel,
    calculate_idiosyncratic_risk,
    fetch_ff_factors,
)
from fincore.attribution.style import (
    StyleResult,
    fetch_style_factors,
)


class TestFamaFrench:
    """Tests for Fama-French models."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        periods = 100

        # Generate correlated returns
        cov = np.array([[1, 0.5, 0.3], [0.5, 1, 0.2], [0.3, 0.2, 0.1]])
        cov = cov @ cov.T + np.eye(3) * 0.01

        returns = pd.DataFrame(np.random.multivariate_normal(np.zeros(3), cov, size=periods))

        # Generate factors
        rf = 0.02  # Risk-free rate
        mktrf = rf + np.random.normal(0, 0.05, periods)  # Market
        smb = np.random.normal(0, 0.02, periods)  # Size
        hml = np.random.normal(0, 0.02, periods)  # Value
        rmw = np.random.normal(0, 0.01, periods)  # Profitability
        cma = np.random.normal(0, 0.01, periods)  # Investment

        factor_data = pd.DataFrame(
            {
                "MKT": mktrf,
                "SMB": smb,
                "HML": hml,
                "RMW": rmw,
                "CMA": cma,
            },
            index=returns.index,
        )

        return returns, factor_data

    def test_fama_french_3factor(self, sample_data):
        """Test 3-factor model estimation."""
        returns, factor_data = sample_data

        model = FamaFrenchModel(model_type="3factor", risk_free_rate=0.02)
        result = model.fit(returns, factor_data)

        assert "alpha" in result
        assert "betas" in result
        assert "r_squared" in result
        assert 0 <= result["r_squared"] <= 1

        # Check alpha is reasonable
        assert -0.5 < result["alpha"] < 0.5

    def test_fama_french_5factor(self, sample_data):
        """Test 5-factor model estimation."""
        returns, factor_data = sample_data

        model = FamaFrenchModel(model_type="5factor", risk_free_rate=0.02)
        result = model.fit(returns, factor_data)

        assert "alpha" in result
        assert "betas" in result
        assert len(result["betas"]) == 5  # MKT, SMB, HML, RMW, CMA

    def test_model_predict(self, sample_data):
        """Test prediction method."""
        returns, factor_data = sample_data

        model = FamaFrenchModel()
        model.fit(returns, factor_data)

        # Predict returns
        predicted = model.predict(factor_data)

        assert predicted.shape[0] == returns.shape[0]

    def test_get_factor_exposures(self, sample_data):
        """Test factor exposure calculation."""
        returns, factor_data = sample_data

        model = FamaFrenchModel()
        model.fit(returns, factor_data)

        # Get exposures
        exposures = model.get_factor_exposures(returns, factor_data)

        assert exposures.shape[0] == returns.shape[0]
        assert exposures.shape[1] == 6  # alpha + 5 factors

    def test_idiosyncratic_risk(self, sample_data):
        """Test idiosyncratic risk calculation."""
        returns = sample_data[0].iloc[:, :3]  # Use first 3 assets

        factor_data = sample_data[1]
        model = FamaFrenchModel()
        model.fit(returns, factor_data)

        idio_risk = calculate_idiosyncratic_risk(returns, factor_data)

        assert len(idio_risk) == 3
        # Idiosyncratic risk should be positive
        assert np.all(idio_risk > 0)

    def test_fit_constant_returns_no_runtime_warning(self):
        """Constant returns should not trigger runtime warnings in Newey-West path."""
        periods = 60
        idx = pd.date_range("2020-01-01", periods=periods)
        returns = pd.Series(np.zeros(periods), index=idx)
        factor_data = pd.DataFrame(
            {
                "MKT": np.random.normal(0, 0.01, periods),
                "SMB": np.random.normal(0, 0.01, periods),
                "HML": np.random.normal(0, 0.01, periods),
                "RMW": np.random.normal(0, 0.01, periods),
                "CMA": np.random.normal(0, 0.01, periods),
            },
            index=idx,
        )

        model = FamaFrenchModel(model_type="5factor")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = model.fit(returns, factor_data, newey_west_lags=2)

        assert "alpha" in result
        runtime_warnings = [w for w in caught if issubclass(w.category, RuntimeWarning)]
        assert len(runtime_warnings) == 0


class TestFetchPlaceholders:
    """Tests that placeholder data fetchers raise NotImplementedError."""

    def test_fetch_ff_factors_raises(self):
        """fetch_ff_factors must raise NotImplementedError."""
        with pytest.raises(NotImplementedError, match="No Fama-French data provider"):
            fetch_ff_factors("2020-01-01", "2020-12-31")

    def test_fetch_style_factors_raises(self):
        """fetch_style_factors must raise NotImplementedError."""
        with pytest.raises(NotImplementedError, match="No style factor data provider"):
            fetch_style_factors(["AAPL", "GOOG"])
