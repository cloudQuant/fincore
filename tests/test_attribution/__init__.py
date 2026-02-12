"""Tests for attribution module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fincore.attribution.brinson import (
    BrinsonAttribution,
    brinson_attribution,
    brinson_cumulative,
    brinson_results,
)
from fincore.attribution.fama_french import (
    FamaFrenchModel,
    calculate_idiosyncratic_risk,
    fetch_ff_factors,
)
from fincore.attribution.style import (
    StyleResult,
    analyze_performance_by_style,
    calculate_regression_attribution,
    calculate_style_tilts,
)


class TestBrinsonAttribution:
    """Tests for Brinson attribution."""

    @pytest.fixture
    def sample_data(self):
        """Create sample returns for testing."""
        np.random.seed(42)
        periods = 4
        n_assets = 3

        # Create correlated returns
        cov = np.array([[1, 0.5, 0.3], [0.5, 1, 0.2], [0.3, 0.2, 0.1]])
        cov = cov @ cov.T + np.eye(3) * 0.01  # Add idioyncratic risk

        returns = pd.DataFrame(np.random.multivariate_normal(np.zeros(3), cov, size=periods*n_assets))

        return returns

    def test_brinson_attribution_basic(self, sample_data):
        """Test basic Brinson attribution."""
        returns = sample_data

        # Create weights (3 sectors x 3 assets = 9 entries)
        portfolio_weights = pd.DataFrame({
            "Sector_A_Stock_1": [0.4, 0.3, 0.2],
            "Sector_A_Stock_2": [0.3, 0.3, 0.3],
            "Sector_B_Stock_1": [0.2, 0.2, 0.2],
            "Sector_B_Stock_2": [0.2, 0.2, 0.2],
            "Sector_B_Stock_3": [0.1, 0.1, 0.1],
        }, index=returns.index)

        benchmark_weights = pd.DataFrame({
            "Sector_A_Stock_1": [0.5, 0.3, 0.2],
            "Sector_A_Stock_2": [0.3, 0.3, 0.3],
            "Sector_B_Stock_1": [0.3, 0.2, 0.2],
            "Sector_B_Stock_2": [0.2, 0.2, 0.2],
            "Sector_B_Stock_3": [0.1, 0.1, 0.1],
        }, index=returns.index)

        result = brinson_attribution(
            returns.values.flatten(),
            returns.values.flatten(),
            portfolio_weights.values,
            benchmark_weights.values,
        )

        assert "allocation" in result
        assert "selection" in result
        assert "interaction" in result
        assert result["total"] > 0  # Should have positive active return
        assert np.isclose(result["allocation"] + result["selection"] + result["interaction"], result["total"])

    def test_brinson_results_multi_period(self, sample_data):
        """Test Brinson attribution over multiple periods."""
        returns = sample_data

        portfolio_weights = pd.DataFrame({
            "Sector_A": [0.5, 0.3, 0.2] * 4,
            "Sector_B": [0.3, 0.3, 0.2] * 4,
        }, index=returns.index)

        benchmark_weights = pd.DataFrame({
            "Sector_A": [0.4, 0.3, 0.2] * 4,
            "Sector_B": [0.3, 0.3, 0.2] * 4,
        }, index=returns.index)

        result = brinson_results(
            returns.values,
            returns.values,
            portfolio_weights,
            benchmark_weights,
        )

        assert len(result) == 4
        assert "period" in result.columns
        # Total should match portfolio return - benchmark return
        total_active = result["total"].sum()
        portfolio_cum = np.cumprod(1 + returns.values[:, 0], axis=0)[-1] - 1
        benchmark_cum = np.cumprod(1 + returns.values[:, 0], axis=0)[-1] - 1
        np.testing.assert_almost_equal(total_active / 100, portfolio_cum - benchmark_cum, decimal=4)

    def test_brinson_class(self):
        """Test BrinsonAttribution class."""
        ba = BrinsonAttribution()

        # Test with sector mapping
        sector_map = {"AAPL": "Tech", "GOOG": "Tech", "MSFT": "Tech"}

        returns = pd.DataFrame({
            "AAPL": [0.05, 0.03, 0.02],
            "GOOG": [0.04, 0.03, 0.01],
            "MSFT": [0.06, 0.02, -0.01],
        })

        result = ba.calculate(returns)

        assert result.exposures.shape == (1, 2)  # 1 period, 2 styles (large/small)
        assert "large" in result.returns_by_style.columns
        assert "small" in result.returns_by_style.columns


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

        returns = pd.DataFrame(np.random.multivariate_normal(np.zeros(3), cov, size=periods*3))

        # Generate factors
        rf = 0.02  # Risk-free rate
        mktrf = rf + np.random.normal(0, 0.05, periods)  # Market
        smb = np.random.normal(0, 0.02, periods)  # Size
        hml = np.random.normal(0, 0.02, periods)  # Value
        rmw = np.random.normal(0, 0.01, periods)  # Profitability
        cma = np.random.normal(0, 0.01, periods)  # Investment

        factor_data = pd.DataFrame({
            "MKT": mktrf,
            "SMB": smb,
            "HML": hml,
            "RMW": rmw,
            "CMA": cma,
        }, index=returns.index)

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

        assert predicted.shape == returns.values.shape

    def test_get_factor_exposures(self, sample_data):
        """Test factor exposure calculation."""
        returns, factor_data = sample_data

        model = FamaFrenchModel()
        model.fit(returns, factor_data)

        # Get exposures
        exposures = model.get_factor_exposures(returns, factor_data)

        assert exposures.shape[0] == returns.shape[0]
        assert exposures.shape[1] == 5  # 3-factor

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


class TestStyleAnalysis:
    """Tests for style analysis."""

    @pytest.fixture
    def sample_data(self):
        """Create sample returns for testing."""
        np.random.seed(42)
        periods = 50

        # Simulate asset returns with different characteristics
        n_assets = 6

        # Large cap stocks (lower vol)
        large_returns = np.random.normal(0.08, 0.12, periods)

        # Small cap stocks (higher vol)
        small_returns = np.random.normal(0.12, 0.20, periods)

        # Growth stocks (higher returns)
        growth_returns = np.random.normal(0.15, 0.22, periods)

        # Value stocks
        value_returns = np.random.normal(0.10, 0.15, periods)

        # Momentum winners
        mom_returns = np.random.normal(0.18, 0.25, periods)

        # Combine all
        all_returns = np.column_stack([
            large_returns, small_returns, growth_returns,
            value_returns, mom_returns
        ])

        returns = pd.DataFrame(
            all_returns.reshape(-1, periods * n_assets),
            columns=["Large1", "Large2", "Small1", "Small2",
                   "Growth1", "Growth2", "Value1", "Value2",
                   "Mom1", "Mom2"]
        )

        return returns

    def test_style_analysis_basic(self, sample_data):
        """Test basic style analysis."""
        returns = sample_data

        result = style_analysis(returns)

        assert isinstance(result, StyleResult)
        assert len(result.exposures) == returns.shape[0]
        assert "large" in result.returns_by_style.columns
        assert "small" in result.returns_by_style.columns
        assert "winner" in result.returns_by_style.columns
        assert "loser" in result.returns_by_style.columns

    def test_style_analysis_with_caps(self, sample_data):
        """Test style analysis with market caps."""
        returns = sample_data

        # Create market caps
        market_caps = pd.Series({
            "Large1": 1e11, "Large2": 1e11,
            "Small1": 2e10, "Small2": 2e10,
            "Growth1": 5e10, "Growth2": 5e10,
            "Value1": 3e10, "Value2": 3e10,
            "Mom1": 2e10, "Mom2": 2e10,
        })

        result = style_analysis(returns, market_caps=market_caps)

        # Check that small cap has higher allocation
        small_total = result.returns_by_style["small"].sum().sum()
        large_total = result.returns_by_style["large"].sum().sum()
        assert small_total > large_total

    def test_regression_attribution(self, sample_data):
        """Test regression-based attribution."""
        returns = sample_data.iloc[:50, :3]  # Use first 3 assets

        # Create factor data
        factor_data = pd.DataFrame({
            "MKT": np.random.normal(0.05, 0.01, 50),
            "SMB": np.random.normal(0.02, 0.02, 50),
            "HML": np.random.normal(0.01, 0.01, 50),
        }, index=returns.index)

        # Use first asset as portfolio
        port_returns = returns.iloc[:, 0]

        result = calculate_regression_attribution(port_returns, factor_data)

        assert "alpha_attribution" in result
        assert "common_return" in result
        assert "specific_return" in result
        assert abs(result["alpha_attribution"] - result["specific_return"]) < 0.1  # Check consistency


class TestStyleTilts:
    """Tests for rolling style tilts."""

    @pytest.fixture
    def sample_data(self):
        """Create sample returns for testing."""
        np.random.seed(42)
        periods = 100

        # Create returns with style persistence
        returns = pd.DataFrame(np.random.randn(periods, 4), columns=["A", "B", "C", "D"])

        # Add style persistence: large caps continue to be large
        # This requires generating correlated returns

        return returns

    def test_calculate_style_tilts(self, sample_data):
        """Test style tilt calculation."""
        returns = sample_data

        # Use market cap as style proxy
        market_caps = pd.Series({"A": 1e11, "B": 5e10, "C": 3e10, "D": 2e10})

        tilts = calculate_style_tilts(returns, window=252)

        assert tilts.shape[0] == returns.shape[0]
        assert "large" in tilts.columns
        assert "small" in tilts.columns
        assert "winner" in tilts.columns
        assert "loser" in tilts.columns
