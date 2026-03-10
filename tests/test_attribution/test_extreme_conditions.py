"""Edge case tests for performance attribution.

Tests for extreme market conditions using the perf_attrib API.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fincore.empyrical import Empyrical


@pytest.mark.p2
class TestAttributionExtremeConditions:
    """Tests for attribution under extreme market conditions."""

    def test_extreme_bull_market(self):
        """Test attribution during extreme bull market (>50% returns)."""
        # Use the conftest helper to generate test data
        from tests.test_pyfolio.perf_attrib.conftest import generate_toy_risk_model_output

        # Generate extended data for bull market scenario
        returns, positions, factor_returns, factor_loadings = generate_toy_risk_model_output(
            periods=100, num_styles=2
        )

        # Modify returns to simulate bull market
        returns = returns.abs() * 0.5  # All positive returns, strong trend
        returns = returns * (1 + np.arange(len(returns)) * 0.01)  # Add upward drift

        # Should compute attribution even with extreme returns
        emp = Empyrical()
        result = emp.perf_attrib(
            returns,
            positions,
            factor_returns,
            factor_loadings,
        )

        # Should produce valid attribution
        assert result is not None
        assert len(result) > 0

    def test_extreme_bear_market(self):
        """Test attribution during extreme bear market (<-50% returns)."""
        from tests.test_pyfolio.perf_attrib.conftest import generate_toy_risk_model_output

        # Generate test data
        returns, positions, factor_returns, factor_loadings = generate_toy_risk_model_output(
            periods=100, num_styles=2
        )

        # Modify returns to simulate bear market
        returns = -returns.abs() * 0.5  # All negative returns
        returns = returns * (1 - np.arange(len(returns)) * 0.005)  # Add downward drift

        # Should compute attribution even with extreme returns
        emp = Empyrical()
        result = emp.perf_attrib(
            returns,
            positions,
            factor_returns,
            factor_loadings,
        )

        # Should produce valid attribution
        assert result is not None
        assert len(result) > 0

    def test_high_volatility_regime(self):
        """Test attribution during high volatility period."""
        from tests.test_pyfolio.perf_attrib.conftest import generate_toy_risk_model_output

        # Generate test data
        returns, positions, factor_returns, factor_loadings = generate_toy_risk_model_output(
            periods=100, num_styles=2
        )

        # Increase volatility dramatically
        returns = returns * 5  # 5x the returns

        # Should handle high volatility
        emp = Empyrical()
        result = emp.perf_attrib(
            returns,
            positions,
            factor_returns,
            factor_loadings,
        )

        # Should produce valid attribution
        assert result is not None
        assert len(result) > 0


@pytest.mark.p2
class TestAttributionMultiAsset:
    """Tests for attribution with multi-asset portfolios."""

    def test_multi_asset_attribution(self):
        """Test attribution for portfolio with multiple assets."""
        from tests.test_pyfolio.perf_attrib.conftest import generate_toy_risk_model_output

        # Generate data with multiple tickers (default is 3 tickers)
        returns, positions, factor_returns, factor_loadings = generate_toy_risk_model_output(
            periods=50, num_styles=2
        )

        # Should compute attribution for all assets
        emp = Empyrical()
        result = emp.perf_attrib(
            returns,
            positions,
            factor_returns,
            factor_loadings,
        )

        # Should produce attribution per ticker
        assert result is not None
        assert len(result) > 0


@pytest.mark.p2
class TestAttributionMissingData:
    """Tests for attribution with missing data scenarios."""

    def test_attribution_short_series(self):
        """Test attribution with very short return series."""
        from tests.test_pyfolio.perf_attrib.conftest import generate_toy_risk_model_output

        # Generate minimal data (5 days)
        returns, positions, factor_returns, factor_loadings = generate_toy_risk_model_output(
            periods=5, num_styles=1
        )

        # Should handle short series
        emp = Empyrical()
        result = emp.perf_attrib(
            returns,
            positions,
            factor_returns,
            factor_loadings,
        )

        # Should produce valid result even with short data
        assert result is not None

    def test_attribution_with_zeros(self):
        """Test attribution with zero returns."""
        from tests.test_pyfolio.perf_attrib.conftest import generate_toy_risk_model_output

        # Generate test data
        returns, positions, factor_returns, factor_loadings = generate_toy_risk_model_output(
            periods=50, num_styles=2
        )

        # Set some returns to zero
        returns.iloc[10:15] = 0

        # Should handle zero returns
        emp = Empyrical()
        result = emp.perf_attrib(
            returns,
            positions,
            factor_returns,
            factor_loadings,
        )

        # Should produce valid result
        assert result is not None
        assert len(result) > 0


@pytest.mark.p3
class TestAttributionEdgeCases:
    """Additional edge cases for attribution."""

    def test_single_style_attribution(self):
        """Test attribution with single factor model."""
        from tests.test_pyfolio.perf_attrib.conftest import generate_toy_risk_model_output

        # Generate data with 1 style
        returns, positions, factor_returns, factor_loadings = generate_toy_risk_model_output(
            periods=50, num_styles=1
        )

        # Should work with single factor
        emp = Empyrical()
        result = emp.perf_attrib(
            returns,
            positions,
            factor_returns,
            factor_loadings,
        )

        # Should produce valid result
        assert result is not None
        assert len(result) > 0
