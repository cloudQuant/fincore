"""Tests for calculate_regression_attribution function.

Split from test_style.py for maintainability.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from fincore.attribution.style import calculate_regression_attribution


@pytest.mark.p1  # High: important attribution function
class TestCalculateRegressionAttribution:
    """Tests for calculate_regression_attribution function."""

    @pytest.fixture
    def sample_portfolio_returns(self):
        """Create sample portfolio returns."""
        np.random.seed(42)
        periods = 100

        returns = pd.Series(
            np.random.normal(0.0005, 0.01, periods),
            index=pd.date_range("2020-01-01", periods=periods),
        )
        return returns

    @pytest.fixture
    def sample_style_returns(self):
        """Create sample style factor returns."""
        np.random.seed(42)
        periods = 100

        returns = pd.DataFrame(
            {
                "value": np.random.normal(0.0003, 0.01, periods),
                "growth": np.random.normal(0.0007, 0.015, periods),
                "large": np.random.normal(0.0004, 0.008, periods),
                "small": np.random.normal(0.0006, 0.012, periods),
            },
            index=pd.date_range("2020-01-01", periods=periods),
        )
        return returns

    @pytest.fixture
    def sample_style_exposures(self):
        """Create sample style exposures."""
        np.random.seed(42)

        exposures = pd.DataFrame(
            {
                "value": [0.5, 0.3, 0.2],
                "growth": [0.5, 0.7, 0.8],
                "large": [0.6, 0.4, 0.3],
                "small": [0.4, 0.6, 0.7],
            }
        )
        return exposures

    def test_regression_attribution_basic(
        self, sample_portfolio_returns, sample_style_returns, sample_style_exposures
    ):
        """Test basic regression attribution."""
        result = calculate_regression_attribution(
            sample_portfolio_returns, sample_style_returns, sample_style_exposures
        )

        assert isinstance(result, dict)
        assert len(result) > 0

    def test_regression_attribution_has_residual(
        self, sample_portfolio_returns, sample_style_returns, sample_style_exposures
    ):
        """Test that regression attribution includes residual."""
        result = calculate_regression_attribution(
            sample_portfolio_returns, sample_style_returns, sample_style_exposures
        )

        assert "residual" in result

    def test_regression_attribution_constant_series_no_runtime_warning(self):
        """Constant inputs should not raise runtime warnings."""
        idx = pd.date_range("2020-01-01", periods=20)
        portfolio_returns = pd.Series(np.zeros(20), index=idx)
        style_returns = pd.DataFrame({"value": np.zeros(20), "growth": np.zeros(20)}, index=idx)
        style_exposures = pd.DataFrame({"value": [1.0], "growth": [0.0]})

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = calculate_regression_attribution(portfolio_returns, style_returns, style_exposures)

        assert "residual" in result
        runtime_warnings = [w for w in caught if issubclass(w.category, RuntimeWarning)]
        assert len(runtime_warnings) == 0
