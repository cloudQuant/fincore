"""Tests for style analysis functions.

Tests for style attribution functions:
- style_analysis
- calculate_style_tilts
- calculate_regression_attribution
- analyze_performance_by_style
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fincore.attribution.style import (
    StyleResult,
    analyze_performance_by_style,
    calculate_regression_attribution,
    calculate_style_tilts,
    style_analysis,
)


class TestStyleAnalysis:
    """Tests for style_analysis function."""

    @pytest.fixture
    def sample_returns(self):
        """Create sample returns data for testing."""
        np.random.seed(42)
        periods = 100
        n_assets = 5
        assets = [f"ASSET_{i}" for i in range(n_assets)]

        returns = pd.DataFrame(
            np.random.normal(0.0005, 0.01, (periods, n_assets)),
            index=pd.date_range("2020-01-01", periods=periods),
            columns=assets,
        )
        return returns

    @pytest.fixture
    def sample_market_caps(self):
        """Create sample market cap data."""
        n_assets = 5
        assets = [f"ASSET_{i}" for i in range(n_assets)]

        market_caps = pd.Series(
            [1e9, 5e8, 2e8, 1e8, 5e7],
            index=assets,
        )
        return market_caps

    @pytest.fixture
    def sample_book_to_price(self):
        """Create sample book-to-price data."""
        n_assets = 5
        assets = [f"ASSET_{i}" for i in range(n_assets)]

        bp = pd.Series(
            [0.8, 0.9, 1.0, 1.1, 1.2],
            index=assets,
        )
        return bp

    def test_style_analysis_basic(self, sample_returns):
        """Test basic style analysis without optional data."""
        result = style_analysis(sample_returns)

        assert isinstance(result, StyleResult)
        assert isinstance(result.exposures, pd.DataFrame)
        assert isinstance(result.returns_by_style, pd.DataFrame)
        assert isinstance(result.overall_returns, pd.Series)

    def test_style_analysis_with_market_caps(
        self, sample_returns, sample_market_caps
    ):
        """Test style analysis with market cap data."""
        result = style_analysis(sample_returns, market_caps=sample_market_caps)

        # Should have size-based exposures
        assert "large" in result.exposures.index or "small" in result.exposures.index

    def test_style_analysis_with_book_to_price(
        self, sample_returns, sample_book_to_price
    ):
        """Test style analysis with book-to-price data."""
        result = style_analysis(sample_returns, book_to_price=sample_book_to_price)

        # Should have value/growth exposures
        assert "value" in result.returns_by_style.index or "growth" in result.returns_by_style.index

    def test_style_result_style_summary(self, sample_returns):
        """Test StyleResult.style_summary property."""
        result = style_analysis(sample_returns)
        summary = result.style_summary

        assert isinstance(summary, dict)
        assert len(summary) > 0

    def test_style_result_to_dict(self, sample_returns):
        """Test StyleResult.to_dict method."""
        result = style_analysis(sample_returns)
        d = result.to_dict()

        assert isinstance(d, dict)
        assert "exposures" in d
        assert "returns_by_style" in d
        assert "overall_returns" in d


class TestCalculateStyleTilts:
    """Tests for calculate_style_tilts function."""

    @pytest.fixture
    def sample_returns(self):
        """Create sample returns data."""
        np.random.seed(42)
        periods = 300  # Need more than window (252)
        n_assets = 3
        assets = [f"ASSET_{i}" for i in range(n_assets)]

        returns = pd.DataFrame(
            np.random.normal(0.0005, 0.01, (periods, n_assets)),
            index=pd.date_range("2020-01-01", periods=periods),
            columns=assets,
        )
        return returns

    def test_calculate_style_tilts_basic(self, sample_returns):
        """Test basic style tilts calculation."""
        result = calculate_style_tilts(sample_returns, window=100)

        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_calculate_style_tilts_columns(self, sample_returns):
        """Test that style tilts has expected column format."""
        result = calculate_style_tilts(sample_returns, window=100)

        # Check for expected column naming pattern
        expected_patterns = ["large", "small", "winner", "loser", "value", "growth"]
        has_expected = any(any(pat in col for pat in expected_patterns) for col in result.columns)
        assert has_expected, f"No expected style columns found in {result.columns.tolist()}"


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


class TestAnalyzePerformanceByStyle:
    """Tests for analyze_performance_by_style function."""

    @pytest.fixture
    def sample_returns(self):
        """Create sample returns data."""
        np.random.seed(42)
        periods = 50
        n_assets = 5
        assets = [f"ASSET_{i}" for i in range(n_assets)]

        returns = pd.DataFrame(
            np.random.normal(0.0005, 0.01, (periods, n_assets)),
            index=pd.date_range("2020-01-01", periods=periods),
            columns=assets,
        )
        return returns

    @pytest.fixture
    def sample_style_exposures(self):
        """Create sample style exposures."""
        np.random.seed(42)
        periods = 50
        n_assets = 5

        exposures = pd.DataFrame(
            np.random.randint(0, 2, (periods, n_assets)),
            columns=[f"ASSET_{i}" for i in range(n_assets)],
        )
        return exposures

    def test_analyze_performance_by_style_basic(
        self, sample_returns, sample_style_exposures
    ):
        """Test basic performance analysis by style."""
        result = analyze_performance_by_style(sample_returns, sample_style_exposures)

        assert isinstance(result, pd.DataFrame)
        # Should have Period column as index
        assert result.index.name == "Period" or "Period" in result.index.names

    def test_analyze_performance_by_style_empty_returns(self):
        """Test with empty returns data."""
        empty_returns = pd.DataFrame()
        empty_exposures = pd.DataFrame()

        result = analyze_performance_by_style(empty_returns, empty_exposures)

        # Should return empty DataFrame
        assert isinstance(result, pd.DataFrame)
