"""Tests for style_analysis function.

Split from test_style.py for maintainability.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fincore.attribution.style import StyleResult, style_analysis


@pytest.mark.p1  # High: important style analysis function
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

    def test_style_analysis_with_market_caps(self, sample_returns, sample_market_caps):
        """Test style analysis with market cap data."""
        result = style_analysis(sample_returns, market_caps=sample_market_caps)

        # Should have size-based exposures
        assert "large" in result.exposures.columns
        assert "small" in result.exposures.columns

    def test_style_analysis_with_book_to_price(self, sample_returns, sample_book_to_price):
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
