"""Tests for analyze_performance_by_style and StyleResult edge cases.

Split from test_style.py for maintainability.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fincore.attribution.style import analyze_performance_by_style, style_analysis


@pytest.mark.p1  # High: important performance analysis function
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

    def test_analyze_performance_by_style_basic(self, sample_returns, sample_style_exposures):
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


@pytest.mark.p2  # Medium: edge case tests
class TestStyleResultEdgeCases:
    """Tests for StyleResult edge cases in style_summary property."""

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

    def test_style_summary_with_duplicate_index_labels(self, sample_returns):
        """Test style_summary when DataFrame has duplicate index labels."""
        result = style_analysis(sample_returns)

        # Create a returns_by_style DataFrame with duplicate index labels
        result._returns_by_style = pd.DataFrame(
            {"return": [0.01, 0.02]},
            index=["growth", "growth"],  # Duplicate label
        )
        result._returns_by_style.loc["growth", "return"] = 0.015

        summary = result.style_summary
        assert isinstance(summary, dict)

    def test_style_summary_with_empty_series(self, sample_returns):
        """Test style_summary when .loc returns empty Series (line 72-73)."""
        result = style_analysis(sample_returns)

        # Create a scenario where .loc returns an empty Series
        result._returns_by_style = pd.DataFrame(
            {"return": [0.01]},
            index=["value"],
        )

        # Try to access a non-existent style, which could return empty Series
        summary = result.style_summary
        assert isinstance(summary, dict)

    def test_style_summary_with_scalar_value(self, sample_returns):
        """Test style_summary when value is already a scalar (line 79)."""
        result = style_analysis(sample_returns)

        # Create a scenario where overall_returns contains a scalar value
        result._overall_returns = pd.Series([0.01, 0.02], index=["growth", "value"])
        result._overall_returns.loc["growth"] = 0.015  # This returns a scalar

        summary = result.style_summary
        assert isinstance(summary, dict)
        assert "growth" in summary
