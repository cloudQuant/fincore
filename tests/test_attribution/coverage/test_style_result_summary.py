"""Tests for StyleResult.style_summary edge cases.

Split from test_style_more_coverage.py for maintainability.
"""
from __future__ import annotations

import pandas as pd
import pytest

from fincore.attribution.style import StyleResult, analyze_performance_by_style


@pytest.mark.p2  # Medium: edge case tests
class TestStyleResultStyleSummary:
    """Tests for StyleResult.style_summary with various input formats."""

    def test_style_result_style_summary_series_and_dataframe_variants(self) -> None:
        """Test style_summary with Series, DataFrame with column, and indexed DataFrame."""
        exposures = pd.DataFrame({"value": [1.0]}, index=["A"])
        overall = pd.Series([0.0], index=[pd.Timestamp("2024-01-01")])

        # returns_by_style as Series
        rbs_s = pd.Series({"value": 0.1, "growth": -0.2})
        sr = StyleResult(exposures=exposures, returns_by_style=rbs_s, overall_returns=overall)
        summ = sr.style_summary
        assert summ["value"] == 0.1
        assert summ["growth"] == -0.2

        # returns_by_style with explicit "style" column
        rbs_df = pd.DataFrame({"style": ["value", "growth"], "return": [0.1, -0.2]})
        sr2 = StyleResult(exposures=exposures, returns_by_style=rbs_df, overall_returns=overall)
        summ2 = sr2.style_summary
        assert summ2["value"] == 0.1
        assert summ2["growth"] == -0.2

        # returns_by_style as DataFrame indexed by style (including duplicate labels)
        rbs_idx = pd.DataFrame({"return": [0.1, 0.2]}, index=pd.Index(["value", "value"], name="style"))
        sr3 = StyleResult(exposures=exposures, returns_by_style=rbs_idx, overall_returns=overall)
        summ3 = sr3.style_summary
        assert summ3["value"] == 0.1

    def test_style_result_empty_dataframe_in_summary(self) -> None:
        """Test style_summary when .loc returns an empty DataFrame."""
        exposures = pd.DataFrame({"value": [1.0]}, index=["A"])
        overall = pd.Series([0.0], index=[pd.Timestamp("2024-01-01")])

        rbs_idx = pd.DataFrame({"return": [0.1, 0.2]}, index=["value", "growth"])
        sr = StyleResult(exposures=exposures, returns_by_style=rbs_idx, overall_returns=overall)
        summ = sr.style_summary
        assert "value" in summ
        assert "growth" in summ

    def test_style_result_empty_series_in_summary(self) -> None:
        """Test style_summary when .loc returns an empty Series."""
        exposures = pd.DataFrame({"value": [1.0]}, index=["A"])
        overall = pd.Series([0.0], index=[pd.Timestamp("2024-01-01")])

        rbs_series = pd.Series({"value": 0.1, "growth": 0.2})
        sr = StyleResult(exposures=exposures, returns_by_style=rbs_series, overall_returns=overall)
        summ = sr.style_summary
        assert summ["value"] == 0.1

    def test_style_result_return_column_in_index(self) -> None:
        """Test style_summary when 'return' is in the Series index."""
        exposures = pd.DataFrame({"value": [1.0]}, index=["A"])
        overall = pd.Series([0.0], index=[pd.Timestamp("2024-01-01")])

        rbs_series = pd.Series([0.1, 0.2], index=["return", "value"])
        sr = StyleResult(exposures=exposures, returns_by_style=rbs_series, overall_returns=overall)
        summ = sr.style_summary
        assert summ["return"] == 0.1

    def test_style_result_dataframe_no_return_column(self) -> None:
        """Test style_summary when DataFrame has duplicate index but no 'return' column."""
        exposures = pd.DataFrame({"value": [1.0]}, index=["A"])
        overall = pd.Series([0.0], index=[pd.Timestamp("2024-01-01")])

        rbs_idx = pd.DataFrame({"rets": [0.1, 0.2]}, index=pd.Index(["value", "value"], name="style"))
        sr = StyleResult(exposures=exposures, returns_by_style=rbs_idx, overall_returns=overall)
        summ = sr.style_summary
        assert summ["value"] == 0.1

    def test_style_result_series_no_return_in_index(self) -> None:
        """Test style_summary when Series has no 'return' in index."""
        exposures = pd.DataFrame({"value": [1.0]}, index=["A"])
        overall = pd.Series([0.0], index=[pd.Timestamp("2024-01-01")])

        rbs_idx = pd.DataFrame({"performance": [0.1]}, index=["value"])
        sr = StyleResult(exposures=exposures, returns_by_style=rbs_idx, overall_returns=overall)
        summ = sr.style_summary
        assert summ["value"] == 0.1

    def test_analyze_performance_by_style_returns_empty_when_exposures_empty(self) -> None:
        """Test analyze_performance_by_style returns empty when exposures empty."""
        idx = pd.date_range("2024-01-01", periods=5, freq="B")
        returns = pd.DataFrame({"A": [0.01, 0.0, 0.01, 0.0, 0.01]}, index=idx)
        out = analyze_performance_by_style(returns, style_exposures=pd.DataFrame())
        assert out.empty
