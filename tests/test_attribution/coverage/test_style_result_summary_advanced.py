"""Advanced edge case tests for StyleResult.style_summary.

Split from test_style_more_coverage.py for maintainability.
"""
from __future__ import annotations

import pandas as pd
import pytest

from fincore.attribution.style import StyleResult


@pytest.mark.p2  # Medium: advanced edge case tests
class TestStyleResultSummaryAdvanced:
    """Advanced edge case tests for StyleResult.style_summary."""

    def test_style_summary_nonexistent_style_with_duplicate_index(self) -> None:
        """Test style_summary when accessing non-existent style with duplicate index."""
        exposures = pd.DataFrame({"value": [1.0]}, index=["A"])
        overall = pd.Series([0.0], index=[pd.Timestamp("2024-01-01")])

        rbs_idx = pd.DataFrame({"return": [0.1, 0.2]}, index=pd.Index(["value", "growth"], name="style"))
        sr = StyleResult(exposures=exposures, returns_by_style=rbs_idx, overall_returns=overall)
        summ = sr.style_summary
        assert summ["value"] == 0.1
        assert summ["growth"] == 0.2

    def test_style_summary_empty_series_from_loc(self) -> None:
        """Test style_summary when .loc returns empty Series."""
        exposures = pd.DataFrame({"value": [1.0]}, index=["A"])
        overall = pd.Series([0.0], index=[pd.Timestamp("2024-01-01")])

        rbs_empty = pd.DataFrame({"returns": []}, index=pd.Index([], name="style"))
        sr = StyleResult(exposures=exposures, returns_by_style=rbs_empty, overall_returns=overall)
        summ = sr.style_summary
        assert summ == {}

    def test_style_summary_duplicate_index_empty_dataframe(self) -> None:
        """Test line 66: DataFrame returned by .loc is empty."""
        exposures = pd.DataFrame({"value": [1.0]}, index=["A"])
        overall = pd.Series([0.0], index=[pd.Timestamp("2024-01-01")])

        class CustomDF:
            def __init__(self, df):
                self._df = df
                self._index = pd.Index(["value"], name="style")

            @property
            def columns(self):
                return pd.Index([])

            @property
            def index(self):
                return self._index

            @property
            def loc(self):
                class _LocIndexer:
                    def __init__(self, parent):
                        self.parent = parent

                    def __getitem__(self, label):
                        return pd.DataFrame()

                return _LocIndexer(self)

        sr = StyleResult(exposures=exposures, returns_by_style=CustomDF(pd.DataFrame()), overall_returns=overall)  # type: ignore
        summ = sr.style_summary
        assert summ["value"] == 0.0

    def test_style_summary_duplicate_index_empty_series(self) -> None:
        """Test line 73: Series returned by .loc is empty."""
        exposures = pd.DataFrame({"value": [1.0]}, index=["A"])
        overall = pd.Series([0.0], index=[pd.Timestamp("2024-01-01")])

        class CustomDF:
            def __init__(self, df):
                self._df = df
                self._index = pd.Index(["value"], name="style")

            @property
            def columns(self):
                return pd.Index([])

            @property
            def index(self):
                return self._index

            @property
            def loc(self):
                class _LocIndexer:
                    def __init__(self, parent):
                        self.parent = parent

                    def __getitem__(self, label):
                        return pd.Series(dtype=float)

                return _LocIndexer(self)

        sr = StyleResult(exposures=exposures, returns_by_style=CustomDF(pd.DataFrame()), overall_returns=overall)  # type: ignore
        summ = sr.style_summary
        assert summ["value"] == 0.0

    def test_style_summary_scalar_value(self) -> None:
        """Test line 79: val is already a scalar."""
        exposures = pd.DataFrame({"value": [1.0]}, index=["A"])
        overall = pd.Series([0.0], index=[pd.Timestamp("2024-01-01")])

        class CustomDF:
            def __init__(self):
                self._index = pd.Index(["value"], name="style")

            @property
            def columns(self):
                return pd.Index([])

            @property
            def index(self):
                return self._index

            @property
            def loc(self):
                class _LocIndexer:
                    def __init__(self, parent):
                        self.parent = parent

                    def __getitem__(self, label):
                        return 0.5

                return _LocIndexer(self)

        sr = StyleResult(exposures=exposures, returns_by_style=CustomDF(), overall_returns=overall)  # type: ignore
        summ = sr.style_summary
        assert summ["value"] == 0.5
