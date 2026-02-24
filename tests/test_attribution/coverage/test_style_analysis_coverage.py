"""Additional coverage tests for style analysis.

Split from test_style_more_coverage.py for maintainability.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fincore.attribution.style import style_analysis


@pytest.mark.p2  # Medium: coverage tests
class TestStyleAnalysisCoverage:
    """Additional coverage tests for style_analysis."""

    def test_style_analysis_value_scores_branch(self) -> None:
        """Test style_analysis with value_scores parameter."""
        idx = pd.date_range("2024-01-01", periods=30, freq="B")
        returns = pd.DataFrame(
            {"A": np.linspace(0.001, -0.001, len(idx)), "B": 0.0, "C": 0.0005},
            index=idx,
        )
        scores = pd.Series({"A": 0.1, "B": 0.9, "C": 0.5})
        out = style_analysis(returns, value_scores=scores)
        assert "value" in out.exposures.columns
        assert "growth" in out.exposures.columns
