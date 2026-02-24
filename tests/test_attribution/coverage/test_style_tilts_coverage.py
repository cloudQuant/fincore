"""Coverage tests for calculate_style_tilts.

Split from test_style_more_coverage.py for maintainability.
"""
from __future__ import annotations

import pandas as pd
import pytest

from fincore.attribution.style import calculate_style_tilts


@pytest.mark.p2  # Medium: coverage tests
class TestCalculateStyleTiltsCoverage:
    """Coverage tests for calculate_style_tilts."""

    def test_calculate_style_tilts_effective_window_too_small_returns_empty(self) -> None:
        """Test calculate_style_tilts when effective window is too small."""
        tiny = pd.DataFrame(
            {"A": [0.01, -0.01], "B": [0.0, 0.0]},
            index=pd.date_range("2024-01-01", periods=2, freq="B"),
        )
        out = calculate_style_tilts(tiny, window=252)
        assert out.empty
