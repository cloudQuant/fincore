"""Coverage tests for calculate_regression_attribution.

Split from test_style_more_coverage.py for maintainability.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fincore.attribution.style import calculate_regression_attribution


@pytest.mark.p2  # Medium: coverage tests
class TestCalculateRegressionAttributionCoverage:
    """Coverage tests for calculate_regression_attribution."""

    def test_calculate_regression_attribution_raises_when_missing_style_inputs_for_series(self) -> None:
        """Test that TypeError is raised when style_returns is not provided for Series input."""
        idx = pd.date_range("2024-01-01", periods=10, freq="B")
        port = pd.Series(np.linspace(0.001, -0.001, len(idx)), index=idx)
        with pytest.raises(TypeError, match="must be a DataFrame"):
            calculate_regression_attribution(port)

    def test_calculate_regression_attribution_skip_and_short_series_branches(self) -> None:
        """Test various edge cases in regression attribution."""
        idx = pd.date_range("2024-01-01", periods=5, freq="B")
        port = pd.Series([0.01, 0.0, 0.01, 0.0, 0.01], index=idx)

        # Skip when style is not in exposures.
        style_returns = pd.DataFrame({"foo": port.values}, index=idx)
        exposures = pd.DataFrame({"bar": [1.0]}, index=["A"])
        out = calculate_regression_attribution(port, style_returns=style_returns, style_exposures=exposures)
        assert "foo" not in out
        assert "residual" in out

        # common_idx < 3 => attribution forced to 0.0
        style_returns_short = pd.DataFrame({"value": [0.01, 0.02]}, index=idx[:2])
        exposures2 = pd.DataFrame({"value": [1.0]}, index=["A"])
        out2 = calculate_regression_attribution(port, style_returns=style_returns_short, style_exposures=exposures2)
        assert out2["value"] == 0.0

        # valid_mask.sum() < 3 => attribution forced to 0.0
        style_returns_nan = pd.DataFrame({"value": [np.nan, np.nan, 0.01, np.nan, np.nan]}, index=idx)
        out3 = calculate_regression_attribution(port, style_returns=style_returns_nan, style_exposures=exposures2)
        assert out3["value"] == 0.0
