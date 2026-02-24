"""Tests for calculate_style_tilts function.

Split from test_style.py for maintainability.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fincore.attribution.style import calculate_style_tilts


@pytest.mark.p1  # High: important style analysis function
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
