"""Tests for private helper functions in style module.

Split from test_style_more_coverage.py for maintainability.
"""
from __future__ import annotations

import pandas as pd
import pytest

from fincore.attribution import style as style_mod


@pytest.mark.p2  # Medium: private helper tests
class TestPrivateHelpers:
    """Tests for private helper functions."""

    def test_private_helpers_momentum_and_lookback_and_size_rank(self) -> None:
        """Test _calculate_momentum, _exposure_from_lookback, _size_rank_to_exposure."""
        idx = pd.date_range("2024-01-01", periods=5, freq="B")
        rets = pd.DataFrame(
            {"A": [0.01, 0.0, 0.0, 0.0, 0.0], "B": [-0.01, 0.0, 0.0, 0.0, 0.0]},
            index=idx
        )

        mom = style_mod._calculate_momentum(rets, window=2)
        assert isinstance(mom, pd.DataFrame)
        assert mom.shape[0] == 1

        pos = style_mod._exposure_from_lookback(rets, periods=2, direction="positive")
        neg = style_mod._exposure_from_lookback(rets, periods=2, direction="negative")
        assert pos.shape == (1, 2)
        assert neg.shape == (1, 2)

        ranks = pd.Series({"A": 0.2, "B": 0.8})
        exp = style_mod._size_rank_to_exposure(ranks)
        assert set(exp.index) == {"large", "small"}
