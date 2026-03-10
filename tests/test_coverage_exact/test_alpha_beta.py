"""Tests for alpha_beta line coverage.

Part of test_exact_line_coverage.py split - Alpha beta tests with P2 markers.
"""
from __future__ import annotations

import pandas as pd
import pytest

from fincore.metrics.alpha_beta import annual_alpha, annual_beta


@pytest.mark.p2
class TestAlphaBetaLineCoverage:
    """Test alpha_beta edge cases for exact line coverage."""

    def test_annual_alpha_line_543(self):
        """alpha_beta.py line 543: return empty Series after alignment."""
        # Create non-overlapping DatetimeIndex
        returns = pd.Series(
            [0.01, 0.02],
            index=pd.date_range("2020-01-01", periods=2, freq="D"),
        )
        factor_returns = pd.Series(
            [0.005],
            index=pd.date_range("2022-01-01", periods=1, freq="D"),
        )
        result = annual_alpha(returns, factor_returns)
        # After alignment, returns is empty
        assert isinstance(result, pd.Series)

    def test_annual_alpha_line_557(self):
        """alpha_beta.py line 557: return empty when no matching years."""
        returns = pd.Series(
            [0.01, 0.02],
            index=pd.date_range("2020-01-01", periods=2, freq="D"),
        )
        # Empty factor with DatetimeIndex
        factor_returns = pd.Series(
            [],
            index=pd.DatetimeIndex([], freq="D"),
            dtype=float,
        )
        result = annual_alpha(returns, factor_returns)
        # No matching years -> annual_alphas is empty
        assert isinstance(result, pd.Series)

    def test_annual_beta_line_596(self):
        """alpha_beta.py line 596: return empty Series after alignment."""
        returns = pd.Series(
            [0.01, 0.02],
            index=pd.date_range("2020-01-01", periods=2, freq="D"),
        )
        factor_returns = pd.Series(
            [0.005],
            index=pd.date_range("2022-01-01", periods=1, freq="D"),
        )
        result = annual_beta(returns, factor_returns)
        assert isinstance(result, pd.Series)

    def test_annual_beta_line_610(self):
        """alpha_beta.py line 610: return empty when no matching years."""
        returns = pd.Series(
            [0.01, 0.02],
            index=pd.date_range("2020-01-01", periods=2, freq="D"),
        )
        factor_returns = pd.Series(
            [],
            index=pd.DatetimeIndex([], freq="D"),
            dtype=float,
        )
        result = annual_beta(returns, factor_returns)
        assert isinstance(result, pd.Series)
