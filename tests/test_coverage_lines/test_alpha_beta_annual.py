"""Tests for metrics/alpha_beta.py annual function edge cases.

Targets:
- metrics/alpha_beta.py: 543, 557, 596, 610 - annual_alpha/annual_beta
"""

import pandas as pd
import pytest


class TestAnnualAlphaBetaEdgeCases:
    """Test annual_alpha and annual_beta edge cases."""

    def test_annual_alpha_empty_after_alignment(self):
        """Line 543: empty series after alignment."""
        from fincore.metrics.alpha_beta import annual_alpha

        returns = pd.Series(
            [0.01, 0.02, 0.015],
            index=pd.date_range("2020-01-01", periods=3),
        )
        # Non-overlapping dates
        factor_returns = pd.Series(
            [0.005, 0.01],
            index=pd.date_range("2021-01-01", periods=2),
        )

        result = annual_alpha(returns, factor_returns)
        # After alignment, no common dates
        assert isinstance(result, pd.Series)

    def test_annual_alpha_no_matching_years(self):
        """Line 557: no matching years found."""
        from fincore.metrics.alpha_beta import annual_alpha

        returns = pd.Series(
            [0.01, 0.02, 0.015],
            index=pd.date_range("2020-01-01", periods=3),
        )
        factor_returns = pd.Series([], dtype=float)
        factor_returns.index = pd.DatetimeIndex([], freq="D")

        result = annual_alpha(returns, factor_returns)
        assert isinstance(result, pd.Series)

    def test_annual_beta_empty_after_alignment(self):
        """Line 596: empty series after alignment."""
        from fincore.metrics.alpha_beta import annual_beta

        returns = pd.Series(
            [0.01, 0.02, 0.015],
            index=pd.date_range("2020-01-01", periods=3),
        )
        factor_returns = pd.Series(
            [0.005, 0.01],
            index=pd.date_range("2021-01-01", periods=2),
        )

        result = annual_beta(returns, factor_returns)
        assert isinstance(result, pd.Series)

    def test_annual_beta_no_matching_years(self):
        """Line 610: no matching years found."""
        from fincore.metrics.alpha_beta import annual_beta

        returns = pd.Series(
            [0.01, 0.02, 0.015],
            index=pd.date_range("2020-01-01", periods=3),
        )
        factor_returns = pd.Series([], dtype=float)
        factor_returns.index = pd.DatetimeIndex([], freq="D")

        result = annual_beta(returns, factor_returns)
        assert isinstance(result, pd.Series)
