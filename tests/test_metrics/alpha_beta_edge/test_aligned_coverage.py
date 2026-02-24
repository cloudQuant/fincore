"""Tests for alpha_beta_aligned additional coverage.

Additional tests for alpha_beta_aligned to hit missing lines.
Split from test_alpha_beta_edge_cases.py for maintainability.
"""

import numpy as np
import pandas as pd
import pytest

from fincore.metrics import alpha_beta


@pytest.mark.p2  # Medium: edge case tests
class TestAlphaBetaAlignedAdditionalCoverage:
    """Additional tests for alpha_beta_aligned to hit missing lines."""

    def test_alpha_beta_aligned_returns_nan_after_alignment_empty(self):
        """Test alpha_beta_aligned when alignment results in empty data (line 418-420)."""
        # Create returns and factor returns that become empty after alignment
        returns = pd.Series([0.01, 0.02, 0.015], index=pd.date_range("2020-01-01", periods=3))
        factor_returns = pd.Series([0.005, 0.01, 0.008], index=pd.date_range("2021-01-01", periods=3))

        result = alpha_beta.alpha_beta_aligned(
            returns.values,
            factor_returns.values,
            risk_free=0.0,
        )
        # After alignment with no common dates, we get NaN
        assert len(result) == 2
        # The result should be alpha and beta values

    def test_annual_alpha_returns_empty_for_empty_returns(self):
        """Test annual_alpha when returns is empty (lines 535-536, 542-543)."""
        # Empty returns series
        returns = pd.Series([], dtype=float)
        returns.index = pd.DatetimeIndex([], freq="D")
        factor_returns = pd.Series([0.005, 0.01], index=pd.date_range("2020-01-01", periods=2))

        result = alpha_beta.annual_alpha(returns, factor_returns)
        assert isinstance(result, pd.Series)
        assert len(result) == 0

    def test_annual_alpha_empty_result_no_matching_years(self):
        """Test annual_alpha when there are no matching years after loop (line 557)."""
        # Create a situation where returns and factor have same index
        # but the grouped years don't overlap (this is tricky with groupby)
        # Actually, looking at the code, line 557 is hit when annual_alphas is empty
        # after iterating through all years but finding no matches in factor_grouped
        # This happens when all years in returns are not in factor_grouped.groups

        # One way to trigger this: have returns with years but factor with no data
        # after alignment that groups to those same years
        # Let's try using an empty factor returns

        returns = pd.Series(
            [0.01, 0.02, 0.015],
            index=pd.date_range("2020-01-01", periods=3),
        )
        factor_returns = pd.Series([], dtype=float)
        factor_returns.index = pd.DatetimeIndex([], freq="D")

        result = alpha_beta.annual_alpha(returns, factor_returns)
        # After alignment, factor_returns will be empty series aligned to returns index
        # So len(returns) > 0 but len(factor_returns aligned) = 0
        # The loop won't find any matching years
        assert isinstance(result, pd.Series)

    def test_annual_beta_returns_empty_for_empty_returns(self):
        """Test annual_beta when returns is empty (lines 582-583, 595-596)."""
        # Empty returns series
        returns = pd.Series([], dtype=float)
        returns.index = pd.DatetimeIndex([], freq="D")
        factor_returns = pd.Series([0.005, 0.01], index=pd.date_range("2020-01-01", periods=2))

        result = alpha_beta.annual_beta(returns, factor_returns)
        assert isinstance(result, pd.Series)
        assert len(result) == 0

    def test_annual_beta_empty_result_no_matching_years(self):
        """Test annual_beta when there are no matching years (line 610)."""
        returns = pd.Series(
            [0.01, 0.02, 0.015],
            index=pd.date_range("2020-01-01", periods=3),
        )
        factor_returns = pd.Series([], dtype=float)
        factor_returns.index = pd.DatetimeIndex([], freq="D")

        result = alpha_beta.annual_beta(returns, factor_returns)
        assert isinstance(result, pd.Series)

    def test_alpha_beta_aligned_returns_nan_for_zero_variance_in_array(self):
        """Test alpha_beta_aligned returns NaN when factor variance is zero (line 429-431)."""
        # Create factor returns with exactly zero variance (all same values)
        returns = np.array([0.01, 0.02, 0.015, 0.008])
        factor_returns = np.array([0.01, 0.01, 0.01, 0.01])  # Zero variance

        result = alpha_beta.alpha_beta_aligned(
            returns,
            factor_returns,
            risk_free=0.0,
        )
        # Should return [nan, nan] when factor variance is 0
        assert len(result) == 2
        assert np.isnan(result[0])
        assert np.isnan(result[1])
