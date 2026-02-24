"""Tests for alpha_beta lines 557 and 610.

These lines are hit when annual_alphas/betas is empty after iterating.
"""

import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from fincore.metrics import alpha_beta


class TestAnnualAlphaLine557:
    """Test to cover line 557 in alpha_beta.py.

    Line 557 is hit when annual_alphas is empty after the loop.
    This happens when no years match between returns and factor returns.
    """

    def test_annual_alpha_no_matching_years_with_mock(self):
        """Test annual_alpha when no years match (line 557).

        We mock groupby to return empty groups for factor_returns.
        """
        returns = pd.Series(
            [0.01, 0.02, 0.015],
            index=pd.date_range("2020-01-01", periods=3),
        )
        factor_returns = pd.Series(
            [0.005, 0.01, 0.008],
            index=pd.date_range("2020-01-01", periods=3),
        )

        # We need to mock the groupby operation so that:
        # 1. grouped has the year 2020
        # 2. factor_grouped.groups.keys() returns empty (no matching years)

        with patch('fincore.metrics.alpha_beta.aligned_series') as mock_aligned:
            # Return the original series
            mock_aligned.return_value = (returns, factor_returns)

            # Create a mock groupby for factor_returns that returns empty groups
            original_groupby = pd.Series.groupby

            def mock_groupby_factory(self, by):
                """Factory that creates a mock groupby for factor_returns only."""
                if self is factor_returns:
                    mock_gb = MagicMock()
                    mock_gb.groups.keys.return_value = []  # Empty - no years
                    return mock_gb
                return original_groupby(self, by)

            with patch('pandas.Series.groupby', mock_groupby_factory):
                result = alpha_beta.annual_alpha(returns, factor_returns)

                # Should return empty Series (line 557)
                assert isinstance(result, pd.Series)
                assert len(result) == 0


class TestAnnualBetaLine610:
    """Test to cover line 610 in alpha_beta.py (similar to line 557)."""

    def test_annual_beta_no_matching_years_with_mock(self):
        """Test annual_beta when no years match (line 610).

        We mock groupby to return empty groups for factor_returns.
        """
        returns = pd.Series(
            [0.01, 0.02, 0.015],
            index=pd.date_range("2020-01-01", periods=3),
        )
        factor_returns = pd.Series(
            [0.005, 0.01, 0.008],
            index=pd.date_range("2020-01-01", periods=3),
        )

        with patch('fincore.metrics.alpha_beta.aligned_series') as mock_aligned:
            mock_aligned.return_value = (returns, factor_returns)

            original_groupby = pd.Series.groupby

            def mock_groupby_factory(self, by):
                if self is factor_returns:
                    mock_gb = MagicMock()
                    mock_gb.groups.keys.return_value = []  # Empty - no years
                    return mock_gb
                return original_groupby(self, by)

            with patch('pandas.Series.groupby', mock_groupby_factory):
                result = alpha_beta.annual_beta(returns, factor_returns)

                # Should return empty Series (line 610)
                assert isinstance(result, pd.Series)
                assert len(result) == 0
