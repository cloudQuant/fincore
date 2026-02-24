"""Tests for alpha_beta missing coverage lines 543, 557, 596, 610.

These lines are defensive checks that are hard to reach with normal input
due to how aligned_series works. We use mocking to force these paths.
"""

import pandas as pd
import pytest
from unittest.mock import patch

from fincore.metrics import alpha_beta


class TestAnnualAlphaLine543:
    """Test to cover line 543 in alpha_beta.py.

    Line 543 is hit when after aligned_series, len(returns) < 1.
    """

    def test_annual_alpha_empty_after_alignment(self):
        """Test annual_alpha when aligned series is empty (line 543)."""
        returns = pd.Series(
            [0.01, 0.02, 0.015],
            index=pd.date_range("2020-01-01", periods=3),
        )
        factor_returns = pd.Series(
            [0.005, 0.01, 0.008],
            index=pd.date_range("2020-01-01", periods=3),
        )

        # Mock aligned_series to return empty results
        with patch('fincore.metrics.alpha_beta.aligned_series') as mock_aligned:
            mock_aligned.return_value = (
                pd.Series([], dtype=float).rename(0),
                pd.Series([], dtype=float).rename(1),
            )

            result = alpha_beta.annual_alpha(returns, factor_returns)

            # Should return empty Series (line 543)
            assert isinstance(result, pd.Series)
            assert len(result) == 0


class TestAnnualBetaLine596:
    """Test to cover line 596 in alpha_beta.py (similar to line 543)."""

    def test_annual_beta_empty_after_alignment(self):
        """Test annual_beta when aligned series is empty (line 596)."""
        returns = pd.Series(
            [0.01, 0.02, 0.015],
            index=pd.date_range("2020-01-01", periods=3),
        )
        factor_returns = pd.Series(
            [0.005, 0.01, 0.008],
            index=pd.date_range("2020-01-01", periods=3),
        )

        # Mock aligned_series to return empty results
        with patch('fincore.metrics.alpha_beta.aligned_series') as mock_aligned:
            mock_aligned.return_value = (
                pd.Series([], dtype=float).rename(0),
                pd.Series([], dtype=float).rename(1),
            )

            result = alpha_beta.annual_beta(returns, factor_returns)

            # Should return empty Series (line 596)
            assert isinstance(result, pd.Series)
            assert len(result) == 0
