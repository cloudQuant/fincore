"""Tests for annual_alpha and annual_beta edge cases.

Tests for annual alpha/beta with empty inputs and non-datetime indices.
Split from test_alpha_beta_edge_cases.py for maintainability.
"""

import pandas as pd
import pytest

from fincore.metrics import alpha_beta


@pytest.mark.p2  # Medium: edge case tests
class TestAnnualAlphaEdgeCases:
    """Test annual_alpha edge cases."""

    def test_annual_alpha_returns_empty_series_for_empty_input(self):
        """Test annual_alpha returns empty Series for empty input (line 530, 537)."""
        returns = pd.Series([], dtype=float)
        factor_returns = pd.Series([], dtype=float)

        result = alpha_beta.annual_alpha(returns, factor_returns)
        assert isinstance(result, pd.Series)
        assert len(result) == 0

    def test_annual_alpha_returns_empty_series_for_non_datetime_index(self):
        """Test annual_alpha returns empty Series for non-DatetimeIndex (line 532-533)."""
        returns = pd.Series([0.01, 0.02, 0.015])
        factor_returns = pd.Series([0.005, 0.01, 0.008])

        result = alpha_beta.annual_alpha(returns, factor_returns)
        assert isinstance(result, pd.Series)
        assert len(result) == 0


@pytest.mark.p2  # Medium: edge case tests
class TestAnnualBetaEdgeCases:
    """Test annual_beta edge cases."""

    def test_annual_beta_returns_empty_series_for_empty_input(self):
        """Test annual_beta returns empty Series for empty input (line 582-583, 590)."""
        returns = pd.Series([], dtype=float)
        factor_returns = pd.Series([], dtype=float)

        result = alpha_beta.annual_beta(returns, factor_returns)
        assert isinstance(result, pd.Series)
        assert len(result) == 0

    def test_annual_beta_returns_empty_series_for_non_datetime_index(self):
        """Test annual_beta returns empty Series for non-DatetimeIndex (line 585-586)."""
        returns = pd.Series([0.01, 0.02, 0.015])
        factor_returns = pd.Series([0.005, 0.01, 0.008])

        result = alpha_beta.annual_beta(returns, factor_returns)
        assert isinstance(result, pd.Series)
        assert len(result) == 0
