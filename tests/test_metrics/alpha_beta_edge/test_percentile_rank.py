"""Tests for alpha_percentile_rank edge cases.

Tests for alpha percentile rank with insufficient data and NaN handling.
Split from test_alpha_beta_edge_cases.py for maintainability.
"""

import numpy as np
import pandas as pd
import pytest

from fincore.metrics import alpha_beta


@pytest.mark.p2  # Medium: edge case tests
class TestAlphaPercentileRankEdgeCases:
    """Test alpha_percentile_rank edge cases."""

    def test_alpha_percentile_rank_returns_nan_for_insufficient_data(self):
        """Test alpha_percentile_rank returns NaN for strategy with less than 3 returns (line 639-640, 645)."""
        strategy_returns = pd.Series([0.01, 0.02])
        all_strategies_returns = [
            pd.Series([0.01, 0.02, 0.015]),
        ]
        factor_returns = pd.Series([0.005, 0.01, 0.008])

        result = alpha_beta.alpha_percentile_rank(
            strategy_returns,
            all_strategies_returns,
            factor_returns,
        )
        assert np.isnan(result)

    def test_alpha_percentile_rank_returns_nan_for_nan_strategy_alpha(self):
        """Test alpha_percentile_rank returns NaN when strategy alpha is NaN (line 644-645)."""
        strategy_returns = pd.Series([np.nan, np.nan, np.nan])
        all_strategies_returns = [
            pd.Series([0.01, 0.02, 0.015]),
        ]
        factor_returns = pd.Series([0.005, 0.01, 0.008])

        result = alpha_beta.alpha_percentile_rank(
            strategy_returns,
            all_strategies_returns,
            factor_returns,
        )
        assert np.isnan(result)

    def test_alpha_percentile_rank_returns_nan_for_no_valid_peer_alphas(self):
        """Test alpha_percentile_rank returns NaN when all peer alphas are NaN or insufficient (line 650, 656)."""
        strategy_returns = pd.Series([0.01, 0.02, 0.015])
        all_strategies_returns = [
            pd.Series([np.nan, np.nan, np.nan]),
            pd.Series([0.01]),  # Too short
        ]
        factor_returns = pd.Series([0.005, 0.01, 0.008])

        result = alpha_beta.alpha_percentile_rank(
            strategy_returns,
            all_strategies_returns,
            factor_returns,
        )
        assert np.isnan(result)

    def test_alpha_percentile_rank_skips_insufficient_peer_returns(self):
        """Test alpha_percentile_rank skips peer returns with less than 3 data points (line 649-650)."""
        strategy_returns = pd.Series([0.01, 0.02, 0.015])
        all_strategies_returns = [
            pd.Series([0.008, 0.012, 0.010]),  # Valid
            pd.Series([0.01]),  # Too short - should be skipped
        ]
        factor_returns = pd.Series([0.005, 0.01, 0.008])

        result = alpha_beta.alpha_percentile_rank(
            strategy_returns,
            all_strategies_returns,
            factor_returns,
        )
        # Should have a valid result since there's one valid peer
        assert not np.isnan(result)
        assert 0 <= result <= 1
