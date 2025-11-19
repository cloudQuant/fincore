"""
Tests for ranking and Cornell model.
"""
from __future__ import division

import numpy as np
import pandas as pd
from unittest import TestCase

from fincore.empyrical import Empyrical

DECIMAL_PLACES = 4


class TestRankingAndCornell(TestCase):
    """Test cases for alpha ranking and Cornell model."""

    np.random.seed(42)

    # Sample strategy returns
    strategy_returns = pd.Series(
        np.random.randn(500) / 100 + 0.0003,
        index=pd.date_range('2020-1-1', periods=500, freq='D'))

    market_returns = pd.Series(
        np.random.randn(500) / 100 + 0.0002,
        index=pd.date_range('2020-1-1', periods=500, freq='D'))

    # Multiple strategies for ranking
    num_strategies = 10
    all_strategies = {
        f'strategy_{i}': pd.Series(
            np.random.randn(500) / 100 + np.random.randn() * 0.0001,
            index=pd.date_range('2020-1-1', periods=500, freq='D'))
        for i in range(num_strategies)
    }

    # Test alpha ranking
    def test_alpha_percentile_rank(self):
        """Test alpha percentile ranking among multiple strategies."""
        emp = Empyrical()
        result = emp.alpha_percentile_rank(
            self.strategy_returns,
            list(self.all_strategies.values()),
            self.market_returns
        )
        # Should return a value between 0 and 1
        assert 0 <= result <= 1, f"Percentile should be in [0,1], got {result}"

    def test_alpha_percentile_rank_best(self):
        """Test ranking when strategy is the best."""
        # Create a strategy that's clearly the best
        best_returns = self.market_returns + 0.001
        others = list(self.all_strategies.values())[:5]

        emp = Empyrical()
        result = emp.alpha_percentile_rank(
            best_returns,
            others,
            self.market_returns
        )
        # Should be close to 1.0 (top performer)
        assert result >= 0.5, f"Best strategy should rank high, got {result}"

    def test_alpha_percentile_rank_worst(self):
        """Test ranking when strategy is the worst."""
        # Create a strategy that's clearly the worst
        worst_returns = self.market_returns - 0.001
        others = list(self.all_strategies.values())[:5]

        emp = Empyrical()
        result = emp.alpha_percentile_rank(
            worst_returns,
            others,
            self.market_returns
        )
        # Should be close to 0.0 (bottom performer)
        assert result <= 0.5, f"Worst strategy should rank low, got {result}"

    def test_alpha_percentile_rank_empty(self):
        """Test that empty returns give NaN."""
        empty_returns = pd.Series([], dtype=float)
        emp = Empyrical()
        result = emp.alpha_percentile_rank(
            empty_returns,
            list(self.all_strategies.values()),
            self.market_returns
        )
        assert np.isnan(result)

    # Test Cornell timing model
    def test_cornell_timing(self):
        """Test Cornell market timing coefficient."""
        emp = Empyrical()
        result = emp.cornell_timing(
            self.strategy_returns,
            self.market_returns
        )
        # Should return a valid number
        assert isinstance(result, (float, np.floating))

    def test_cornell_timing_empty(self):
        """Test that empty returns give NaN."""
        empty_returns = pd.Series([], dtype=float)
        emp = Empyrical()
        result = emp.cornell_timing(
            empty_returns,
            self.market_returns
        )
        assert np.isnan(result)

    def test_cornell_timing_values(self):
        """Test Cornell timing with known behavior."""
        # Create returns with timing ability
        timing_signal = np.where(self.market_returns.values > 0, 0.0002, -0.0001)
        timing_returns = pd.Series(
            np.random.randn(500) / 100 + 0.0003 + timing_signal,
            index=pd.date_range('2020-1-1', periods=500, freq='D'))

        emp = Empyrical()
        gamma = emp.cornell_timing(timing_returns, self.market_returns)
        # Should be a valid number
        assert not np.isnan(gamma)
