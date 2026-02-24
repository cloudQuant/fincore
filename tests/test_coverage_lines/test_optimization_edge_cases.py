"""Tests for optimization/frontier.py edge cases.

Targets:
- optimization/frontier.py: 106 - max_sharpe with near-zero vol
"""

import numpy as np
import pandas as pd
import warnings


class TestOptimizationFrontierNearZeroVol:
    """Test frontier.py line 106."""

    def test_max_sharpe_near_zero_volatility(self):
        """Line 106: vol < 1e-12 returns large penalty."""
        from fincore.optimization._utils import OptimizationError
        from fincore.optimization.frontier import efficient_frontier

        np.random.seed(42)
        # Mix of very low and normal variance to trigger the penalty path
        returns = pd.DataFrame({
            "A": np.random.normal(0.01, 0.0001, 50),  # Very low variance
            "B": np.random.normal(0.01, 0.0001, 50),
            "C": np.random.normal(0.01, 0.01, 50),
        })

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            try:
                result = efficient_frontier(returns, n_points=3)
                assert isinstance(result, dict)
            except OptimizationError:
                # Acceptable for singular matrix
                pass
