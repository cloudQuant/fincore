"""Tests for alpha function edge cases.

Tests for alpha when returns is a DataFrame.
Split from test_alpha_beta_edge_cases.py for maintainability.
"""

import pandas as pd
import pytest

from fincore.metrics import alpha_beta


@pytest.mark.p2  # Medium: edge case test
class TestAlphaEdgeCases:
    """Test alpha function edge cases."""

    def test_alpha_with_returns_dataframe(self):
        """Test alpha when returns is a DataFrame."""
        returns = pd.DataFrame(
            {
                "asset1": [0.01, 0.02, 0.015, 0.008, 0.012],
                "asset2": [0.008, 0.015, 0.012, 0.006, 0.010],
            }
        )
        factor_returns = pd.Series([0.005, 0.01, 0.008, 0.004, 0.006])

        result = alpha_beta.alpha(returns, factor_returns)
        assert isinstance(result, pd.Series)
        assert len(result) == 2
