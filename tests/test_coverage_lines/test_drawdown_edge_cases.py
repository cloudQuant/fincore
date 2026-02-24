"""Tests for metrics/drawdown.py edge cases.

Targets:
- metrics/drawdown.py: 325 - get_all_drawdowns break condition
"""

import pandas as pd


class TestDrawdownBreakCondition:
    """Test drawdown.py line 325."""

    def test_get_all_drawdowns_break_condition(self):
        """Line 325: break when returns or underwater is empty."""
        from fincore.metrics.drawdown import get_all_drawdowns

        # Create returns that will empty underwater during iteration
        idx = pd.date_range("2024-01-01", periods=10, freq="B")
        returns = pd.Series([0.01, -0.05, 0.02, 0.0] * 2 + [0.0, 0.0], index=idx)

        result = get_all_drawdowns(returns)
        assert isinstance(result, list)
