"""Test to document unreachable line 325 in drawdown.py.

Line 325 appears to be unreachable with normal input due to:
1. `len(returns) == 0` is never true because `returns` is copied once and never modified
2. `len(underwater) == 0` is never true because:
   - When recovery is NaT, `underwater.loc[:peak]` returns at least the peak element
   - When recovery is not NaT, `drop` removes inner elements but keeps peak and recovery

This test documents the current behavior and may be useful if the implementation changes.
"""

import pandas as pd
import pytest

from fincore.metrics import drawdown


class TestDrawdownLine325Unreachable:
    """Test to document that line 325 is currently unreachable.

    The line `if (len(returns) == 0) or (len(underwater) == 0): break` is a
    defensive check that cannot be triggered with normal input due to:
    - `returns` is never modified after being copied
    - `underwater` slicing never produces an empty result

    This test exists to document this behavior for future maintenance.
    """

    def test_get_top_drawdowns_normal_flow(self):
        """Test normal flow through get_top_drawdowns.

        The function terminates either:
        1. When no more drawdowns are found (line 313-314)
        2. When `top` drawdowns have been found (loop completes)

        Line 325 is never reached in normal operation.
        """
        # Test with multiple drawdowns
        idx = pd.date_range("2024-01-01", periods=10, freq="D")
        returns = pd.Series([
            0.01, -0.05, 0.03, 0.02, -0.08, 0.04, 0.01, -0.03, 0.02, 0.01
        ], index=idx)

        result = drawdown.get_top_drawdowns(returns, top=5)

        # Should return drawdowns
        assert isinstance(result, list)
        assert len(result) > 0

    def test_get_top_drawdowns_with_recovery(self):
        """Test when drawdowns have recovery (line 318 path)."""
        idx = pd.date_range("2024-01-01", periods=10, freq="D")
        # Pattern: peak, drop, recovery, peak, drop, recovery
        returns = pd.Series([
            0.05, -0.03, 0.04, 0.02, -0.05, 0.03, 0.01, -0.02, 0.01, 0.02
        ], index=idx)

        result = drawdown.get_top_drawdowns(returns, top=5)

        assert isinstance(result, list)

    def test_get_top_drawdowns_without_recovery(self):
        """Test when drawdowns don't recover (line 321 path)."""
        idx = pd.date_range("2024-01-01", periods=10, freq="D")
        # Continuous decline at the end
        returns = pd.Series([
            0.01, 0.02, 0.03, -0.05, -0.03, -0.02, -0.01, -0.01, -0.01, -0.01
        ], index=idx)

        result = drawdown.get_top_drawdowns(returns, top=5)

        assert isinstance(result, list)
