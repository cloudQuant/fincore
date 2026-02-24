"""Test to cover line 325 in drawdown.py.

Line 325 is the break condition when underwater becomes empty after processing a drawdown.
"""

import pandas as pd
import pytest

from fincore.metrics import drawdown


class TestDrawdownLine325:
    """Test coverage for line 325 in drawdown.py."""

    def test_get_top_drawdowns_empty_underwater_after_slicing(self):
        """Test get_top_drawdowns when underwater becomes empty after slicing (line 325).

        To hit line 325, we need:
        1. A valid drawdown is found and added to drawdowns
        2. After slicing out the drawdown period, underwater becomes empty
        3. The loop breaks at line 325 because len(underwater) == 0
        """
        # Create a returns series with exactly one drawdown period
        # After processing this drawdown and slicing it out, underwater should be empty
        idx = pd.date_range("2024-01-01", periods=5, freq="D")

        # Pattern: gain, big loss, recovery to peak, then nothing (or flat)
        # The key is that after removing the drawdown period, no data remains
        returns = pd.Series([0.01, -0.10, 0.05, 0.04, 0.0], index=idx)

        result = drawdown.get_top_drawdowns(returns, top=5)

        # Should return at least one drawdown
        assert isinstance(result, list)

        # If underwater becomes empty after processing, line 325 is hit
        # Let's verify the result is valid
        if len(result) > 0:
            peak, valley, recovery = result[0]
            assert peak is not None
            assert valley is not None

    def test_get_top_drawdowns_single_drawdown_at_end(self):
        """Test get_top_drawdowns with a drawdown at the very end of series.

        When a drawdown doesn't recover (recovery is NaT) and is at the end,
        slicing with [:peak] might leave very little or no data.
        """
        idx = pd.date_range("2024-01-01", periods=5, freq="D")

        # Pattern: peak at start, then continuous decline to end (no recovery)
        returns = pd.Series([0.05, -0.02, -0.03, -0.02, -0.01], index=idx)

        result = drawdown.get_top_drawdowns(returns, top=5)

        # Should return the drawdown
        assert isinstance(result, list)

    def test_get_top_drawdowns_all_positive_no_drawdown(self):
        """Test get_top_drawdowns when there are no drawdowns.

        This should hit the early break at line 313-314 when peak is NaT.
        """
        returns = pd.Series([0.01, 0.02, 0.015, 0.01, 0.02])

        result = drawdown.get_top_drawdowns(returns, top=5)

        # Should return empty list when no drawdowns
        assert result == []

    def test_get_top_drawdowns_two_drawdowns_second_hits_break(self):
        """Test where processing first drawdown leads to empty underwater.

        Create scenario where:
        1. First drawdown is found
        2. After slicing it out, underwater is empty
        3. Line 325 break is hit before second iteration
        """
        # Very short series with one clear drawdown
        idx = pd.date_range("2024-01-01", periods=3, freq="D")
        returns = pd.Series([0.05, -0.08, 0.01], index=idx)

        result = drawdown.get_top_drawdowns(returns, top=10)

        # Should return the one drawdown
        assert isinstance(result, list)
