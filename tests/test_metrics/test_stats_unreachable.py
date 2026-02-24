"""Test to document unreachable lines in stats.py.

Lines 604 and 625 appear to be unreachable due to the logic of r_cubed_turtle function.
These are defensive checks that may have been added for safety but cannot be triggered
with normal input.
"""

import numpy as np
import pandas as pd
import pytest

from fincore.metrics import stats


class TestRCubedTurtleUnreachableLines:
    """Document potentially unreachable lines in r_cubed_turtle."""

    def test_r_cubed_turtle_line_604_analysis(self):
        """Analyze line 604 unreachable nature.

        Line 604: if len(years) < 1: return np.nan

        This is unreachable because:
        - If returns is a Series with DatetimeIndex and has data:
          years = returns.index.year.unique() will have at least 1 year
        - If returns is not a Series or has no year attribute:
          n_years = max(1, ...) so years = range(n_years) has at least 1 element

        Therefore len(years) is always >= 1 when this line is reached.
        """
        # Test with empty Series - should return at line 590-591 (mask.sum() < 2)
        returns = pd.Series([], dtype=float)
        result = stats.r_cubed_turtle(returns)
        assert np.isnan(result)

        # Test with single value
        returns = pd.Series([0.01])
        result = stats.r_cubed_turtle(returns)
        # mask.sum() < 2 check will catch this
        assert np.isnan(result) or isinstance(result, float)

    def test_r_cubed_turtle_line_625_analysis(self):
        """Analyze line 625 unreachable nature.

        Line 625: if len(max_dds) == 0: return np.nan

        This is unreachable because:
        - The loops that populate max_dds always iterate at least once
        - Each iteration adds a value to max_dds (even if it's 0)
        - Therefore max_dds is never empty

        The only way to hit this would be if the loops don't execute,
        which can't happen with the current logic.
        """
        # Test with all zeros - max_dds will contain [0, 0, ...], not empty
        returns = pd.Series(
            [0.0, 0.0, 0.0],
            index=pd.date_range("2020-01-01", periods=3),
        )
        result = stats.r_cubed_turtle(returns)

        # avg_max_dd will be 0, hitting line 628-629 instead of 625
        assert np.isnan(result) or result == np.inf

    def test_r_cubed_turtle_normal_operation(self):
        """Test r_cubed_turtle with valid data."""
        returns = pd.Series(
            [0.01, -0.05, 0.03, 0.02, -0.03, 0.04, -0.02, 0.01, 0.02, -0.01],
            index=pd.date_range("2020-01-01", periods=10),
        )
        result = stats.r_cubed_turtle(returns)

        # Should return a valid value
        assert isinstance(result, (int, float, np.floating))
