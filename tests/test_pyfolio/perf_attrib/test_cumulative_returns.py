"""Tests for cumulative returns calculation with costs.

Tests _cumulative_returns_less_costs function.
Split from test_perf_attrib.py for maintainability.
"""

from __future__ import annotations

import unittest

import pandas as pd

from .conftest import _cumulative_returns_less_costs


class TestCumulativeReturnsWithCosts(unittest.TestCase):
    """Test cumulative returns less costs calculation."""

    def test_cumulative_returns_less_costs(self):
        """Test _cumulative_returns_less_costs with and without costs."""
        returns = pd.Series([0.1] * 3, index=pd.date_range("2017-01-01", periods=3))
        cost = pd.Series([0.001] * len(returns), index=returns.index)

        expected_returns = pd.Series([0.1, 0.21, 0.331], index=returns.index)
        pd.testing.assert_series_equal(expected_returns, _cumulative_returns_less_costs(returns, None))

        expected_returns = pd.Series([0.099000, 0.207801, 0.327373], index=returns.index)
        pd.testing.assert_series_equal(expected_returns, _cumulative_returns_less_costs(returns, cost))
