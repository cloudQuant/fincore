"""Tests targeting specific uncovered lines in data_utils.py.

Part of test_coverage_gaps.py split - Data utilities edge cases.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fincore.utils.data_utils import roll, rolling_window, up, down


@pytest.fixture
def daily_returns():
    """Generate daily returns for testing."""
    rng = np.random.RandomState(42)
    r = rng.normal(0.0005, 0.01, 300)
    idx = pd.bdate_range("2020-01-01", periods=300)
    return pd.Series(r, index=idx)


@pytest.mark.p2
class TestRollingWindow:
    """Tests for rolling_window utility function."""

    def test_2d_raises(self):
        """Cover line 47."""
        with pytest.raises(ValueError, match="1D"):
            rolling_window(np.ones((3, 3)), 2)

    def test_window_too_large_raises(self):
        """Cover line 50."""
        with pytest.raises(ValueError, match="greater"):
            rolling_window(np.array([1, 2, 3]), 5)

    def test_normal(self):
        """Cover lines 44-56."""
        result = rolling_window(np.arange(5), 3)
        assert result.shape == (3, 3)


@pytest.mark.p2
class TestRollPandasSingleArg:
    """Tests for roll function with pandas input."""

    def test_single_arg_path(self, daily_returns):
        """Cover lines 74-76: single-arg _roll_pandas path."""
        r = daily_returns[:30]
        result = roll(r, function=np.mean, window=10)
        assert isinstance(result, pd.Series)
        assert len(result) == 21


@pytest.mark.p2
class TestRollNdarraySingleArg:
    """Tests for roll function with ndarray input."""

    def test_single_arg_path(self):
        """Cover lines 99-101: single-arg _roll_ndarray path."""
        r = np.random.default_rng(0).normal(0, 0.01, 30)
        result = roll(r, function=np.mean, window=10)
        assert isinstance(result, np.ndarray)
        assert len(result) == 21

    def test_short_ndarray(self):
        """Cover line 94."""
        r = np.array([0.01])
        result = roll(r, function=np.mean, window=10)
        assert isinstance(result, np.ndarray)
        assert len(result) == 0


@pytest.mark.p2
class TestRollValidation:
    """Tests for roll function validation."""

    def test_too_many_args_raises(self, daily_returns):
        """Cover line 140."""
        r = daily_returns
        with pytest.raises(ValueError, match="more than 2"):
            roll(r, r, r, function=np.mean, window=5)

    def test_mismatched_types_raises(self, daily_returns):
        """Cover lines 142-144."""
        r = daily_returns[:10]
        n = np.random.default_rng(0).normal(0, 0.01, 10)
        with pytest.raises(ValueError, match="not the same"):
            roll(r, n, function=np.mean, window=5)


def _sum_two(returns, factor_returns):
    """Helper: sum of returns (ignores factor_returns)."""
    return float(np.sum(returns))


@pytest.mark.p2
class TestUpDown:
    """Tests for up and down filtering functions."""

    def test_up_filters_positive(self):
        r = pd.Series([0.01, -0.02, 0.03])
        f = pd.Series([0.01, -0.01, 0.02])
        result = up(r, f, function=_sum_two)
        assert isinstance(result, float)

    def test_down_filters_negative(self):
        r = pd.Series([0.01, -0.02, 0.03])
        f = pd.Series([0.01, -0.01, 0.02])
        result = down(r, f, function=_sum_two)
        assert isinstance(result, float)
