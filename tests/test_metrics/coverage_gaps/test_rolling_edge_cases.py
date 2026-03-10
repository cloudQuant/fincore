"""Tests targeting specific uncovered lines in rolling.py.

Part of test_coverage_gaps.py split - Rolling module edge cases.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fincore.metrics import rolling as rl


@pytest.fixture
def daily_returns():
    """Generate daily returns for testing."""
    rng = np.random.RandomState(42)
    r = rng.normal(0.0005, 0.01, 300)
    idx = pd.bdate_range("2020-01-01", periods=300)
    return pd.Series(r, index=idx)


@pytest.fixture
def factor_returns():
    """Generate factor returns for testing."""
    rng = np.random.RandomState(99)
    r = rng.normal(0.0003, 0.008, 300)
    idx = pd.bdate_range("2020-01-01", periods=300)
    return pd.Series(r, index=idx)


@pytest.mark.p2
class TestRollAlphaBeta:
    """Tests for roll_alpha_beta function."""

    def test_ndarray_input(self):
        """Cover lines 180-198: non-Series input path."""
        r = np.random.default_rng(0).normal(0, 0.01, 50)
        f = np.random.default_rng(1).normal(0, 0.01, 50)
        result = rl.roll_alpha_beta(r, f, window=10)
        assert isinstance(result, pd.DataFrame)
        assert "alpha" in result.columns
        assert "beta" in result.columns

    def test_short_series_returns_empty_df(self):
        """Cover lines 173-178."""
        r = pd.Series([0.01, 0.02], index=pd.bdate_range("2020-01-01", periods=2))
        f = pd.Series([0.01, 0.02], index=pd.bdate_range("2020-01-01", periods=2))
        result = rl.roll_alpha_beta(r, f, window=100)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0


@pytest.mark.p2
class TestRollSharpeRatio:
    """Tests for roll_sharpe_ratio function."""

    def test_ndarray_input(self):
        """Cover lines 222-249: ndarray path."""
        r = np.random.default_rng(0).normal(0, 0.01, 50)
        result = rl.roll_sharpe_ratio(r, window=10)
        assert isinstance(result, np.ndarray)
        assert len(result) == 41

    def test_short_ndarray(self):
        """Cover line 229."""
        r = np.array([0.01, 0.02])
        result = rl.roll_sharpe_ratio(r, window=100)
        assert isinstance(result, np.ndarray)
        assert len(result) == 0

    def test_short_series_datetime(self):
        """Cover lines 224-228."""
        r = pd.Series([0.01], index=pd.bdate_range("2020-01-01", periods=1))
        result = rl.roll_sharpe_ratio(r, window=100)
        assert isinstance(result, pd.Series)
        assert len(result) == 0


@pytest.mark.p2
class TestRollMaxDrawdown:
    """Tests for roll_max_drawdown function."""

    def test_ndarray_input(self):
        """Cover lines 267-303: ndarray path."""
        r = np.random.default_rng(0).normal(0, 0.01, 50)
        result = rl.roll_max_drawdown(r, window=10)
        assert isinstance(result, np.ndarray)
        assert len(result) == 41

    def test_short_ndarray(self):
        """Cover line 274."""
        r = np.array([0.01])
        result = rl.roll_max_drawdown(r, window=100)
        assert isinstance(result, np.ndarray)
        assert len(result) == 0


@pytest.mark.p2
class TestRollUpCapture:
    """Tests for roll_up_capture function."""

    def test_ndarray_input(self):
        """Cover lines 323-346: ndarray path."""
        r = np.random.default_rng(0).normal(0, 0.01, 20)
        f = np.random.default_rng(1).normal(0, 0.01, 20)
        result = rl.roll_up_capture(r, f, window=5)
        assert isinstance(result, np.ndarray)

    def test_short_ndarray(self):
        """Cover line 332."""
        r = np.array([0.01])
        f = np.array([0.02])
        result = rl.roll_up_capture(r, f, window=100)
        assert isinstance(result, np.ndarray)
        assert len(result) == 0


@pytest.mark.p2
class TestRollDownCapture:
    """Tests for roll_down_capture function."""

    def test_ndarray_input(self):
        """Cover lines 366-389: ndarray path."""
        r = np.random.default_rng(0).normal(0, 0.01, 20)
        f = np.random.default_rng(1).normal(0, 0.01, 20)
        result = rl.roll_down_capture(r, f, window=5)
        assert isinstance(result, np.ndarray)


@pytest.mark.p2
class TestRollUpDownCapture:
    """Tests for roll_up_down_capture function."""

    def test_basic(self, daily_returns, factor_returns):
        """Cover lines 409-413."""
        r = daily_returns[:30]
        f = factor_returns[:30]
        result = rl.roll_up_down_capture(r, f, window=10)
        assert isinstance(result, pd.Series)


@pytest.mark.p2
class TestRollingRegression:
    """Tests for rolling_regression function."""

    def test_short_input(self):
        """Cover line 522."""
        r = pd.Series([0.01, 0.02])
        f = pd.Series([0.01, 0.02])
        result = rl.rolling_regression(r, f, rolling_window=100)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_ndarray_input(self):
        """Cover lines 524-526."""
        r = np.random.default_rng(0).normal(0, 0.01, 50)
        f = np.random.default_rng(1).normal(0, 0.01, 50)
        result = rl.rolling_regression(r, f, rolling_window=10)
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
