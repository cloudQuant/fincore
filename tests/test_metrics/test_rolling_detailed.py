"""Tests for metrics.rolling module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fincore.metrics.rolling import (
    rolling_beta,
    rolling_sharpe,
    rolling_volatility,
)


@pytest.fixture
def daily_returns():
    np.random.seed(42)
    return pd.Series(
        np.random.randn(252) * 0.01,
        index=pd.date_range("2020-01-01", periods=252, freq="B"),
    )


@pytest.fixture
def benchmark_returns():
    np.random.seed(123)
    return pd.Series(
        np.random.randn(252) * 0.008,
        index=pd.date_range("2020-01-01", periods=252, freq="B"),
    )


class TestRollingSharpe:
    def test_returns_series(self, daily_returns):
        rs = rolling_sharpe(daily_returns, 63)
        assert isinstance(rs, pd.Series)
        assert len(rs) == len(daily_returns)

    def test_window_parameter(self, daily_returns):
        rs = rolling_sharpe(daily_returns, 126)
        assert isinstance(rs, pd.Series)

    def test_short_data(self):
        short = pd.Series([0.01, -0.02], index=pd.date_range("2020-01-01", periods=2))
        rs = rolling_sharpe(short, 2)
        assert isinstance(rs, pd.Series)


class TestRollingVolatility:
    def test_returns_series(self, daily_returns):
        rv = rolling_volatility(daily_returns, 63)
        assert isinstance(rv, pd.Series)
        assert len(rv) == len(daily_returns)

    def test_values_non_negative(self, daily_returns):
        rv = rolling_volatility(daily_returns, 63)
        valid = rv.dropna()
        assert (valid >= 0).all()


class TestRollingBeta:
    def test_returns_series(self, daily_returns, benchmark_returns):
        rb = rolling_beta(daily_returns, benchmark_returns)
        assert isinstance(rb, pd.Series)

    def test_short_data(self):
        short_r = pd.Series([0.01, -0.02, 0.005], index=pd.date_range("2020-01-01", periods=3))
        short_b = pd.Series([0.005, -0.01, 0.003], index=pd.date_range("2020-01-01", periods=3))
        rb = rolling_beta(short_r, short_b)
        assert isinstance(rb, pd.Series)
