"""Tests for metrics.drawdown module — numerical correctness."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fincore.metrics.drawdown import (
    gen_drawdown_table,
    max_drawdown,
)


@pytest.fixture
def simple_returns():
    """Returns that produce a known drawdown."""
    # Up 10%, down 20%, up 5% → peak at 1.10, trough at 0.88
    return pd.Series(
        [0.10, -0.20, 0.05],
        index=pd.date_range("2020-01-01", periods=3, freq="B"),
    )


@pytest.fixture
def daily_returns():
    np.random.seed(42)
    return pd.Series(
        np.random.randn(252) * 0.01,
        index=pd.date_range("2020-01-01", periods=252, freq="B"),
    )


class TestMaxDrawdown:
    def test_known_drawdown(self, simple_returns):
        mdd = max_drawdown(simple_returns)
        # After +10%, -20%: wealth goes 1.0→1.10→0.88, dd = (0.88-1.10)/1.10 = -0.2
        assert mdd < 0, "max_drawdown should be negative"
        assert abs(mdd - (-0.2)) < 1e-10

    def test_all_positive(self):
        pos = pd.Series([0.01, 0.02, 0.03], index=pd.date_range("2020-01-01", periods=3))
        mdd = max_drawdown(pos)
        assert mdd == 0.0 or np.isclose(mdd, 0.0)

    def test_single_value(self):
        single = pd.Series([0.05], index=pd.date_range("2020-01-01", periods=1))
        mdd = max_drawdown(single)
        assert isinstance(mdd, float)

    def test_daily_returns(self, daily_returns):
        mdd = max_drawdown(daily_returns)
        assert -1 <= mdd <= 0


class TestGenDrawdownTable:
    def test_returns_dataframe(self, daily_returns):
        table = gen_drawdown_table(daily_returns, top=5)
        assert isinstance(table, pd.DataFrame)

    def test_top_parameter(self, daily_returns):
        table = gen_drawdown_table(daily_returns, top=3)
        assert len(table) <= 3

    def test_columns_present(self, daily_returns):
        table = gen_drawdown_table(daily_returns, top=5)
        if len(table) > 0:
            expected_cols = {"Net drawdown in %", "Peak date", "Valley date", "Recovery date"}
            assert expected_cols.issubset(set(table.columns)), f"Missing columns: {expected_cols - set(table.columns)}"
