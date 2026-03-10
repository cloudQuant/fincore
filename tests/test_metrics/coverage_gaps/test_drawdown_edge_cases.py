"""Tests targeting specific uncovered lines in drawdown.py.

Part of test_coverage_gaps.py split - Drawdown module edge cases.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fincore.metrics import drawdown as dd


@pytest.fixture
def daily_returns():
    """Generate daily returns for testing."""
    rng = np.random.RandomState(42)
    r = rng.normal(0.0005, 0.01, 300)
    idx = pd.bdate_range("2020-01-01", periods=300)
    return pd.Series(r, index=idx)


@pytest.mark.p1
class TestMaxDrawdownEdge:
    """Tests for max_drawdown edge cases."""

    def test_empty_returns_nan(self):
        """Cover lines 81-85: len(returns) < 1."""
        r = np.array([], dtype=float)
        assert np.isnan(dd.max_drawdown(r))

    def test_dataframe_returns_series(self):
        """Cover line 102-103: DataFrame branch."""
        idx = pd.bdate_range("2020-01-01", periods=50)
        df = pd.DataFrame(
            {"a": np.random.default_rng(0).normal(0, 0.01, 50), "b": np.random.default_rng(1).normal(0, 0.01, 50)},
            index=idx,
        )
        result = dd.max_drawdown(df)
        assert isinstance(result, pd.Series)


@pytest.mark.p2
class TestMaxDrawdownDays:
    """Tests for max_drawdown_days function."""

    def test_empty_returns_nan(self):
        """Cover line 435."""
        assert np.isnan(dd.max_drawdown_days(pd.Series([], dtype=float)))

    def test_ndarray_input(self):
        """Cover line 438: non-Series conversion."""
        r = np.array([0.01, -0.02, 0.03, -0.01, -0.05, 0.02])
        result = dd.max_drawdown_days(r)
        assert isinstance(result, (int, np.integer))

    def test_datetime_index(self, daily_returns):
        """Cover line 448: DatetimeIndex path."""
        result = dd.max_drawdown_days(daily_returns)
        assert isinstance(result, (int, np.integer))


@pytest.mark.p2
class TestMaxDrawdownWeeks:
    """Tests for max_drawdown_weeks function."""

    def test_empty_returns_nan(self):
        """Cover lines 469-472."""
        assert np.isnan(dd.max_drawdown_weeks(pd.Series([], dtype=float)))

    def test_normal(self, daily_returns):
        result = dd.max_drawdown_weeks(daily_returns)
        assert isinstance(result, float)


@pytest.mark.p2
class TestMaxDrawdownMonths:
    """Tests for max_drawdown_months function."""

    def test_empty_returns_nan(self):
        """Cover lines 489-492."""
        assert np.isnan(dd.max_drawdown_months(pd.Series([], dtype=float)))


@pytest.mark.p2
class TestMaxDrawdownRecoveryDays:
    """Tests for max_drawdown_recovery_days function."""

    def test_empty_returns_nan(self):
        """Cover lines 513-514."""
        assert np.isnan(dd.max_drawdown_recovery_days(pd.Series([], dtype=float)))

    def test_no_recovery(self):
        """Cover line 536: recovery doesn't happen."""
        r = pd.Series([-0.10, -0.10, -0.10, -0.10], index=pd.bdate_range("2020-01-01", periods=4))
        result = dd.max_drawdown_recovery_days(r)
        assert np.isnan(result)

    def test_with_recovery(self):
        """Cover lines 529-534."""
        r = pd.Series([0.10, -0.05, -0.03, 0.15, 0.10], index=pd.bdate_range("2020-01-01", periods=5))
        result = dd.max_drawdown_recovery_days(r)
        assert isinstance(result, (int, np.integer))

    def test_integer_index_recovery(self):
        """Cover line 534: non-DatetimeIndex branch."""
        r = pd.Series([0.10, -0.05, -0.03, 0.15, 0.10])
        result = dd.max_drawdown_recovery_days(r)
        assert isinstance(result, (int, np.integer))


@pytest.mark.p2
class TestMaxDrawdownRecoveryWeeks:
    """Tests for max_drawdown_recovery_weeks function."""

    def test_empty_returns_nan(self):
        """Cover lines 553-556."""
        assert np.isnan(dd.max_drawdown_recovery_weeks(pd.Series([], dtype=float)))


@pytest.mark.p2
class TestMaxDrawdownRecoveryMonths:
    """Tests for max_drawdown_recovery_months function."""

    def test_empty_returns_nan(self):
        """Cover lines 573-576."""
        assert np.isnan(dd.max_drawdown_recovery_months(pd.Series([], dtype=float)))


@pytest.mark.p2
class TestSecondMaxDrawdown:
    """Tests for second_max_drawdown function."""

    def test_normal(self, daily_returns):
        """Cover lines 601-602, 627-628."""
        result = dd.second_max_drawdown(daily_returns)
        assert isinstance(result, float)
        assert result <= 0


@pytest.mark.p2
class TestDrawdownDetailedAndTopN:
    """Tests for detailed drawdown analysis functions."""

    def test_identify_drawdown_periods_empty(self):
        """Cover line 132."""
        result = dd._identify_drawdown_periods(pd.Series([], dtype=float))
        assert result is None

    def test_get_top_drawdowns(self, daily_returns):
        """Cover line 673 via get_top_drawdowns."""
        result = dd.get_top_drawdowns(daily_returns, top=3)
        assert isinstance(result, list)
        assert len(result) <= 3

    def test_gen_drawdown_table(self, daily_returns):
        """Cover line 722 via gen_drawdown_table."""
        result = dd.gen_drawdown_table(daily_returns, top=3)
        assert isinstance(result, pd.DataFrame)
