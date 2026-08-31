"""Branch-completion tests for report.compute (weekly/monthly, positions, trades)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fincore.report.compute import (
    _approx_months,
    _compute_extended_stats,
    _compute_trades,
    _compute_transactions,
    _period_defs,
    _risk_tag,
    compute_sections,
)


def _returns(n: int = 520) -> pd.Series:
    rng = np.random.default_rng(7)
    return pd.Series(rng.normal(0.0, 0.01, n), index=pd.date_range("2020-01-01", periods=n, freq="B"))


def _benchmark(n: int = 520) -> pd.Series:
    rng = np.random.default_rng(8)
    return pd.Series(rng.normal(0.0, 0.008, n), index=pd.date_range("2020-01-01", periods=n, freq="B"))


def _positions(n: int = 520) -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {
            "AAPL": np.random.default_rng(9).uniform(0.1, 0.5, n),
            "MSFT": np.random.default_rng(10).uniform(-0.3, 0.4, n),
            "cash": np.random.default_rng(11).uniform(0.1, 0.5, n),
        },
        index=idx,
    )


def _transactions() -> pd.DataFrame:
    idx = pd.to_datetime(["2020-01-02 10:00", "2020-01-02 14:30", "2020-01-03 09:45"])
    return pd.DataFrame(
        {
            "symbol": ["AAPL", "MSFT", "AAPL"],
            "amount": [100.0, -50.0, 30.0],
            "price": [150.0, 200.0, 152.0],
        },
        index=idx,
    )


def _trades() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "pnlcomm": [10.0, -5.0, 20.0, -2.0],
            "commission": [1.0, 1.0, 1.5, 1.0],
            "long": [1, 1, 0, 0],
            "barlen": [3, 5, 2, 4],
        }
    )


# ---------------------------------------------------------------------------
# period helpers
# ---------------------------------------------------------------------------


def test_approx_months_weekly() -> None:
    assert _approx_months(52, "weekly") == 12


def test_approx_months_monthly() -> None:
    assert _approx_months(12, "monthly") == 12


def test_approx_months_daily() -> None:
    assert _approx_months(21, "daily") == 1


def test_period_defs_weekly() -> None:
    defs = _period_defs("weekly")
    assert defs[0] == ("1W", 1)


def test_period_defs_monthly() -> None:
    defs = _period_defs("monthly")
    assert defs[0] == ("1M", 1)


def test_risk_tag_nan() -> None:
    assert _risk_tag(np.nan) == "N/A"


# ---------------------------------------------------------------------------
# _compute_extended_stats weekly/monthly
# ---------------------------------------------------------------------------


def test_compute_extended_stats_weekly() -> None:
    ext = _compute_extended_stats(_returns(), "weekly")
    assert "Max Drawdown Weeks" in ext


def test_compute_extended_stats_monthly() -> None:
    ext = _compute_extended_stats(_returns(), "monthly")
    assert "Max Drawdown Months" in ext


# ---------------------------------------------------------------------------
# _compute_transactions
# ---------------------------------------------------------------------------


def test_compute_transactions_with_positions() -> None:
    result = _compute_transactions(_transactions(), _positions())
    assert result["has_transactions"] is True
    assert "txn_hours" in result
    assert "Unique Symbols Traded" in result["txn_summary"]


# ---------------------------------------------------------------------------
# _compute_trades full branches
# ---------------------------------------------------------------------------


def test_compute_trades_full() -> None:
    result = _compute_trades(_trades())
    assert result["trade_stats"]["Total Trades"] == 4
    assert "Total Commission" in result["trade_stats"]
    assert "Long Win Rate" in result["trade_stats"]
    assert "Short Win Rate" in result["trade_stats"]
    assert "Avg Holding Bars" in result["trade_stats"]
    assert "trade_pnl_long" in result
    assert "trade_barlen" in result


# ---------------------------------------------------------------------------
# compute_sections with full inputs (weekly + positions + transactions + trades)
# ---------------------------------------------------------------------------


def test_compute_sections_weekly_with_everything() -> None:
    sections = compute_sections(
        _returns(),
        _benchmark(),
        _positions(),
        _transactions(),
        _trades(),
        52,
        period="weekly",
    )
    assert isinstance(sections, dict) or sections is not None
    assert "period_returns" in sections
    assert "trade_stats" in sections
    assert "benchmark_period_returns" in sections


def test_compute_sections_monthly_with_benchmark() -> None:
    sections = compute_sections(
        _returns(),
        _benchmark(),
        None,
        None,
        None,
        12,
        period="monthly",
    )
    assert isinstance(sections, dict) or sections is not None
    assert "benchmark_period_returns" in sections


def test_compute_transactions_without_symbol() -> None:
    idx = pd.to_datetime(["2020-01-02 10:00", "2020-01-03 11:00"])
    transactions = pd.DataFrame({"amount": [10.0, -5.0], "price": [150.0, 152.0]}, index=idx)
    result = _compute_transactions(transactions, None)
    assert "Unique Symbols Traded" not in result["txn_summary"]


def test_compute_transactions_with_turnover_provided() -> None:
    turnover = pd.Series([0.1, 0.2], index=pd.date_range("2020-01-01", periods=2))
    result = _compute_transactions(_transactions(), None, turnover=turnover)
    assert "turnover" in result


def test_compute_trades_minimal_columns() -> None:
    trades = pd.DataFrame({"pnlcomm": [10.0, -5.0]})
    result = _compute_trades(trades)
    assert "Total Commission" not in result["trade_stats"]
    assert "Long Trades" not in result["trade_stats"]
    assert "Avg Holding Bars" not in result["trade_stats"]
    assert "trade_pnl_long" not in result
    assert "trade_barlen" not in result


def test_compute_trades_all_long() -> None:
    trades = pd.DataFrame({"pnlcomm": [10.0, -5.0, 20.0], "long": [1, 1, 1]})
    result = _compute_trades(trades)
    assert "Short Win Rate" not in result["trade_stats"]
