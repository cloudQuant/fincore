"""Tests for full and simple tear sheet creation.

Tests create_full_tear_sheet and create_simple_tear_sheet delegation.
Split from test_sheets_delegation.py for maintainability.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import fincore.tearsheets.sheets as sheets

from .test_fakes import _FakePyfolioFull, _FakePyfolioSimple


def test_create_full_tear_sheet_delegates_through_optional_sections(monkeypatch) -> None:
    """Test that full tear sheet delegates through all optional sections."""
    # Avoid depending on the real intraday/position heuristics.
    monkeypatch.setattr(sheets, "check_intraday", lambda *_args, **_kwargs: _args[2])

    idx = pd.date_range("2024-01-01", periods=5, freq="B", tz="UTC")
    returns = pd.Series(np.linspace(0.001, -0.001, len(idx)), index=idx, name="r")
    positions = pd.DataFrame(
        {"AAA": [10, 12, 11, 13, 10], "cash": [100, 100, 100, 100, 100]},
        index=idx,
    )
    transactions = pd.DataFrame({"amount": [1], "price": [10.0], "symbol": ["AAA"]}, index=[idx[0]])
    market_data = {
        "price": pd.DataFrame({"AAA": [10.0]}, index=[idx[0]]),
        "volume": pd.DataFrame({"AAA": [100]}, index=[idx[0]]),
    }

    pyf = _FakePyfolioFull()
    sheets.create_full_tear_sheet(
        pyf,
        returns,
        positions=positions,
        transactions=transactions,
        market_data=market_data,
        slippage=0.01,
        round_trips=True,
        bayesian=True,
        style_factor_panel=object(),
        factor_returns=object(),
        factor_loadings=object(),
    )

    called = [name for name, _ in pyf.calls]
    assert "adjust_returns_for_slippage" in called
    assert "create_returns_tear_sheet" in called
    assert "create_interesting_times_tear_sheet" in called
    assert "create_position_tear_sheet" in called
    assert "create_txn_tear_sheet" in called
    assert "create_round_trip_tear_sheet" in called
    assert "create_capacity_tear_sheet" in called
    assert "create_risk_tear_sheet" in called
    assert "create_perf_attrib_tear_sheet" in called
    assert "create_bayesian_tear_sheet" in called
    assert pyf.unadjusted_returns is not None


def test_create_simple_tear_sheet_runs_without_real_pyfolio(monkeypatch) -> None:
    """Test that simple tear sheet runs with fake pyfolio."""
    monkeypatch.setattr(sheets, "check_intraday", lambda *_args, **_kwargs: _args[2])

    idx = pd.date_range("2024-01-01", periods=6, freq="B", tz="UTC")
    returns = pd.Series([0.001, -0.001, 0.002, -0.002, 0.001, 0.0], index=idx, name="r")
    benchmark = returns * 0.5
    positions = pd.DataFrame(
        {"AAA": [10, 10, 11, 11, 12, 12], "cash": [100, 100, 100, 100, 100, 100]},
        index=idx,
    )
    transactions = pd.DataFrame({"amount": [1], "price": [10.0], "symbol": ["AAA"]}, index=[idx[1]])

    pyf = _FakePyfolioSimple()
    sheets.create_simple_tear_sheet(
        pyf,
        returns,
        positions=positions,
        transactions=transactions,
        benchmark_rets=benchmark,
        slippage=0.01,
    )

    assert "show_perf_stats" in pyf.calls
    assert "plot_rolling_returns" in pyf.calls
    assert "plot_rolling_beta" in pyf.calls
    assert "plot_rolling_sharpe" in pyf.calls
    assert "plot_drawdown_underwater" in pyf.calls
    assert "get_percent_alloc" in pyf.calls
    assert "plot_turnover" in pyf.calls
    assert "plot_txn_time_hist" in pyf.calls
