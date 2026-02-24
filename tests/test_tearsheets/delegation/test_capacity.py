"""Tests for capacity tear sheet.

Tests create_capacity_tear_sheet delegation.
Split from test_sheets_delegation.py for maintainability.
"""

from __future__ import annotations

import pandas as pd
import pytest

import fincore.tearsheets.sheets as sheets

from .test_fakes import _FakePyfolioCapacity


def test_create_capacity_tear_sheet_runs_with_stubbed_pyfolio(monkeypatch) -> None:
    """Test that capacity tear sheet runs with fake pyfolio."""
    monkeypatch.setattr(sheets, "check_intraday", lambda *_args, **_kwargs: _args[2])
    monkeypatch.setattr(sheets, "print_table", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(sheets, "format_asset", lambda x: str(x))

    idx = pd.date_range("2024-01-01", periods=6, freq="B", tz="UTC")
    returns = pd.Series([0.001, -0.001, 0.002, -0.002, 0.001, 0.0], index=idx, name="r")
    positions = pd.DataFrame(
        {"AAA": [100, 100, 100, 100, 100, 100], "cash": [1_000, 1_000, 1_000, 1_000, 1_000, 1_000]},
        index=idx,
    )
    transactions = pd.DataFrame({"amount": [1], "price": [10.0], "symbol": ["AAA"]}, index=[idx[1]])
    market_data = {
        "price": pd.DataFrame({"AAA": [10.0]}, index=[idx[1]]),
        "volume": pd.DataFrame({"AAA": [100]}, index=[idx[1]]),
    }

    pyf = _FakePyfolioCapacity()
    sheets.create_capacity_tear_sheet(pyf, returns, positions, transactions, market_data)
    assert "get_max_days_to_liquidate_by_ticker" in pyf.calls
    assert "get_low_liquidity_transactions" in pyf.calls
    assert "plot_capacity_sweep" in pyf.calls
