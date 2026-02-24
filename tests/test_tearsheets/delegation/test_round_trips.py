"""Tests for round-trip tear sheet.

Tests create_round_trip_tear_sheet delegation.
Split from test_sheets_delegation.py for maintainability.
"""

from __future__ import annotations

import pandas as pd
import pytest

import fincore.tearsheets.sheets as sheets

from .test_fakes import _FakePyfolioRoundTrips


def test_create_round_trip_tear_sheet_warns_and_returns_when_too_few_trades(monkeypatch) -> None:
    """Test round-trip tear sheet warns when too few trades."""
    monkeypatch.setattr(sheets, "check_intraday", lambda *_args, **_kwargs: _args[2])
    from fincore.empyrical import Empyrical

    monkeypatch.setattr(Empyrical, "add_closing_transactions", staticmethod(lambda _p, t: t))
    monkeypatch.setattr(
        Empyrical, "extract_round_trips", staticmethod(lambda *_a, **_k: pd.DataFrame({"pnl": [1, 2, 3]}))
    )

    idx = pd.date_range("2024-01-01", periods=6, freq="B", tz="UTC")
    returns = pd.Series([0.001] * len(idx), index=idx, name="r")
    positions = pd.DataFrame({"AAA": [10] * len(idx), "cash": [100] * len(idx)}, index=idx)
    txns = pd.DataFrame({"amount": [1], "price": [10.0], "symbol": ["AAA"]}, index=[idx[0]])

    pyf = _FakePyfolioRoundTrips()
    with pytest.warns(UserWarning, match="Fewer than 5 round-trip"):
        out = sheets.create_round_trip_tear_sheet(pyf, returns, positions, txns)
    assert out is None


def test_create_round_trip_tear_sheet_smoke_with_sector_mappings(monkeypatch) -> None:
    """Test round-trip tear sheet with sector mappings."""
    # Import Empyrical after check_intraday setup to avoid early import issues
    monkeypatch.setattr(sheets, "check_intraday", lambda *_args, **_kwargs: _args[2])

    # Create a mock Empyrical class with the required methods
    class MockEmpyrical:
        @staticmethod
        def add_closing_transactions(positions, transactions):
            return transactions

        @staticmethod
        def extract_round_trips(transactions, portfolio_value=None):
            return pd.DataFrame(
                {
                    "duration": pd.to_timedelta([1, 2, 3, 4, 5], unit="D"),
                    "pnl": [1.0, -1.0, 2.0, -2.0, 0.5],
                    "returns": pd.Series([0.01, -0.01, 0.02, -0.02, 0.005]),
                }
            )

        @staticmethod
        def apply_sector_mappings_to_round_trips(trades, sector_mappings):
            return trades

    # Replace sheets.Empyrical with our mock
    monkeypatch.setattr(sheets, "Empyrical", MockEmpyrical)

    idx = pd.date_range("2024-01-01", periods=6, freq="B", tz="UTC")
    returns = pd.Series([0.001] * len(idx), index=idx, name="r")
    positions = pd.DataFrame({"AAA": [10] * len(idx), "cash": [100] * len(idx)}, index=idx)
    txns = pd.DataFrame({"amount": [1], "price": [10.0], "symbol": ["AAA"]}, index=[idx[0]])

    pyf = _FakePyfolioRoundTrips()
    fig = sheets.create_round_trip_tear_sheet(
        pyf,
        returns,
        positions,
        txns,
        sector_mappings={"AAA": "tech"},
        run_flask_app=True,
    )
    assert fig is not None
    assert "print_round_trip_stats" in pyf.calls
    assert pyf.calls.count("show_profit_attribution") == 2
    assert "plot_round_trip_lifetimes" in pyf.calls
    assert "plot_prob_profit_trade" in pyf.calls
