"""Tests for transactions tear sheet.

Tests create_txn_tear_sheet delegation.
Split from test_sheets_delegation.py for maintainability.
"""

from __future__ import annotations

import pandas as pd
import pytest

import fincore.tearsheets.sheets as sheets

from .test_fakes import _FakePyfolioTxns


def test_create_txn_tear_sheet_smoke_warns_on_turnover_hist_failure(monkeypatch) -> None:
    """Test transactions tear sheet warns when turnover hist fails."""
    monkeypatch.setattr(sheets, "check_intraday", lambda *_args, **_kwargs: _args[2])

    idx = pd.date_range("2024-01-01", periods=6, freq="B", tz="UTC")
    returns = pd.Series([0.001] * len(idx), index=idx, name="r")
    positions = pd.DataFrame({"AAA": [10] * len(idx), "cash": [100] * len(idx)}, index=idx)
    txns = pd.DataFrame({"amount": [1], "price": [10.0], "symbol": ["AAA"]}, index=[idx[0]])
    unadj = returns.copy()

    pyf = _FakePyfolioTxns()
    with pytest.warns(UserWarning, match="Unable to generate turnover plot"):
        fig = sheets.create_txn_tear_sheet(
            pyf,
            returns,
            positions,
            txns,
            unadjusted_returns=unadj,
            run_flask_app=True,
        )
    assert fig is not None
    assert "plot_turnover" in pyf.calls
    assert "plot_daily_volume" in pyf.calls
    assert "plot_txn_time_hist" in pyf.calls
    assert "plot_slippage_sweep" in pyf.calls
    assert "plot_slippage_sensitivity" in pyf.calls
