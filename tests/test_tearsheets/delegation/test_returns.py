"""Tests for returns tear sheet.

Tests create_returns_tear_sheet delegation.
Split from test_sheets_delegation.py for maintainability.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import fincore.tearsheets.sheets as sheets

from .test_fakes import _FakePyfolioReturns


def test_create_returns_tear_sheet_smoke_with_benchmark_live_and_bootstrap(monkeypatch) -> None:
    """Test returns tear sheet with benchmark, live date, and bootstrap."""
    monkeypatch.setattr(sheets, "clip_returns_to_benchmark", lambda r, b: r.loc[b.index])

    idx = pd.date_range("2024-01-01", periods=10, freq="B", tz="UTC")
    returns = pd.Series(np.linspace(0.001, -0.001, len(idx)), index=idx, name="r")
    benchmark = returns.loc[idx[2:]] * 0.5

    pyf = _FakePyfolioReturns()
    fig = sheets.create_returns_tear_sheet(
        pyf,
        returns,
        benchmark_rets=benchmark,
        live_start_date="2024-01-10",
        bootstrap=True,
        run_flask_app=True,
    )
    assert fig is not None
    assert "show_perf_stats" in pyf.calls
    assert "show_worst_drawdown_periods" in pyf.calls
    assert pyf.calls.count("plot_rolling_returns") == 3
    assert "plot_rolling_beta" in pyf.calls
    assert "plot_perf_stats" in pyf.calls


def test_create_returns_tear_sheet_bootstrap_requires_benchmark() -> None:
    """Test that bootstrap requires benchmark."""
    pyf = _FakePyfolioReturns()
    idx = pd.date_range("2024-01-01", periods=5, freq="B", tz="UTC")
    returns = pd.Series([0.001] * len(idx), index=idx, name="r")
    with pytest.raises(ValueError, match="bootstrap requires"):
        sheets.create_returns_tear_sheet(pyf, returns, benchmark_rets=None, bootstrap=True)
