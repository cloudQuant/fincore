"""Tests for positions tear sheet.

Tests create_position_tear_sheet delegation.
Split from test_sheets_delegation.py for maintainability.
"""

from __future__ import annotations

import pandas as pd
import pytest

import fincore.tearsheets.sheets as sheets

from .test_fakes import _FakePyfolioPositions


def test_create_position_tear_sheet_smoke_with_sector_mappings_and_hide_positions(monkeypatch) -> None:
    """Test positions tear sheet with sector mappings and hide_positions."""
    monkeypatch.setattr(sheets, "check_intraday", lambda *_args, **_kwargs: _args[2])

    idx = pd.date_range("2024-01-01", periods=6, freq="B", tz="UTC")
    returns = pd.Series([0.001] * len(idx), index=idx, name="r")
    positions = pd.DataFrame({"AAA": [10, 10, 0, 0, 5, 0], "cash": [100] * len(idx)}, index=idx)

    pyf = _FakePyfolioPositions()
    fig = sheets.create_position_tear_sheet(
        pyf,
        returns,
        positions,
        hide_positions=True,
        sector_mappings={"AAA": "tech"},
        run_flask_app=True,
    )
    assert fig is not None
    top_calls = [c for c, _ in pyf.calls if c == "show_and_plot_top_positions"]
    assert top_calls
    # hide_positions forces show_and_plot_top_pos=0
    top_kwargs = [kw for c, kw in pyf.calls if c == "show_and_plot_top_positions"][0]
    assert top_kwargs.get("show_and_plot") == 0
    assert any(c == "plot_sector_allocations" for c, _ in pyf.calls)
