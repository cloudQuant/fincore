"""Tests for interesting times tear sheet.

Tests create_interesting_times_tear_sheet delegation.
Split from test_sheets_delegation.py for maintainability.
"""

from __future__ import annotations

import pandas as pd
import pytest

import fincore.tearsheets.sheets as sheets

from .test_fakes import _FakePyfolioInterestingTimes


def test_create_interesting_times_tear_sheet_warns_and_returns_when_no_overlap(monkeypatch) -> None:
    """Test that interesting times tear sheet warns when no overlap."""
    monkeypatch.setattr(sheets, "print_table", lambda *_args, **_kwargs: None)
    pyf = _FakePyfolioInterestingTimes()
    idx = pd.date_range("2024-01-01", periods=5, freq="B", tz="UTC")
    returns = pd.Series([0.001] * len(idx), index=idx, name="r")
    with pytest.warns(UserWarning):
        sheets.create_interesting_times_tear_sheet(pyf, returns)
