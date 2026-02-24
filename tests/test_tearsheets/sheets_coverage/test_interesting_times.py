"""Tests for interesting times tear sheet.

Tests create_interesting_times_tear_sheet coverage scenarios.
Split from test_sheets_more_coverage.py for maintainability.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import fincore.tearsheets.sheets as sheets

from .test_fixtures import _FakePyfolioInterestingTimesHappy


def test_create_interesting_times_tear_sheet_happy_path_with_benchmark(monkeypatch) -> None:
    """Test interesting times tear sheet with benchmark."""
    monkeypatch.setattr(sheets, "print_table", lambda *_args, **_kwargs: None)
    pyf = _FakePyfolioInterestingTimesHappy()

    idx = pd.date_range("2024-01-01", periods=8, freq="B", tz="UTC")
    returns = pd.Series(np.linspace(0.001, -0.001, len(idx)), index=idx, name="r")
    benchmark = returns * 0.5

    fig = sheets.create_interesting_times_tear_sheet(pyf, returns, benchmark_rets=benchmark, run_flask_app=True)
    assert fig is not None
    assert len(fig.axes) == 2


def test_interesting_times_tear_sheet_without_benchmark_line_678() -> None:
    """Test interesting_times_tear_sheet without benchmark (line 678)."""
    pyf = _FakePyfolioInterestingTimesHappy()

    idx = pd.date_range("2024-01-01", periods=8, freq="B", tz="UTC")
    returns = pd.Series(np.linspace(0.001, -0.001, len(idx)), index=idx, name="r")

    fig = sheets.create_interesting_times_tear_sheet(pyf, returns, benchmark_rets=None, run_flask_app=True)
    assert fig is not None
