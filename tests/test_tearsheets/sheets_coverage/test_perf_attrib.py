"""Tests for performance attribution tear sheet.

Tests create_perf_attrib_tear_sheet coverage scenarios.
Split from test_sheets_more_coverage.py for maintainability.
"""

from __future__ import annotations

import pandas as pd
import pytest

import fincore.tearsheets.sheets as sheets

from .test_fixtures import _FakePyfolioPerfAttrib


def test_create_perf_attrib_tear_sheet_with_and_without_partitions(monkeypatch) -> None:
    """Test perf attrib tear sheet with and without factor partitions."""
    monkeypatch.setattr(sheets, "display", lambda *_args, **_kwargs: None)

    idx = pd.date_range("2024-01-01", periods=10, freq="B", tz="UTC")
    returns = pd.Series(0.001, index=idx, name="r")
    positions = pd.DataFrame({"AAA": 1.0, "cash": 0.0}, index=idx)
    factor_returns = pd.DataFrame({"MKT": 0.0, "SMB": 0.0, "TECH": 0.0}, index=idx)
    factor_loadings = pd.DataFrame({"AAA": [1.0, 0.5, 0.2]}, index=["MKT", "SMB", "TECH"])

    pyf = _FakePyfolioPerfAttrib()
    fig = sheets.create_perf_attrib_tear_sheet(
        pyf,
        returns,
        positions,
        factor_returns,
        factor_loadings,
        run_flask_app=True,
        factor_partitions={"style": ["MKT", "SMB"], "industry": ["TECH"]},
    )
    assert fig is not None

    fig2 = sheets.create_perf_attrib_tear_sheet(
        pyf,
        returns,
        positions,
        factor_returns,
        factor_loadings,
        run_flask_app=True,
        factor_partitions=None,
    )
    assert fig2 is not None
    assert "perf_attrib" in pyf.calls
