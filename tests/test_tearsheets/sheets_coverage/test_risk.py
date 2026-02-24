"""Tests for risk tear sheet.

Tests create_risk_tear_sheet coverage scenarios.
Split from test_sheets_more_coverage.py for maintainability.
"""

from __future__ import annotations

import pandas as pd
import pytest

import fincore.tearsheets.sheets as sheets

from .test_fixtures import _FakePyfolioRisk


def test_create_risk_tear_sheet_handles_optional_panels(monkeypatch) -> None:
    """Test risk tear sheet with optional panels."""
    monkeypatch.setattr(sheets, "check_intraday", lambda *_args, **_kwargs: _args[2])

    idx = pd.date_range("2024-01-01", periods=10, freq="B", tz="UTC")
    positions = pd.DataFrame({"AAA": 1.0, "cash": 0.0}, index=idx)
    sectors = pd.DataFrame({"AAA": ["Tech"] * len(idx)}, index=idx)
    caps = pd.DataFrame({"AAA": [1e9] * len(idx)}, index=idx)
    volumes = pd.DataFrame({"AAA": [1000] * len(idx)}, index=idx)
    style_panel = {"Momentum": pd.DataFrame({"AAA": 0.1}, index=idx)}

    pyf = _FakePyfolioRisk()
    fig = sheets.create_risk_tear_sheet(
        pyf,
        positions=positions,
        style_factor_panel=style_panel,
        sectors=sectors,
        caps=caps,
        volumes=volumes,
        percentile=None,
        returns=None,
        transactions=None,
        run_flask_app=True,
    )
    assert fig is not None
    assert "compute_style_factor_exposures" in pyf.calls
    assert "compute_sector_exposures" in pyf.calls
    assert "compute_cap_exposures" in pyf.calls
    assert "compute_volume_exposures" in pyf.calls

    # Also exercise the path without style_factor_panel (previously crashed due to undefined variables).
    fig2 = sheets.create_risk_tear_sheet(
        pyf,
        positions=positions,
        style_factor_panel=None,
        sectors=sectors,
        caps=None,
        volumes=None,
        returns=None,
        transactions=None,
        run_flask_app=True,
    )
    assert fig2 is not None


def test_risk_tear_sheet_empty_index_warns_and_returns_line_939() -> None:
    """Test risk tear sheet with non-overlapping indices (lines 939-940)."""
    import warnings

    pyf = _FakePyfolioRisk()

    # Create positions and sectors with non-overlapping indices
    idx_pos = pd.date_range("2024-01-01", periods=10, freq="B", tz="UTC")
    idx_sec = pd.date_range("2023-01-01", periods=10, freq="B", tz="UTC")

    positions = pd.DataFrame({"AAA": 1.0, "cash": 0.0}, index=idx_pos)
    sectors = pd.DataFrame({"AAA": ["Tech"] * len(idx_sec)}, index=idx_sec)

    # Should warn and return early
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = sheets.create_risk_tear_sheet(
            pyf,
            positions=positions,
            style_factor_panel=None,
            sectors=sectors,
            caps=None,
            volumes=None,
            returns=None,
            transactions=None,
            run_flask_app=True,
        )
        # Should return None due to warning
        assert result is None
        assert any("No overlapping index" in str(warning.message) for warning in w)


def test_risk_tear_sheet_no_panels_raises_line_968() -> None:
    """Test risk tear sheet with no panels raises ValueError (line 968)."""
    pyf = _FakePyfolioRisk()

    idx = pd.date_range("2024-01-01", periods=10, freq="B", tz="UTC")
    positions = pd.DataFrame({"AAA": 1.0, "cash": 0.0}, index=idx)

    # No style_factor_panel, sectors, caps, or volumes
    with pytest.raises(ValueError, match="requires at least one of"):
        sheets.create_risk_tear_sheet(
            pyf,
            positions=positions,
            style_factor_panel=None,
            sectors=None,
            caps=None,
            volumes=None,
            returns=None,
            transactions=None,
        )
