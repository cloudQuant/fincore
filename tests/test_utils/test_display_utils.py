"""Tests for display utilities in fincore.utils.common_utils.

Split from test_common_display.py for maintainability.
"""

from __future__ import annotations

import contextlib
import io

import pandas as pd
import pytest

from fincore.utils import common_utils as cu


@pytest.mark.p2  # Medium: display utility tests
class TestDisplayFunctions:
    """Test display and fallback functions."""

    def test_fallback_display_prints_to_stdout(self):
        """Test _fallback_display prints to stdout."""
        buf = io.StringIO()
        old = cu.display
        try:
            cu.display = cu._fallback_display
            with contextlib.redirect_stdout(buf):
                cu.display("x", 1)
        finally:
            cu.display = old
        assert "x 1" in buf.getvalue()


@pytest.mark.p2  # Medium: table printing tests
class TestPrintTable:
    """Test print_table function."""

    def test_print_table_injects_header_rows_and_calls_display(self, monkeypatch):
        """Test print_table injects header rows and calls display."""
        captured = {}

        def fake_display(obj):
            captured["obj"] = obj

        monkeypatch.setattr(cu, "display", fake_display)
        monkeypatch.setattr(cu, "HTML", lambda s: s)

        df = pd.DataFrame({"a": [1, 2]}, index=["x", "y"])
        cu.print_table(df, name="T", header_rows={"H": "V"})
        html = captured["obj"]
        assert "<thead>" in html
        assert "H" in html and "V" in html

    def test_print_table_with_series_input(self, monkeypatch):
        """Test print_table converts Series to DataFrame."""
        captured = {}

        def fake_display(obj):
            captured["obj"] = obj

        monkeypatch.setattr(cu, "display", fake_display)
        monkeypatch.setattr(cu, "HTML", lambda s: s)

        s = pd.Series([1, 2], index=["x", "y"])
        cu.print_table(s, name="T")
        html = captured["obj"]
        assert "<table" in html

    def test_print_table_run_flask_app_does_not_implicitly_export(self, monkeypatch):
        """Test Flask-display mode preserves HTML output without an implicit export."""

        def forbidden_export(*_args, **_kwargs):
            pytest.fail("run_flask_app attempted an implicit XLSX export")

        monkeypatch.setattr(pd.DataFrame, "to_excel", forbidden_export, raising=True)
        monkeypatch.setattr(cu, "display", lambda *_args, **_kwargs: None)
        monkeypatch.setattr(cu, "HTML", lambda s: s)

        df = pd.DataFrame({"a": [1, 2]})
        cu.print_table(df, name="X", run_flask_app=True)
