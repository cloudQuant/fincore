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

    def test_print_table_run_flask_app_uses_temp_static_dir(self, monkeypatch, tmp_path):
        """Test print_table with run_flask_app uses temp static dir."""
        captured = {"excel_path": None}

        def fake_to_excel(self, path, index=True):  # noqa: ARG001
            captured["excel_path"] = str(path)

        monkeypatch.setattr(pd.DataFrame, "to_excel", fake_to_excel, raising=True)
        monkeypatch.setattr(cu, "__file__", str(tmp_path / "common_utils.py"), raising=False)
        monkeypatch.setattr(cu, "display", lambda *_args, **_kwargs: None)
        monkeypatch.setattr(cu, "HTML", lambda s: s)

        df = pd.DataFrame({"a": [1, 2]})
        cu.print_table(df, name="X", run_flask_app=True)
        assert captured["excel_path"] is not None
        assert str(tmp_path / "static") in captured["excel_path"]

    def test_print_table_run_flask_app_logs_warning_when_to_excel_fails(
        self, monkeypatch, tmp_path
    ):
        """Test print_table logs warning when to_excel fails."""
        def fake_to_excel(self, path, index=True):  # noqa: ARG001
            raise RuntimeError("boom")

        monkeypatch.setattr(pd.DataFrame, "to_excel", fake_to_excel, raising=True)
        monkeypatch.setattr(cu, "__file__", str(tmp_path / "common_utils.py"), raising=False)
        monkeypatch.setattr(cu, "display", lambda *_args, **_kwargs: None)
        monkeypatch.setattr(cu, "HTML", lambda s: s)

        df = pd.DataFrame({"a": [1, 2]})
        cu.print_table(df, name="X", run_flask_app=True)
