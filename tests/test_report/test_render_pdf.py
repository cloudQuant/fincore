from __future__ import annotations

from pathlib import Path

import pandas as pd
from PyPDF2 import PdfReader, PdfWriter

from fincore.report.render_pdf import generate_pdf


class _FakePage:
    def __init__(self) -> None:
        self._section_info = {
            "sections": [
                {"id": "overview", "title": "Overview", "top": 0},
                {"id": "performance", "title": "Performance", "top": 1400},
            ],
            "totalHeight": 2000,
        }

    def goto(self, *_args, **_kwargs) -> None:
        return None

    def evaluate(self, script: str):
        # The implementation calls evaluate twice: once to wait for charts, and once
        # to collect section positions for bookmarks.
        if "querySelectorAll('.sec')" in script:
            return self._section_info
        return None

    def wait_for_timeout(self, *_args, **_kwargs) -> None:
        return None

    def pdf(self, path: str, **_kwargs) -> None:
        # Write a minimal valid PDF so that PyPDF2 can read it and bookmarks can be added.
        writer = PdfWriter()
        writer.add_blank_page(width=612, height=792)
        with open(path, "wb") as f:
            writer.write(f)


class _FakeBrowser:
    def new_page(self, **_kwargs) -> _FakePage:
        return _FakePage()

    def close(self) -> None:
        return None


class _FakeChromium:
    def launch(self, **_kwargs) -> _FakeBrowser:
        return _FakeBrowser()


class _FakePlaywright:
    chromium = _FakeChromium()


class _FakePlaywrightCM:
    def __enter__(self) -> _FakePlaywright:
        return _FakePlaywright()

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


def test_generate_pdf_with_mocked_playwright(tmp_path, monkeypatch) -> None:
    import playwright.sync_api as ps

    monkeypatch.setattr(ps, "sync_playwright", lambda: _FakePlaywrightCM())

    idx = pd.date_range("2024-01-01", periods=80, freq="B", tz="UTC")
    returns = pd.Series([0.001 if (i % 2 == 0) else -0.0006 for i in range(len(idx))], index=idx, name="strategy")

    out = tmp_path / "report.pdf"
    result = generate_pdf(
        returns,
        benchmark_rets=None,
        positions=None,
        transactions=None,
        trades=None,
        title="PDF Unit Test",
        output=str(out),
        rolling_window=20,
    )

    assert result == str(out)
    assert out.exists()

    reader = PdfReader(str(out))
    assert len(reader.pages) == 1


def test_generate_pdf_ignores_cleanup_oserrors(tmp_path, monkeypatch) -> None:
    import os

    import playwright.sync_api as ps

    monkeypatch.setattr(ps, "sync_playwright", lambda: _FakePlaywrightCM())

    # Force OSError during cleanup of tmp_html and tmp_pdf.
    calls = {"n": 0}
    real_remove = os.remove

    def _remove(path: str) -> None:
        calls["n"] += 1
        if calls["n"] <= 2:
            raise OSError("boom")
        real_remove(path)

    monkeypatch.setattr(os, "remove", _remove)

    idx = pd.date_range("2024-01-01", periods=20, freq="B", tz="UTC")
    returns = pd.Series([0.001 if (i % 2 == 0) else -0.0006 for i in range(len(idx))], index=idx, name="strategy")

    out = tmp_path / "cleanup.pdf"
    result = generate_pdf(
        returns,
        benchmark_rets=None,
        positions=None,
        transactions=None,
        trades=None,
        title="Cleanup Errors",
        output=str(out),
        rolling_window=10,
    )

    assert result == str(out)
    assert out.exists()


def test_generate_pdf_raises_when_playwright_missing(tmp_path, monkeypatch) -> None:
    # Make HTML generation fast and deterministic.
    import fincore.report.render_html as rh

    def _stub_generate_html(*_args, output: str, **_kwargs) -> str:
        Path(output).write_text(
            "<html><body><div class='sec'><div class='sec-title'>Overview</div></div></body></html>"
        )
        return output

    monkeypatch.setattr(rh, "generate_html", _stub_generate_html)

    # Force ImportError when importing playwright.sync_api.
    import builtins

    real_import = builtins.__import__

    def _import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith("playwright"):
            raise ImportError("no playwright")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _import)

    idx = pd.date_range("2024-01-01", periods=10, freq="B", tz="UTC")
    returns = pd.Series([0.001] * len(idx), index=idx)

    out = tmp_path / "report.pdf"
    try:
        generate_pdf(
            returns,
            benchmark_rets=None,
            positions=None,
            transactions=None,
            trades=None,
            title="PDF Missing Playwright",
            output=str(out),
            rolling_window=5,
        )
    except ImportError as e:
        assert "Playwright" in str(e) or "playwright" in str(e).lower()
    else:
        raise AssertionError("Expected ImportError")


def test_add_bookmarks_copies_when_pypdf2_missing(tmp_path, monkeypatch) -> None:
    # Hit the ImportError branch in _add_pdf_bookmarks.
    from fincore.report import render_pdf

    in_pdf = tmp_path / "in.pdf"
    out_pdf = tmp_path / "out.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    with in_pdf.open("wb") as f:
        writer.write(f)

    import builtins

    real_import = builtins.__import__

    def _import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "PyPDF2":
            raise ImportError("no pypdf2")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _import)

    render_pdf._add_pdf_bookmarks(str(in_pdf), str(out_pdf), {"sections": []}, "Report")
    assert out_pdf.exists()


def test_add_bookmarks_handles_empty_pdf(tmp_path) -> None:
    from fincore.report import render_pdf

    # Write a valid PDF with zero pages.
    in_pdf = tmp_path / "empty.pdf"
    out_pdf = tmp_path / "empty_out.pdf"
    writer = PdfWriter()
    with in_pdf.open("wb") as f:
        writer.write(f)

    render_pdf._add_pdf_bookmarks(str(in_pdf), str(out_pdf), {"sections": []}, "Report")
    assert out_pdf.exists()
    r = PdfReader(str(out_pdf))
    assert len(r.pages) == 0
