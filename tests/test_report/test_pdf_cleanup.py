"""Task 8 PDF temp-file lifecycle contract.

Every temporary file created during PDF generation (intermediate HTML,
pre-bookmark PDF) lives inside a ``TemporaryDirectory`` and is removed on
all paths: success, playwright failure, and HTML-generation failure.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from PyPDF2 import PdfWriter

from fincore.exceptions import InputContractError
from fincore.report.render_pdf import generate_pdf


def _returns(n: int = 40) -> pd.Series:
    idx = pd.date_range("2024-01-01", periods=n, freq="B", tz="UTC")
    return pd.Series(
        [0.001 if i % 2 == 0 else -0.0006 for i in range(n)],
        index=idx,
        name="strategy",
    )


class _FakePage:
    def goto(self, *_args, **_kwargs) -> None:
        return None

    def evaluate(self, _script: str):
        return {"sections": [], "totalHeight": 800}

    def wait_for_timeout(self, *_args, **_kwargs) -> None:
        return None

    def pdf(self, path: str, **_kwargs) -> None:
        writer = PdfWriter()
        writer.add_blank_page(width=612, height=792)
        with Path(path).open("wb") as f:
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

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        return None


def test_success_path_leaves_only_the_final_pdf(tmp_path, monkeypatch) -> None:
    pytest.importorskip("playwright")
    import playwright.sync_api as ps

    monkeypatch.setattr(ps, "sync_playwright", lambda: _FakePlaywrightCM())

    out = tmp_path / "report.pdf"
    result = generate_pdf(
        _returns(),
        benchmark_rets=None,
        positions=None,
        transactions=None,
        trades=None,
        title="Clean Success",
        output=str(out),
        rolling_window=10,
    )

    assert result == str(out)
    assert [p.name for p in tmp_path.iterdir()] == ["report.pdf"]


def test_playwright_startup_failure_leaves_no_temp_files(tmp_path, monkeypatch) -> None:
    pytest.importorskip("playwright")
    import playwright.sync_api as ps

    class Crash(Exception):
        pass

    class BrokenContextManager:
        def __enter__(self):
            raise Crash("browser launch failed")

        def __exit__(self, *_args) -> None:
            return None

    monkeypatch.setattr(ps, "sync_playwright", lambda: BrokenContextManager())

    out = tmp_path / "report.pdf"
    with pytest.raises(Crash, match="browser launch failed"):
        generate_pdf(
            _returns(),
            benchmark_rets=None,
            positions=None,
            transactions=None,
            trades=None,
            title="Crashing PDF",
            output=str(out),
            rolling_window=10,
        )

    assert not out.exists()
    assert list(tmp_path.iterdir()) == []


def test_html_generation_failure_leaves_no_temp_files(tmp_path) -> None:
    out = tmp_path / "report.pdf"
    empty = pd.Series(dtype=float)

    with pytest.raises(InputContractError, match="at least one"):
        generate_pdf(
            empty,
            benchmark_rets=None,
            positions=None,
            transactions=None,
            trades=None,
            title="Invalid PDF",
            output=str(out),
            rolling_window=10,
        )

    assert list(tmp_path.iterdir()) == []


def test_render_failure_leaves_no_temp_files(tmp_path, monkeypatch) -> None:
    pytest.importorskip("playwright")
    import playwright.sync_api as ps

    class RenderCrash(Exception):
        pass

    class _ExplodingPage(_FakePage):
        def pdf(self, path: str, **_kwargs) -> None:
            raise RenderCrash("page.pdf failed")

    class _ExplodingBrowser(_FakeBrowser):
        def new_page(self, **_kwargs) -> _ExplodingPage:
            return _ExplodingPage()

    class _ExplodingChromium:
        def launch(self, **_kwargs) -> _ExplodingBrowser:
            return _ExplodingBrowser()

    class _ExplodingPlaywright:
        chromium = _ExplodingChromium()

    class _ExplodingCM:
        def __enter__(self):
            return _ExplodingPlaywright()

        def __exit__(self, _exc_type, _exc, _tb) -> None:
            return None

    monkeypatch.setattr(ps, "sync_playwright", lambda: _ExplodingCM())

    out = tmp_path / "report.pdf"
    with pytest.raises(RenderCrash, match="page.pdf failed"):
        generate_pdf(
            _returns(),
            benchmark_rets=None,
            positions=None,
            transactions=None,
            trades=None,
            title="Exploding PDF",
            output=str(out),
            rolling_window=10,
        )

    assert not out.exists()
    assert list(tmp_path.iterdir()) == []
