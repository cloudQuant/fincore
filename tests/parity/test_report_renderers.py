"""Renderer contracts: consume a report document and never recompute finance."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _document():
    from fincore.report.portfolio.compute import build_portfolio_report

    index = pd.date_range("2024-01-02", periods=48, freq="B")
    returns = pd.Series(np.where(np.arange(len(index)) % 2, 0.002, -0.001), index=index)
    return build_portfolio_report(returns, rolling_window=12, title="Renderer Contract")


def test_html_renderer_uses_precomputed_sections_and_embeds_offline_assets(tmp_path) -> None:
    from fincore.report.renderers.html import render_html, write_html

    document = _document()
    html = render_html(document, offline_assets={"report.css": "body{color:#123456;}"})
    bundle = write_html(document, tmp_path / "report.html")

    assert "Renderer Contract" in html
    assert 'data-report-domain="portfolio"' in html
    assert "body{color:#123456;}" in html
    assert bundle.named_artifacts["html"].startswith("<!doctype html>")
    assert bundle.named_artifacts["file"] == tmp_path / "report.html"


def test_matplotlib_renderer_records_fincore_ownership_but_never_closes_caller_axes() -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    from fincore.report.renderers.matplotlib import render_matplotlib

    figure, axis = plt.subplots()
    bundle = render_matplotlib(_document(), axes={"performance.cumulative_returns": axis})

    assert bundle.named_artifacts["axis:performance.cumulative_returns"] is axis
    bundle.close()
    assert plt.fignum_exists(figure.number)
    plt.close(figure)


def test_xlsx_renderer_writes_the_precomputed_document(tmp_path) -> None:
    pytest.importorskip("openpyxl")

    from fincore.report.renderers.xlsx import write_xlsx

    output = tmp_path / "report.xlsx"
    bundle = write_xlsx(_document(), output)

    assert output.exists()
    assert output.stat().st_size > 0
    assert bundle.named_artifacts["file"] == output


def test_interactive_renderers_project_precomputed_series() -> None:
    pytest.importorskip("bokeh")
    pytest.importorskip("plotly")

    from fincore.report.renderers.interactive import render_bokeh, render_plotly

    document = _document()
    plotly_bundle = render_plotly(document)
    bokeh_bundle = render_bokeh(document)

    assert plotly_bundle.named_artifacts["figure"].data
    assert bokeh_bundle.named_artifacts["figure"].renderers


def test_optional_renderer_imports_fail_only_when_the_renderer_is_invoked(monkeypatch, tmp_path) -> None:
    import builtins

    from fincore.exceptions import DependencyError
    from fincore.report.renderers.pdf import write_pdf

    original_import = builtins.__import__

    def missing_playwright(name, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith("playwright"):
            raise ImportError("playwright unavailable")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", missing_playwright)

    try:
        write_pdf(_document(), tmp_path / "report.pdf")
    except DependencyError as error:
        assert "optional_dependency_missing" in str(error)
    else:  # pragma: no cover - test environments should honour the import block.
        raise AssertionError("expected the optional renderer to require Playwright")
