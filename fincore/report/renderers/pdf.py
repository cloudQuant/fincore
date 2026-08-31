"""Chromium-backed PDF rendering of a precomputed HTML report document."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from fincore.report.renderers.html import render_html
from fincore.runtime import ArtifactBundle
from fincore.runtime.validation import load_optional_module

if TYPE_CHECKING:
    from fincore.report.models import ReportDocument

__all__ = ["write_pdf"]


def write_pdf(document: ReportDocument, target: str | Path) -> ArtifactBundle:
    """Render a document with Chromium; Playwright is imported only on invocation."""

    sync_playwright = load_optional_module(
        "playwright.sync_api",
        dependency="playwright",
        extra="report-pdf",
        message="optional_dependency_missing: Playwright/Chromium is required for PDF report rendering",
    ).sync_playwright
    output = Path(target)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="fincore-report-") as temporary_directory:
        html_path = Path(temporary_directory) / "report.html"
        html_path.write_text(render_html(document), encoding="utf-8")
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            try:
                page = browser.new_page(viewport={"width": 1200, "height": 900})
                page.goto(html_path.resolve().as_uri(), wait_until="networkidle", timeout=60000)
                page.pdf(
                    path=str(output),
                    format="A4",
                    print_background=True,
                    margin={"top": "12mm", "bottom": "12mm", "left": "10mm", "right": "10mm"},
                )
            finally:
                browser.close()
    bundle = ArtifactBundle(metadata={"backend": "pdf", "report_digest": document.semantic_digest})
    bundle.add(output, owned=False, name="file")
    return bundle
