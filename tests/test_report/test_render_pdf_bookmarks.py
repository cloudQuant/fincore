from __future__ import annotations

from pathlib import Path

from PyPDF2 import PdfReader, PdfWriter

from fincore.report.render_pdf import _add_pdf_bookmarks


def _write_blank_pdf(path: Path, n_pages: int) -> None:
    writer = PdfWriter()
    for _ in range(n_pages):
        # Default 612x792 points (US Letter). Page size isn't important for bookmark logic.
        writer.add_blank_page(width=612, height=792)
    with path.open("wb") as f:
        writer.write(f)


def test_add_pdf_bookmarks_preserves_pages(tmp_path) -> None:
    in_pdf = tmp_path / "in.pdf"
    out_pdf = tmp_path / "out.pdf"
    _write_blank_pdf(in_pdf, n_pages=3)

    section_info = {
        "sections": [
            {"id": "overview", "title": "Overview", "top": 0},
            {"id": "returns", "title": "Returns", "top": 1200},
        ],
        "totalHeight": 2000,
    }

    _add_pdf_bookmarks(str(in_pdf), str(out_pdf), section_info, report_title="Report")

    assert out_pdf.exists()
    r = PdfReader(str(out_pdf))
    assert len(r.pages) == 3
