"""Factor report builders and the bilingual self-contained HTML renderer."""

from __future__ import annotations

from .compute import build_factor_report
from .html import render_factor_html, write_factor_html, write_factor_report, write_factor_report_from_model

__all__ = [
    "build_factor_report",
    "render_factor_html",
    "write_factor_html",
    "write_factor_report",
    "write_factor_report_from_model",
]
