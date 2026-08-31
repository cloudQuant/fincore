"""Pure HTML projection of a precomputed :class:`ReportDocument`."""

from __future__ import annotations

import html
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from fincore.report.models import ReportDocument
from fincore.report.styles import DEFAULT_HTML_CSS
from fincore.runtime import ArtifactBundle

if TYPE_CHECKING:
    from collections.abc import Mapping

    import pandas as pd

    from fincore.report.models import ReportSection

__all__ = ["render_html", "write_html"]


def _display(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, np.generic):
        return _display(value.item())
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if math.isnan(number):
            return "N/A"
        if math.isinf(number):
            return "∞" if number > 0 else "−∞"
        return f"{number:.6g}"
    return str(value)


def _metrics_html(section: ReportSection) -> str:
    if not section.metrics:
        return ""
    rows = "".join(
        f"<tr><th>{html.escape(key)}</th><td>{html.escape(_display(value))}</td><td>{html.escape(section.units.get(key, ''))}</td></tr>"
        for key, value in section.metrics.items()
    )
    return f"<table><thead><tr><th>Metric</th><th>Value</th><th>Unit</th></tr></thead><tbody>{rows}</tbody></table>"


def _table_html(name: str, table: pd.DataFrame, section: ReportSection) -> str:
    label = section.legends.get(name, name)
    return f"<h3>{html.escape(label)}</h3>{table.to_html(escape=True, border=0)}"


def _series_html(name: str, values: pd.Series, section: ReportSection) -> str:
    label = section.legends.get(name, name)
    unit = section.units.get(name, "")
    frame = values.to_frame(name=label)
    return (
        f'<div class="report-series" data-series="{html.escape(name)}">'
        f"<h3>{html.escape(label)} <small>{html.escape(unit)}</small></h3>"
        f"{frame.to_html(escape=True, border=0)}</div>"
    )


def _section_html(section: ReportSection) -> str:
    notes = "".join(f'<p class="report-note">{html.escape(note)}</p>' for note in section.notes)
    tables = "".join(_table_html(name, table, section) for name, table in section.tables.items())
    series = "".join(_series_html(name, values, section) for name, values in section.series.items())
    return f'<section id="{html.escape(section.key)}"><h2>{html.escape(section.title)}</h2>{_metrics_html(section)}{tables}{series}{notes}</section>'


def _asset_html(assets: Mapping[str, str]) -> str:
    blocks: list[str] = []
    for name, content in assets.items():
        escaped_name = html.escape(name)
        if name.endswith(".css"):
            blocks.append(f'<style data-offline-asset="{escaped_name}">{content}</style>')
        else:
            blocks.append(f'<script data-offline-asset="{escaped_name}">{content}</script>')
    return "".join(blocks)


def render_html(document: ReportDocument, *, offline_assets: Mapping[str, str] | None = None) -> str:
    """Render data already present in *document*, without financial computation."""

    if not isinstance(document, ReportDocument):
        raise TypeError("document must be a ReportDocument")
    assets = {**document.offline_assets, **dict(offline_assets or {})}
    if any(not isinstance(name, str) or not isinstance(content, str) for name, content in assets.items()):
        raise TypeError("offline_assets must map text names to text content")
    body = "".join(_section_html(section) for section in document.sections)
    return (
        '<!doctype html><html><head><meta charset="utf-8">'
        f"<title>{html.escape(document.title)}</title><style>{DEFAULT_HTML_CSS}</style>{_asset_html(assets)}"
        "</head>"
        f'<body data-report-domain="{html.escape(document.domain)}" data-report-digest="{document.semantic_digest}">'
        f"<h1>{html.escape(document.title)}</h1>{body}</body></html>"
    )


def write_html(
    document: ReportDocument, target: str | Path, *, offline_assets: Mapping[str, str] | None = None
) -> ArtifactBundle:
    """Write a self-contained HTML document and register inert output artifacts."""

    path = Path(target)
    rendered = render_html(document, offline_assets=offline_assets)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="utf-8")
    bundle = ArtifactBundle(metadata={"backend": "html", "report_digest": document.semantic_digest})
    bundle.add(rendered, owned=False, name="html")
    bundle.add(path, owned=False, name="file")
    return bundle
