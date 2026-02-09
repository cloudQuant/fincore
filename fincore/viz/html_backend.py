"""HTML report generator backend.

Builds a self-contained HTML report from an
:class:`~fincore.core.context.AnalysisContext` without requiring
matplotlib or any other visualization library.
"""
from __future__ import annotations

import html as _html
from typing import Any, Optional

import numpy as np
import pandas as pd


class HtmlReportBuilder:
    """Build a standalone HTML performance report.

    This class also satisfies the :class:`~fincore.viz.base.VizBackend`
    protocol (the ``plot_*`` methods append HTML sections and return
    ``self`` for chaining).

    Usage::

        from fincore.viz.html_backend import HtmlReportBuilder
        builder = HtmlReportBuilder()
        builder.add_title("Performance Report")
        builder.add_stats_table(perf_stats_series)
        html_str = builder.build()
    """

    _CSS = """\
<style>
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
       margin: 2em auto; max-width: 960px; color: #333; line-height: 1.6; }
h1 { border-bottom: 2px solid #2563eb; padding-bottom: 0.3em; color: #1e40af; }
h2 { color: #1e40af; margin-top: 1.5em; }
table { border-collapse: collapse; width: 100%; margin: 1em 0; }
th, td { border: 1px solid #d1d5db; padding: 8px 12px; text-align: right; }
th { background: #eff6ff; font-weight: 600; text-align: left; }
tr:nth-child(even) { background: #f9fafb; }
.positive { color: #16a34a; }
.negative { color: #dc2626; }
.metric-card { display: inline-block; border: 1px solid #e5e7eb; border-radius: 8px;
               padding: 12px 20px; margin: 6px; min-width: 140px; text-align: center; }
.metric-card .value { font-size: 1.4em; font-weight: 700; }
.metric-card .label { font-size: 0.85em; color: #6b7280; }
.footer { margin-top: 2em; padding-top: 1em; border-top: 1px solid #e5e7eb;
          font-size: 0.8em; color: #9ca3af; }
</style>
"""

    def __init__(self) -> None:
        self._sections: list[str] = []

    # ------------------------------------------------------------------
    # Building blocks
    # ------------------------------------------------------------------

    def add_title(self, title: str = "Performance Report") -> HtmlReportBuilder:
        self._sections.append(f"<h1>{_html.escape(title)}</h1>")
        return self

    def add_heading(self, text: str, level: int = 2) -> HtmlReportBuilder:
        tag = f"h{min(max(level, 1), 6)}"
        self._sections.append(f"<{tag}>{_html.escape(text)}</{tag}>")
        return self

    def add_stats_table(self, stats: pd.Series) -> HtmlReportBuilder:
        rows = []
        for key, val in stats.items():
            if isinstance(val, float) and np.isfinite(val):
                css = "positive" if val > 0 else ("negative" if val < 0 else "")
                formatted = f"{val:.4f}"
            else:
                css = ""
                formatted = str(val)
            rows.append(
                f'<tr><th>{_html.escape(str(key))}</th>'
                f'<td class="{css}">{formatted}</td></tr>'
            )
        table = "<table>\n" + "\n".join(rows) + "\n</table>"
        self._sections.append(table)
        return self

    def add_metric_cards(self, stats: pd.Series, keys: Optional[list] = None) -> HtmlReportBuilder:
        if keys is None:
            keys = list(stats.index)
        cards = []
        for key in keys:
            val = stats.get(key, np.nan)
            if isinstance(val, float) and np.isfinite(val):
                css = "positive" if val > 0 else ("negative" if val < 0 else "")
                formatted = f"{val:.4f}"
            else:
                css = ""
                formatted = "N/A"
            cards.append(
                f'<div class="metric-card">'
                f'<div class="value {css}">{formatted}</div>'
                f'<div class="label">{_html.escape(str(key))}</div>'
                f'</div>'
            )
        self._sections.append("<div>" + "\n".join(cards) + "</div>")
        return self

    def add_html(self, raw_html: str) -> HtmlReportBuilder:
        self._sections.append(raw_html)
        return self

    # ------------------------------------------------------------------
    # VizBackend protocol methods (append HTML sections)
    # ------------------------------------------------------------------

    def plot_returns(self, cum_returns: pd.Series, **kwargs: Any) -> HtmlReportBuilder:
        self.add_heading("Cumulative Returns")
        self.add_html(self._series_to_html_table(cum_returns, "Date", "Cum Return"))
        return self

    def plot_drawdown(self, drawdown: pd.Series, **kwargs: Any) -> HtmlReportBuilder:
        self.add_heading("Drawdown")
        self.add_html(self._series_to_html_table(drawdown, "Date", "Drawdown"))
        return self

    def plot_rolling_sharpe(self, rolling_sharpe: pd.Series, **kwargs: Any) -> HtmlReportBuilder:
        self.add_heading("Rolling Sharpe Ratio")
        self.add_html(self._series_to_html_table(rolling_sharpe, "Date", "Sharpe"))
        return self

    def plot_monthly_heatmap(self, monthly_returns: pd.DataFrame, **kwargs: Any) -> HtmlReportBuilder:
        self.add_heading("Monthly Returns")
        self.add_html(monthly_returns.to_html(classes="monthly", float_format="%.2f%%"))
        return self

    # ------------------------------------------------------------------
    # Build / export
    # ------------------------------------------------------------------

    def build(self) -> str:
        body = "\n".join(self._sections)
        footer = (
            '<div class="footer">'
            "Generated by <strong>fincore</strong>"
            "</div>"
        )
        return (
            "<!DOCTYPE html>\n<html><head><meta charset='utf-8'>\n"
            "<title>Performance Report</title>\n"
            f"{self._CSS}\n</head>\n<body>\n{body}\n{footer}\n</body></html>"
        )

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.build())

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _series_to_html_table(s: pd.Series, col_index: str, col_value: str,
                              max_rows: int = 20) -> str:
        if len(s) > max_rows:
            s = pd.concat([s.head(max_rows // 2), s.tail(max_rows // 2)])
        rows = []
        for idx, val in s.items():
            idx_str = str(idx)[:10] if hasattr(idx, 'strftime') else str(idx)
            val_str = f"{val:.6f}" if isinstance(val, float) else str(val)
            rows.append(f"<tr><td>{idx_str}</td><td>{val_str}</td></tr>")
        return (
            f"<table><tr><th>{col_index}</th><th>{col_value}</th></tr>\n"
            + "\n".join(rows)
            + "\n</table>"
        )
