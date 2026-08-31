"""Report-private styles; presentation constants do not live in a global package."""

from __future__ import annotations

DEFAULT_HTML_CSS = """
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #1f2937; margin: 2rem auto; max-width: 1100px; }
h1 { border-bottom: 2px solid #1d4ed8; padding-bottom: .4rem; }
h2 { margin-top: 2rem; color: #1e3a8a; }
table { border-collapse: collapse; width: 100%; margin: .8rem 0; }
th, td { border: 1px solid #d1d5db; padding: .45rem .65rem; text-align: right; }
th { background: #eff6ff; text-align: left; }
.report-note { color: #4b5563; }
.report-series { margin: 1rem 0; }
""".strip()

__all__ = ["DEFAULT_HTML_CSS"]
