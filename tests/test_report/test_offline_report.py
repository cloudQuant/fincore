"""Task 8 offline-report contract.

Generated HTML reports embed the pinned ECharts library as a package asset
and never reference CDN hosts.  Rendering works with network access
completely disabled.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import fincore.report
from fincore.report.render_html import generate_html

CDN_MARKERS = (
    "cdn.bootcdn.net",
    "cdnjs.cloudflare.com",
    "cdn.jsdelivr.net",
    "https://cdn.",
    "http://cdn.",
    "unpkg.com",
)


def _returns(n: int = 60) -> pd.Series:
    idx = pd.date_range("2024-01-01", periods=n, freq="B", tz="UTC")
    values = [(0.001 if i % 2 == 0 else -0.0007) + (i % 7) * 1e-5 for i in range(n)]
    return pd.Series(values, index=idx, name="strategy")


def test_generated_html_contains_no_cdn_requests(tmp_path) -> None:
    out = tmp_path / "offline.html"
    generate_html(
        _returns(),
        benchmark_rets=None,
        positions=None,
        transactions=None,
        trades=None,
        title="Offline Report",
        output=str(out),
        rolling_window=20,
    )

    html = out.read_text(encoding="utf-8")
    for marker in CDN_MARKERS:
        assert marker not in html
    # Charts are driven by the embedded library, not a remote one.
    assert "echarts.init" in html


def test_vendored_echarts_asset_is_complete_and_pinned() -> None:
    asset = Path(fincore.report.__file__).parent / "assets" / "echarts.min.js"
    assert asset.is_file()

    text = asset.read_text(encoding="utf-8")
    assert len(text) > 900_000
    assert 'version:"5.5.0"' in text
    assert "echarts" in text
    # Must be safe to inline inside a <script> tag.
    assert "</script" not in text


def test_generated_html_embeds_the_echarts_library(tmp_path) -> None:
    out = tmp_path / "embedded.html"
    generate_html(
        _returns(),
        benchmark_rets=None,
        positions=None,
        transactions=None,
        trades=None,
        title="Embedded Library",
        output=str(out),
        rolling_window=20,
    )

    html = out.read_text(encoding="utf-8")
    # The pinned library version marker must be present in the output itself.
    assert 'version:"5.5.0"' in html
    assert "var D=" in html


def test_html_report_renders_with_network_disabled(tmp_path, monkeypatch) -> None:
    import socket

    def blocked_socket(*_args, **_kwargs):
        raise OSError("network access disabled by test")

    monkeypatch.setattr(socket, "socket", blocked_socket)

    out = tmp_path / "blocked.html"
    generate_html(
        _returns(),
        benchmark_rets=None,
        positions=None,
        transactions=None,
        trades=None,
        title="Network Blocked",
        output=str(out),
        rolling_window=20,
    )

    html = out.read_text(encoding="utf-8")
    assert "echarts.init" in html
    for marker in CDN_MARKERS:
        assert marker not in html
