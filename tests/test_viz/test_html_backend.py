"""Tests for fincore.viz.html_backend."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fincore.viz.html_backend import HtmlReportBuilder


def test_html_report_builder_stats_table_and_metric_cards():
    stats = pd.Series(
        {
            "pos": 1.0,
            "neg": -1.0,
            "zero": 0.0,
            "nan": np.nan,
            "text": "x",
        }
    )

    b = HtmlReportBuilder()
    b.add_title("T").add_stats_table(stats).add_metric_cards(stats)
    html = b.build()

    assert "<h1>T</h1>" in html
    assert 'class="positive"' in html
    assert 'class="negative"' in html
    assert "N/A" in html


def test_html_report_builder_plot_rolling_sharpe_with_benchmark_branch():
    idx = pd.date_range("2020-01-01", periods=3, freq="D")
    sharpe = pd.Series([0.1, 0.2, 0.3], index=idx)
    bench = pd.Series([0.0, 0.1, 0.1], index=idx)

    b = HtmlReportBuilder()
    b.plot_rolling_sharpe(sharpe, benchmark_sharpe=bench, window=10)
    html = b.build()
    assert "Rolling Sharpe Ratio (10-day window)" in html
    assert "Benchmark Rolling Sharpe Ratio" in html


def test_html_report_builder_monthly_heatmap_series_and_dataframe_paths():
    idx = pd.date_range("2020-01-01", periods=40, freq="D")
    s = pd.Series([0.01] * 40, index=idx)

    df = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]], index=[2020, 2021], columns=[1, 2])

    b = HtmlReportBuilder()
    b.plot_monthly_heatmap(s)
    b.plot_monthly_heatmap(df)
    html = b.build()

    assert "Monthly Returns" in html
    assert 'class="dataframe monthly"' in html


def test_series_to_html_table_truncates_to_max_rows():
    idx = pd.date_range("2020-01-01", periods=25, freq="D")
    s = pd.Series(np.arange(25, dtype=float), index=idx)
    html = HtmlReportBuilder._series_to_html_table(s, "Date", "Value", max_rows=20)

    # Should include head and tail, but not the mid point.
    assert "2020-01-01" in html
    assert "2020-01-25" in html
    assert "2020-01-13" not in html


def test_html_report_builder_save(tmp_path):
    b = HtmlReportBuilder().add_title("X")
    out = tmp_path / "r.html"
    b.save(str(out))
    assert out.read_text(encoding="utf-8").startswith("<!DOCTYPE html>")
