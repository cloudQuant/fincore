"""Tests for visualization backends and AnalysisContext viz methods."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fincore.core.context import AnalysisContext
from fincore.viz.base import VizBackend, get_backend


@pytest.fixture
def returns():
    np.random.seed(42)
    return pd.Series(
        np.random.randn(252) * 0.01,
        index=pd.bdate_range("2020-01-01", periods=252),
    )


@pytest.fixture
def ctx(returns):
    return AnalysisContext(returns)


# ------------------------------------------------------------------
# Backend registry
# ------------------------------------------------------------------


class TestGetBackend:
    def test_matplotlib(self):
        backend = get_backend("matplotlib")
        assert isinstance(backend, VizBackend)
        assert backend.__class__.__name__ == "MatplotlibBackend"

    def test_html(self):
        backend = get_backend("html")
        assert isinstance(backend, VizBackend)
        assert backend.__class__.__name__ == "HtmlReportBuilder"

    def test_unknown(self):
        """Test that unknown backend raises ValueError."""
        # Note: plotly and bokeh are now valid backends
        with pytest.raises(ValueError, match="Unknown viz backend"):
            get_backend("invalid_backend")

    def test_case_insensitive(self):
        backend = get_backend("Matplotlib")
        assert isinstance(backend, VizBackend)
        assert backend.__class__.__name__ == "MatplotlibBackend"


# ------------------------------------------------------------------
# Protocol conformance
# ------------------------------------------------------------------


class TestProtocol:
    def test_matplotlib_is_viz_backend(self):
        backend = get_backend("matplotlib")
        assert isinstance(backend, VizBackend)

    def test_html_is_viz_backend(self):
        backend = get_backend("html")
        assert isinstance(backend, VizBackend)


# ------------------------------------------------------------------
# HtmlReportBuilder
# ------------------------------------------------------------------


class TestHtmlReportBuilder:
    def test_build_empty(self):
        builder = get_backend("html")
        html = builder.build()
        assert "<!DOCTYPE html>" in html
        assert "fincore" in html

    def test_add_title(self):
        builder = get_backend("html")
        builder.add_title("Test Report")
        html = builder.build()
        assert "Test Report" in html

    def test_add_stats_table(self, ctx):
        builder = get_backend("html")
        builder.add_stats_table(ctx.perf_stats())
        html = builder.build()
        assert "Annual return" in html
        assert "Sharpe ratio" in html

    def test_add_metric_cards(self, ctx):
        builder = get_backend("html")
        builder.add_metric_cards(
            ctx.perf_stats(),
            keys=["Annual return", "Sharpe ratio"],
        )
        html = builder.build()
        assert "metric-card" in html

    def test_chaining(self, ctx):
        html = get_backend("html").add_title("Report").add_stats_table(ctx.perf_stats()).build()
        assert "Report" in html

    def test_save(self, ctx, tmp_path):
        path = str(tmp_path / "report.html")
        builder = get_backend("html")
        builder.add_title("Test")
        builder.add_stats_table(ctx.perf_stats())
        builder.save(path)
        content = (tmp_path / "report.html").read_text()
        assert "<!DOCTYPE html>" in content


# ------------------------------------------------------------------
# MatplotlibBackend
# ------------------------------------------------------------------


class TestMatplotlibBackend:
    def test_plot_returns(self, returns):
        import matplotlib

        matplotlib.use("Agg")
        backend = get_backend("matplotlib")
        cum_ret = (1 + returns).cumprod() - 1
        ax = backend.plot_returns(cum_ret)
        assert ax is not None

    def test_plot_drawdown(self, returns):
        import matplotlib

        matplotlib.use("Agg")
        backend = get_backend("matplotlib")
        cum_ret = (1 + returns).cumprod()
        dd = cum_ret / cum_ret.cummax() - 1
        ax = backend.plot_drawdown(dd)
        assert ax is not None

    def test_plot_rolling_sharpe(self, returns):
        import matplotlib

        matplotlib.use("Agg")
        backend = get_backend("matplotlib")
        rolling_sharpe = returns.rolling(60).mean() / returns.rolling(60).std()
        ax = backend.plot_rolling_sharpe(rolling_sharpe.dropna())
        assert ax is not None


# ------------------------------------------------------------------
# AnalysisContext integration
# ------------------------------------------------------------------


class TestContextVisualization:
    def test_to_html(self, ctx):
        html = ctx.to_html()
        assert "<!DOCTYPE html>" in html
        assert "Performance Report" in html
        assert "Sharpe ratio" in html

    def test_to_html_save(self, ctx, tmp_path):
        path = tmp_path / "ctx_report.html"
        html = ctx.to_html(path=str(path))
        assert len(html) > 100
        assert "Performance Report" in path.read_text()

    def test_plot_html_backend(self, ctx):
        viz = ctx.plot(backend="html")
        assert isinstance(viz, VizBackend)
        assert viz.__class__.__name__ == "HtmlReportBuilder"

    def test_plot_matplotlib_backend(self, ctx):
        import matplotlib

        matplotlib.use("Agg")
        viz = ctx.plot(backend="matplotlib")
        assert isinstance(viz, VizBackend)
        assert viz.__class__.__name__ == "MatplotlibBackend"
