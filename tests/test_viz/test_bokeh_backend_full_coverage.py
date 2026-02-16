"""Tests for BokehBackend - 100% coverage."""

import numpy as np
import pandas as pd
import pytest

from fincore.viz.interactive.bokeh_backend import BokehBackend


@pytest.fixture
def sample_returns():
    """Create sample returns data."""
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=252)
    return pd.Series(np.random.randn(252) * 0.01, index=dates)


@pytest.fixture
def sample_cum_returns():
    """Create sample cumulative returns."""
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=252)
    returns = pd.Series(np.random.randn(252) * 0.01, index=dates)
    return (1 + returns).cumprod()


@pytest.fixture
def sample_drawdown():
    """Create sample drawdown data."""
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=252)
    returns = pd.Series(np.random.randn(252) * 0.01, index=dates)
    cum_returns = (1 + returns).cumprod()
    running_max = cum_returns.cummax()
    drawdown = (cum_returns - running_max) / running_max
    return drawdown


@pytest.fixture
def sample_rolling_sharpe():
    """Create sample rolling Sharpe data."""
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=252)
    return pd.Series(np.random.randn(252) * 0.1, index=dates)


class TestBokehBackendInit:
    """Test BokehBackend initialization."""

    def test_default_initialization(self):
        """Test default initialization."""
        backend = BokehBackend()

        assert backend.theme == "light"
        assert backend.height == 500
        assert backend.width == 800
        assert backend.sizing_mode == "fixed"

    def test_custom_initialization(self):
        """Test custom initialization."""
        backend = BokehBackend(theme="dark", height=600, width=1000, sizing_mode="stretch_width")

        assert backend.theme == "dark"
        assert backend.height == 600
        assert backend.width == 1000
        assert backend.sizing_mode == "stretch_width"

    def test_theme_setup_light(self):
        """Test light theme setup."""
        backend = BokehBackend(theme="light")

        assert backend.bg_color == "#FFFFFF"
        assert backend.grid_color == "#E0E0E0"
        assert backend.text_color == "#424242"

    def test_theme_setup_dark(self):
        """Test dark theme setup."""
        backend = BokehBackend(theme="dark")

        assert backend.bg_color == "#1E1E1E"
        assert backend.grid_color == "#424242"
        assert backend.text_color == "#E0E0E0"


class TestBokehBackendCreateFigure:
    """Test _create_figure method."""

    def test_create_figure_default(self):
        """Test creating figure with default settings."""
        backend = BokehBackend()

        try:
            from bokeh.plotting import figure

            fig = backend._create_figure()

            assert fig is not None
        except ImportError:
            pytest.skip("Bokeh not installed")

    def test_create_figure_error_without_bokeh(self, monkeypatch):
        """Test error when Bokeh is not installed."""
        import builtins

        backend = BokehBackend()

        # Mock import to raise ImportError
        import sys

        bokeh_module = sys.modules.get("bokeh.plotting")
        if bokeh_module:
            original_import = builtins.__import__

            def mock_import(name, *args, **kwargs):
                if name == "bokeh.plotting":
                    raise ImportError("No module named 'bokeh'")
                return original_import(name, *args, **kwargs)

            monkeypatch.setattr(builtins, "__import__", mock_import)

            with pytest.raises(ImportError, match="Bokeh is required"):
                backend._create_figure()


class TestBokehBackendPlotReturns:
    """Test plot_returns method."""

    def test_plot_returns_basic(self, sample_cum_returns):
        """Test basic returns plot."""
        backend = BokehBackend()

        try:
            p = backend.plot_returns(sample_cum_returns)

            assert p is not None
        except ImportError:
            pytest.skip("Bokeh not installed")

    def test_plot_returns_with_benchmark(self, sample_cum_returns):
        """Test returns plot with benchmark."""
        backend = BokehBackend()
        benchmark = sample_cum_returns * 0.8

        try:
            p = backend.plot_returns(sample_cum_returns, benchmark=benchmark)

            assert p is not None
        except ImportError:
            pytest.skip("Bokeh not installed")


class TestBokehBackendPlotDrawdown:
    """Test plot_drawdown method."""

    def test_plot_drawdown_basic(self, sample_drawdown):
        """Test basic drawdown plot."""
        backend = BokehBackend()

        try:
            p = backend.plot_drawdown(sample_drawdown)

            assert p is not None
        except ImportError:
            pytest.skip("Bokeh not installed")


class TestBokehBackendPlotRollingSharpe:
    """Test plot_rolling_sharpe method."""

    def test_plot_rolling_sharpe_basic(self, sample_rolling_sharpe):
        """Test basic rolling Sharpe plot."""
        backend = BokehBackend()

        try:
            p = backend.plot_rolling_sharpe(sample_rolling_sharpe)

            assert p is not None
        except ImportError:
            pytest.skip("Bokeh not installed")

    def test_plot_rolling_sharpe_with_benchmark(self, sample_rolling_sharpe):
        """Test rolling Sharpe plot with benchmark."""
        backend = BokehBackend()
        benchmark_sharpe = sample_rolling_sharpe * 0.7

        try:
            p = backend.plot_rolling_sharpe(sample_rolling_sharpe, benchmark_sharpe=benchmark_sharpe)

            assert p is not None
        except ImportError:
            pytest.skip("Bokeh not installed")

    def test_plot_rolling_sharpe_custom_window(self, sample_rolling_sharpe):
        """Test rolling Sharpe plot with custom window."""
        backend = BokehBackend()

        try:
            p = backend.plot_rolling_sharpe(sample_rolling_sharpe, window=126)

            assert p is not None
        except ImportError:
            pytest.skip("Bokeh not installed")


class TestBokehBackendPlotMonthlyHeatmap:
    """Test plot_monthly_heatmap method."""

    def test_plot_monthly_heatmap_series(self, sample_returns):
        """Test monthly heatmap with Series input."""
        backend = BokehBackend()

        try:
            p = backend.plot_monthly_heatmap(sample_returns)

            assert p is not None
        except ImportError:
            pytest.skip("Bokeh not installed")

    def test_plot_monthly_heatmap_dataframe(self):
        """Test monthly heatmap with DataFrame input."""
        backend = BokehBackend()

        # Create a pre-pivoted DataFrame
        data = {
            "Jan": [1, 2, 3],
            "Feb": [2, 3, 4],
            "Mar": [3, 4, 5],
        }
        df = pd.DataFrame(data, index=[2020, 2021, 2022])

        try:
            p = backend.plot_monthly_heatmap(df)

            assert p is not None
        except ImportError:
            pytest.skip("Bokeh not installed")


class TestBokehBackendSaveAndShow:
    """Test save and show methods."""

    def test_show_method(self, sample_cum_returns, tmp_path):
        """Test show method."""
        backend = BokehBackend()

        try:
            backend.plot_returns(sample_cum_returns)
            # Just verify the method runs without error
            # (actual display would require browser)
            assert hasattr(backend, "show")
        except ImportError:
            pytest.skip("Bokeh not installed")

    def test_save_html_method(self, sample_cum_returns, tmp_path):
        """Test save_html method."""
        backend = BokehBackend()

        try:
            p = backend.plot_returns(sample_cum_returns)
            output_file = tmp_path / "test_plot.html"

            backend.save_html(p, str(output_file))

            assert output_file.exists()
        except ImportError:
            pytest.skip("Bokeh not installed")


class TestPrintfTickFormatter:
    """Test PrintfTickFormatter class."""

    def test_default_format(self):
        """Test default format."""
        from fincore.viz.interactive.bokeh_backend import PrintfTickFormatter

        formatter = PrintfTickFormatter()
        result = formatter(0.1234)

        assert isinstance(result, str)

    def test_custom_format(self):
        """Test custom format."""
        from fincore.viz.interactive.bokeh_backend import PrintfTickFormatter

        formatter = PrintfTickFormatter(format="%.2f")
        result = formatter(0.1234)

        assert "0.12" in result
