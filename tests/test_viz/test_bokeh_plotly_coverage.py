"""Tests for visualization backends - error handling coverage."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


class TestBokehBackendCoverage:
    """Test BokehBackend error handling paths."""

    def test_create_figure_raises_import_error_when_bokeh_not_available(self, monkeypatch):
        """Test that _create_figure raises ImportError when bokeh is not available (line 96-97)."""
        # Mock sys.modules to make bokeh import fail
        import sys

        from fincore.viz.interactive.bokeh_backend import BokehBackend

        original_import = __import__

        def mock_import(name, *args, **kwargs):
            if name == "bokeh" or name.startswith("bokeh."):
                raise ImportError("No module named 'bokeh'")
            return original_import(name, *args, **kwargs)

        # Temporarily remove bokeh from sys.modules
        bokeh_keys = [k for k in sys.modules if k.startswith("bokeh")]
        saved_modules = {}
        for key in bokeh_keys:
            saved_modules[key] = sys.modules.pop(key, None)

        monkeypatch.setattr("builtins.__import__", mock_import)

        backend = BokehBackend()

        with pytest.raises(ImportError, match="Bokeh is required"):
            backend._create_figure()

        # Restore bokeh modules
        for key, val in saved_modules.items():
            if val is not None:
                sys.modules[key] = val

    def test_show_method(self):
        """Test show method calls bokeh_show (line 484-486)."""
        from fincore.viz.interactive.bokeh_backend import BokehBackend

        mock_show = MagicMock()

        with patch("bokeh.io.show", mock_show):
            backend = BokehBackend()
            fig = MagicMock()
            backend.show(fig)

            mock_show.assert_called_once_with(fig)

    def test_save_html_method(self):
        """Test save_html method calls bokeh_save."""
        from fincore.viz.interactive.bokeh_backend import BokehBackend

        mock_save = MagicMock()

        with patch("bokeh.io.save", mock_save):
            backend = BokehBackend()
            fig = MagicMock()
            backend.save_html(fig, "test.html")

            mock_save.assert_called_once_with(fig, "test.html")


class TestPlotlyBackendCoverage:
    """Test PlotlyBackend error handling paths."""

    def test_create_figure_raises_import_error_when_plotly_not_available(self, monkeypatch):
        """Test that _create_figure raises ImportError when plotly is not available (line 131-132)."""
        # Mock sys.modules to make plotly import fail
        import sys

        from fincore.viz.interactive.plotly_backend import PlotlyBackend

        original_import = __import__

        def mock_import(name, *args, **kwargs):
            if name == "plotly" or name.startswith("plotly."):
                raise ImportError("No module named 'plotly'")
            return original_import(name, *args, **kwargs)

        # Temporarily remove plotly from sys.modules
        plotly_keys = [k for k in sys.modules if k.startswith("plotly")]
        saved_modules = {}
        for key in plotly_keys:
            saved_modules[key] = sys.modules.pop(key, None)

        monkeypatch.setattr("builtins.__import__", mock_import)

        backend = PlotlyBackend()

        with pytest.raises(ImportError, match="Plotly is required"):
            backend._create_figure()

        # Restore plotly modules
        for key, val in saved_modules.items():
            if val is not None:
                sys.modules[key] = val

    def test_show_raises_value_error_when_no_figure(self):
        """Test that show raises ValueError when no figure exists (line 528)."""
        from fincore.viz.interactive.plotly_backend import PlotlyBackend

        backend = PlotlyBackend()

        with pytest.raises(ValueError, match="No figure to display"):
            backend.show()

    def test_save_html_raises_value_error_when_no_figure(self):
        """Test that save_html raises ValueError when no figure exists."""
        from fincore.viz.interactive.plotly_backend import PlotlyBackend

        backend = PlotlyBackend()

        with pytest.raises(ValueError, match="No figure to save"):
            backend.save_html("test.html")

    def test_save_image_raises_value_error_when_no_figure(self):
        """Test that save_image raises ValueError when no figure exists."""
        from fincore.viz.interactive.plotly_backend import PlotlyBackend

        backend = PlotlyBackend()

        with pytest.raises(ValueError, match="No figure to save"):
            backend.save_image("test.png")


class TestBokehBackendAdditionalCoverage:
    """Additional tests to improve BokehBackend coverage."""

    def test_plot_returns_with_benchmark(self):
        """Test plot_returns with benchmark (covers lines 169-175)."""
        from fincore.viz.interactive.bokeh_backend import BokehBackend

        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        returns = pd.Series(np.random.randn(100).cumsum() / 100, index=dates)
        benchmark = pd.Series(np.random.randn(100).cumsum() / 100, index=dates)

        backend = BokehBackend()
        fig = backend.plot_returns(returns, benchmark)

        assert fig is not None

    def test_plot_rolling_sharpe_with_benchmark(self):
        """Test plot_rolling_sharpe with benchmark (covers lines 319-325)."""
        from fincore.viz.interactive.bokeh_backend import BokehBackend

        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        sharpe = pd.Series(np.random.randn(100).cumsum() / 100, index=dates)
        benchmark_sharpe = pd.Series(np.random.randn(100).cumsum() / 100, index=dates)

        backend = BokehBackend()
        fig = backend.plot_rolling_sharpe(sharpe, benchmark_sharpe)

        assert fig is not None

    def test_plot_rolling_sharpe_no_span_fallback_to_hspan(self):
        """Test plot_rolling_sharpe when Span is not available (covers lines 342-344)."""
        from bokeh.models import Span

        from fincore.viz.interactive.bokeh_backend import BokehBackend

        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        sharpe = pd.Series(np.random.randn(100).cumsum() / 100, index=dates)

        backend = BokehBackend()

        # Patch importlib to simulate missing Span but present HSpan
        original_import_module = BokehBackend.plot_rolling_sharpe.__globals__.get("importlib")

        # Create a mock module with Span=None
        mock_models = MagicMock()
        mock_models.Span = None

        # Use the real Span for HSpan since they're typically in the same module
        mock_models.HSpan = Span

        with patch("fincore.viz.interactive.bokeh_backend.importlib") as mock_importlib:
            mock_importlib.import_module.return_value = mock_models

            # Call the actual method from the module level
            import fincore.viz.interactive.bokeh_backend as be_module

            be_module.importlib = mock_importlib

            fig = backend.plot_rolling_sharpe(sharpe)
            assert fig is not None

            # Restore
            be_module.importlib = original_import_module or __import__("importlib")

    def test_plot_monthly_heatmap_with_dataframe(self):
        """Test plot_monthly_heatmap with DataFrame input (covers line 394)."""
        from fincore.viz.interactive.bokeh_backend import BokehBackend

        # Create pre-pivoted DataFrame
        pivot = pd.DataFrame(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            index=[2020, 2021],
            columns=[1, 2, 3],
        )

        backend = BokehBackend()
        fig = backend.plot_monthly_heatmap(pivot)

        assert fig is not None

    def test_plot_monthly_heatmap_empty_values(self):
        """Test plot_monthly_heatmap when values list is empty (covers line 419)."""
        from fincore.viz.interactive.bokeh_backend import BokehBackend

        # Create DataFrame that results in empty values list
        dates = pd.date_range("2020-01-01", periods=3, freq="D")
        returns = pd.Series([np.nan, np.nan, np.nan], index=dates)

        backend = BokehBackend()
        # This will create pivot with all NaN values
        fig = backend.plot_monthly_heatmap(returns)

        assert fig is not None

    def test_printf_tick_formatter_call(self):
        """Test PrintfTickFormatter __call__ method (covers line 510)."""
        from fincore.viz.interactive.bokeh_backend import PrintfTickFormatter

        formatter = PrintfTickFormatter("%.1f")
        result = formatter(1.2345)

        assert result == "1.2"


class TestPlotlyBackendAdditionalCoverage:
    """Additional tests to improve PlotlyBackend coverage."""

    def test_get_template_with_custom_template(self):
        """Test _get_template with custom template set (covers line 124)."""
        from fincore.viz.interactive.plotly_backend import PlotlyBackend

        backend = PlotlyBackend(template="plotly_dark")
        assert backend._get_template() == "plotly_dark"

    def test_plot_efficient_frontier(self):
        """Test plot_efficient_frontier method (covers lines 394-480)."""
        from fincore.viz.interactive.plotly_backend import PlotlyBackend

        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        returns = pd.DataFrame(
            np.random.randn(100, 3) / 100,
            index=dates,
            columns=["A", "B", "C"],
        )

        backend = PlotlyBackend()
        fig = backend.plot_efficient_frontier(returns)

        assert fig is not None
        assert backend._fig is not None

    def test_plot_correlation_matrix(self):
        """Test plot_correlation_matrix method (covers lines 482-522)."""
        from fincore.viz.interactive.plotly_backend import PlotlyBackend

        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        returns = pd.DataFrame(
            np.random.randn(100, 3) / 100,
            index=dates,
            columns=["A", "B", "C"],
        )

        backend = PlotlyBackend()
        fig = backend.plot_correlation_matrix(returns)

        assert fig is not None
        assert backend._fig is not None

    def test_show_with_existing_figure(self):
        """Test show method with existing figure (line 526-528 positive case)."""
        from fincore.viz.interactive.plotly_backend import PlotlyBackend

        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        returns = pd.Series(np.random.randn(100).cumsum() / 100, index=dates)

        backend = PlotlyBackend()
        fig = backend.plot_returns(returns)

        # Create a mock show method
        fig.show = MagicMock()
        backend.show()

        fig.show.assert_called_once()

    def test_save_html_with_existing_figure(self):
        """Test save_html with existing figure (line 540 positive case)."""
        import tempfile

        from fincore.viz.interactive.plotly_backend import PlotlyBackend

        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        returns = pd.Series(np.random.randn(100).cumsum() / 100, index=dates)

        backend = PlotlyBackend()
        fig = backend.plot_returns(returns)

        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            temp_path = f.name

        try:
            # Mock write_html to avoid actual file writing
            fig.write_html = MagicMock()
            backend.save_html(temp_path)
            fig.write_html.assert_called_once_with(temp_path)
        finally:
            import os

            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_save_image_with_existing_figure(self):
        """Test save_image with existing figure (line 556 positive case)."""
        import tempfile

        from fincore.viz.interactive.plotly_backend import PlotlyBackend

        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        returns = pd.Series(np.random.randn(100).cumsum() / 100, index=dates)

        backend = PlotlyBackend()
        fig = backend.plot_returns(returns)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            temp_path = f.name

        try:
            # Mock write_image to avoid actual file writing
            fig.write_image = MagicMock()
            backend.save_image(temp_path)
            fig.write_image.assert_called_once_with(temp_path)
        finally:
            import os

            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_monthly_heatmap_with_dataframe_input(self):
        """Test plot_monthly_heatmap with DataFrame input (line 354)."""
        from fincore.viz.interactive.plotly_backend import PlotlyBackend

        # Create pre-pivoted DataFrame
        pivot = pd.DataFrame(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            index=[2020, 2021],
            columns=[1, 2, 3],
        )

        backend = PlotlyBackend()
        fig = backend.plot_monthly_heatmap(pivot)

        assert fig is not None
