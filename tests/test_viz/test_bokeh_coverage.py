"""Tests for Bokeh visualization backend.

Tests error handling and edge cases for BokehBackend.
Split from test_bokeh_plotly_coverage.py for maintainability.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


class TestBokehBackendCoverage:
    """Test BokehBackend error handling paths."""

    def test_create_figure_raises_import_error_when_bokeh_not_available(self, monkeypatch):
        """Test that _create_figure raises ImportError when bokeh is not available (line 96-97)."""
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
