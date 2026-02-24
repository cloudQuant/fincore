"""Tests for Plotly visualization backend.

Tests error handling and edge cases for PlotlyBackend.
Split from test_bokeh_plotly_coverage.py for maintainability.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


class TestPlotlyBackendCoverage:
    """Test PlotlyBackend error handling paths."""

    def test_create_figure_raises_import_error_when_plotly_not_available(self, monkeypatch):
        """Test that _create_figure raises ImportError when plotly is not available (line 131-132)."""
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
