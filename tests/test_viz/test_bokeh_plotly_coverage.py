"""Tests for visualization backends - error handling coverage."""

from unittest.mock import MagicMock, patch

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
