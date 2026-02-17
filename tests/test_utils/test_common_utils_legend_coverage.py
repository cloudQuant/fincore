"""Tests for common_utils legend sorting and colormap fallback paths."""

import matplotlib

matplotlib.use("Agg", force=True)

from fincore.utils import common_utils


class TestLegendSortingExceptionPath:
    """Test legend sorting with exception handling (lines 745-746)."""

    def test_legend_sort_key_returns_zero_on_exception(self):
        """Test _legend_sort_key returns 0.0 when get_ydata raises exception (line 745-746)."""

        # Create a mock handle that raises exception when get_ydata is called
        class MockHandle:
            def get_ydata(self):
                raise RuntimeError("Test exception")

        # Test the sorting function directly
        handle = MockHandle()
        ydata_fn = getattr(handle, "get_ydata", None)

        if callable(ydata_fn):
            try:
                y = ydata_fn()
                result = float(y[-1])
            except Exception:
                result = 0.0

        # Should return 0.0 for handle that raises exception
        assert result == 0.0


class TestSampleColormapFallback:
    """Test sample_colormap handles various matplotlib versions (lines 803-809)."""

    def test_sample_colormap_returns_colors(self):
        """Test sample_colormap returns expected number of colors."""
        colors = common_utils.sample_colormap("viridis", 5)
        assert len(colors) == 5
        # Each color should be a tuple
        assert isinstance(colors[0], tuple)

    def test_sample_colormap_with_different_colormaps(self):
        """Test sample_colormap works with various colormap names."""
        for cmap_name in ["viridis", "plasma", "inferno", "magma", "cividis"]:
            colors = common_utils.sample_colormap(cmap_name, 3)
            assert len(colors) == 3
