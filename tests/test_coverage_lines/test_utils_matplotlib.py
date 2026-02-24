"""Tests for utils/common_utils.py edge cases.

Targets:
- utils/common_utils.py: 745-746, 803-809 - matplotlib utilities
"""

from matplotlib.figure import Figure
from matplotlib.lines import Line2D


class TestCommonUtilsMatplotlibUtilities:
    """Test common_utils.py lines 745-746, 803-809."""

    def test_configure_legend_get_ydata_exception(self):
        """Lines 745-746: get_ydata() raises exception."""
        from fincore.utils.common_utils import configure_legend

        fig = Figure()
        ax = fig.add_subplot(111)

        class BrokenHandle:
            def get_ydata(self):
                raise RuntimeError("Cannot get ydata")

        line = Line2D([], [], label="normal")
        broken = BrokenHandle()

        # Should handle exception gracefully
        configure_legend(ax, [line, broken], ["normal", "broken"])

    def test_sample_colormap_older_api_fallback(self):
        """Lines 803-809: fallback to older matplotlib API."""
        from fincore.utils.common_utils import sample_colormap

        colors = sample_colormap("viridis", 5)
        assert len(colors) == 5
