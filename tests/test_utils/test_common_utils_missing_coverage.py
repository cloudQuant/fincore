"""Tests for missing coverage in utils/common_utils.py module.

This module covers edge cases and branches that were previously uncovered:
- Line 745-746: configure_legend when get_ydata() raises exception
- Line 803-809: sample_colormap fallback paths
"""

import pytest

try:
    import matplotlib.pyplot as plt

    from fincore.utils import common_utils as cu

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


@pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not installed")
class TestCommonUtilsMissingCoverage:
    """Test common_utils edge cases for 100% coverage."""

    def test_configure_legend_get_ydata_raises(self):
        """Test configure_legend when get_ydata() raises exception (line 745-746)."""
        fig, ax = plt.subplots()

        # Create a handle with a get_ydata that raises exception
        class BrokenHandle:
            def get_ydata(self):
                raise ValueError("Simulated error")

        # Add a normal handle and a broken one
        (line,) = ax.plot([0, 1], [0, 1], label="normal")

        # Manually add broken handle to legend

        broken = BrokenHandle()
        ax.legend([line, broken], ["normal", "broken"])

        # Should handle the exception gracefully
        cu.configure_legend(ax, change_colors=False)

        plt.close(fig)

    def test_sample_colormap_various_cmaps(self):
        """Test sample_colormap works with different colormaps."""
        for cmap_name in ("viridis", "plasma", "coolwarm"):
            colors = cu.sample_colormap(cmap_name, 5)
            assert len(colors) == 5, f"Expected 5 colors for {cmap_name}"

    def test_sample_colormap_returns_rgba_tuples(self):
        """Test sample_colormap returns RGBA tuples."""
        colors = cu.sample_colormap("viridis", 3)
        for color in colors:
            assert len(color) == 4, "Each color should be an RGBA tuple"

    def test_sample_colormap_invalid_name_raises(self):
        """Test sample_colormap raises KeyError for unknown colormap."""
        with pytest.raises(KeyError):
            cu.sample_colormap("nonexistent_cmap_xyz", 5)
