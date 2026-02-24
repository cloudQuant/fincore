"""Tests for colormap utilities in fincore.utils.common_utils.

Split from test_common_display.py for maintainability.
"""

from __future__ import annotations

import pytest

from fincore.utils import common_utils as cu


@pytest.mark.p2  # Medium: colormap utility tests
class TestSampleColormap:
    """Test sample_colormap function with various matplotlib versions."""

    def test_sample_colormap_returns_n_colors(self):
        """Test sample_colormap returns requested number of colors."""
        colors = cu.sample_colormap("viridis", 3)
        assert len(colors) == 3

    def test_sample_colormap_uses_mpl_colormaps_registry(self):
        """Test sample_colormap uses mpl.colormaps registry directly."""
        import matplotlib as mpl

        cmap = mpl.colormaps["viridis"]
        colors = cu.sample_colormap("viridis", 2)
        assert len(colors) == 2
        # Verify it returns the same colors as the registry colormap
        assert colors[0] == cmap(0.0)
        assert colors[1] == cmap(1.0)

    def test_sample_colormap_different_cmaps(self):
        """Test sample_colormap works with various colormaps."""
        for cmap in ("viridis", "plasma", "inferno"):
            colors = cu.sample_colormap(cmap, 4)
            assert len(colors) == 4

    def test_sample_colormap_invalid_cmap_raises(self):
        """Test sample_colormap raises KeyError for unknown colormap."""
        with pytest.raises(KeyError):
            cu.sample_colormap("not_a_real_colormap_name", 2)

    def test_sample_colormap_single_sample(self):
        """Test sample_colormap with a single sample."""
        colors = cu.sample_colormap("viridis", 1)
        assert len(colors) == 1
        assert len(colors[0]) == 4  # RGBA tuple
