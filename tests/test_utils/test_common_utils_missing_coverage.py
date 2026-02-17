"""Tests for missing coverage in utils/common_utils.py module.

This module covers edge cases and branches that were previously uncovered:
- Line 745-746: configure_legend when get_ydata() raises exception
- Line 803-809: sample_colormap fallback paths
"""

import numpy as np
import pandas as pd
import pytest

try:
    import matplotlib.pyplot as plt
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Rectangle

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
        from matplotlib.lines import Line2D

        broken = BrokenHandle()
        ax.legend([line, broken], ["normal", "broken"])

        # Should handle the exception gracefully
        cu.configure_legend(ax, change_colors=False)

        plt.close(fig)

    def test_sample_colormap_fallback_intermediate_api(self, monkeypatch):
        """Test sample_colormap intermediate API fallback (line 803)."""
        import matplotlib as mpl

        # Mock modern API to fail
        class MockColormaps:
            def __getitem__(self, key):
                raise KeyError(f"{key} not found")

        monkeypatch.setattr(plt, "colormaps", MockColormaps(), raising=False)

        # Mock mpl.colormaps to fail
        class MockColormapsModule:
            @staticmethod
            def get_cmap(name):
                raise AttributeError("get_cmap not available")

        monkeypatch.setattr(mpl, "colormaps", MockColormapsModule(), raising=False)

        # Save original _cm
        from matplotlib.pyplot import cm as _cm

        original_cm = _cm

        try:
            # The intermediate API (_cm.get_cmap) should still work
            colors = cu.sample_colormap("viridis", 5)

            # Should return colors
            assert len(colors) == 5
        finally:
            # Restore
            import matplotlib.pyplot as _plt

            _plt.cm = original_cm

    def test_sample_colormap_fallback_older_api(self, monkeypatch):
        """Test sample_colormap older API fallback (line 806)."""
        from matplotlib.pyplot import cm as _cm

        # Mock all modern APIs to fail
        class MockColormaps:
            def __getitem__(self, key):
                raise KeyError(f"{key} not found")

        monkeypatch.setattr(plt, "colormaps", MockColormaps(), raising=False)

        import matplotlib as mpl

        class MockColormapsModule:
            @staticmethod
            def get_cmap(name):
                raise AttributeError("get_cmap not available")

        monkeypatch.setattr(mpl, "colormaps", MockColormapsModule(), raising=False)

        # Mock get_cmap to fail but keep cmap_d
        class MockCM:
            @staticmethod
            def get_cmap(name):
                raise AttributeError("get_cmap not available")

            @property
            def cmap_d(self):
                # Return a dict with the colormap
                return {"viridis": plt.get_cmap("viridis")}

        monkeypatch.setattr(cu, "_cm", MockCM(), raising=False)

        colors = cu.sample_colormap("viridis", 5)

        # Should use cmap_d fallback
        assert len(colors) == 5

    def test_sample_colormap_fallback_registry(self, monkeypatch):
        """Test sample_colormap _colormaps fallback (line 809)."""
        from matplotlib.pyplot import cm as _cm

        # Mock all modern APIs to fail
        class MockColormaps:
            def __getitem__(self, key):
                raise KeyError(f"{key} not found")

        monkeypatch.setattr(plt, "colormaps", MockColormaps(), raising=False)

        import matplotlib as mpl

        class MockColormapsModule:
            @staticmethod
            def get_cmap(name):
                raise AttributeError("get_cmap not available")

        monkeypatch.setattr(mpl, "colormaps", MockColormapsModule(), raising=False)

        # Mock everything to fail except _colormaps
        class MockCM:
            @staticmethod
            def get_cmap(name):
                raise AttributeError("get_cmap not available")

            @property
            def cmap_d(self):
                raise AttributeError("cmap_d not available")

            @property
            def _colormaps(self):
                return {"viridis": plt.get_cmap("viridis")}

        monkeypatch.setattr(cu, "_cm", MockCM(), raising=False)

        colors = cu.sample_colormap("viridis", 5)

        # Should use _colormaps fallback
        assert len(colors) == 5
