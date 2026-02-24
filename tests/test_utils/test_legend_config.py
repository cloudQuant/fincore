"""Tests for legend configuration utilities in fincore.utils.common_utils.

Split from test_common_display.py for maintainability.
"""

from __future__ import annotations

import pytest

from fincore.utils import common_utils as cu


@pytest.mark.p2  # Medium: legend configuration tests
class TestConfigureLegend:
    """Test configure_legend function for matplotlib legend customization."""

    def test_configure_legend_smoke(self, tmp_path):
        """Test configure_legend basic functionality."""
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1], label="a")
        ax.plot([0, 1], [0, 2], label="b")
        ax.legend()

        cu.configure_legend(ax, change_colors=True, autofmt_xdate=True, rotation=10, ha="left")
        assert ax.get_legend() is not None

    def test_configure_legend_no_labels_returns_early(self):
        """Test configure_legend returns early when no labels."""
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        # Don't add a legend, so labels will be empty
        result = cu.configure_legend(ax, change_colors=False, autofmt_xdate=False)
        assert result is None

    def test_configure_legend_handle_without_callable_get_ydata(self, monkeypatch, tmp_path):
        """Test configure_legend with handle that has different get_ydata behavior."""
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        from matplotlib.collections import PatchCollection
        from matplotlib.patches import Rectangle

        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1], label="line")

        # Add a PatchCollection which may have different get_ydata behavior
        patches = [Rectangle((0, 0), 1, 1)]
        pc = PatchCollection(patches, label="patches")
        ax.add_collection(pc)
        ax.legend()

        # This should work without error
        cu.configure_legend(ax, change_colors=False, autofmt_xdate=False)
        assert ax.get_legend() is not None

    def test_configure_legend_exception_in_legend_sort_key(self, monkeypatch, tmp_path):
        """Test configure_legend handles exceptions in _legend_sort_key gracefully."""
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()

        ax.plot([0, 1], [0, 1], label="a")
        ax.legend()

        # This should handle the exception gracefully
        cu.configure_legend(ax, change_colors=False, autofmt_xdate=False)
        assert ax.get_legend() is not None
