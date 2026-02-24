"""Tests for style factor exposure plotting.

Split from test_risk_plots_full_coverage.py for maintainability.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def _mpl_cleanup():
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    yield
    plt.close("all")


@pytest.mark.p2  # Medium: plotting tests
class TestStyleFactorPlots:
    """Test cases for style factor exposure plotting."""

    def test_plot_style_factor_exposures_with_ax_none(self):
        """Test plot_style_factor_exposures without providing ax."""
        from fincore.tearsheets import risk as tr

        idx = pd.date_range("2020-01-01", periods=10, freq="D")
        style = pd.Series(np.linspace(-0.2, 0.3, len(idx)), index=idx, name="Momentum")

        ax = tr.plot_style_factor_exposures(style)

        assert ax is not None
        assert ax.get_title() == "Exposure to Momentum"

    def test_plot_style_factor_exposures_with_factor_name(self):
        """Test plot_style_factor_exposures with explicit factor_name."""
        from fincore.tearsheets import risk as tr

        idx = pd.date_range("2020-01-01", periods=10, freq="D")
        style = pd.Series(np.linspace(-0.1, 0.1, len(idx)), index=idx)

        ax = tr.plot_style_factor_exposures(style, factor_name="Value")

        assert ax is not None
        assert "Value" in ax.get_title()

    def test_plot_style_factor_exposures_with_series_no_name(self):
        """Test plot_style_factor_exposures when Series has no name."""
        from fincore.tearsheets import risk as tr

        idx = pd.date_range("2020-01-01", periods=10, freq="D")
        style = pd.Series(np.linspace(-0.1, 0.1, len(idx)), index=idx)
        style.name = None

        ax = tr.plot_style_factor_exposures(style)

        assert ax is not None

    def test_plot_style_factor_with_extreme_values(self):
        """Test plot_style_factor_exposures with extreme values."""
        from fincore.tearsheets import risk as tr

        idx = pd.date_range("2020-01-01", periods=10, freq="D")
        extreme_style = pd.Series(
            np.concatenate(([10] * 5, [-10] * 5)),
            index=idx,
            name="Extreme"
        )

        ax = tr.plot_style_factor_exposures(extreme_style)

        assert ax is not None
        ylim = ax.get_ylim()
        assert abs(ylim[1]) >= 10
