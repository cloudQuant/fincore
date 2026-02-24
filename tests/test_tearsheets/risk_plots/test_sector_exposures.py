"""Tests for sector exposure plotting.

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
class TestSectorExposurePlots:
    """Test cases for sector exposure plotting."""

    def test_plot_sector_exposures_longshort_with_ax_none(self):
        """Test plot_sector_exposures_longshort without providing ax."""
        from fincore.tearsheets import risk as tr

        idx = pd.date_range("2020-01-01", periods=10, freq="D")
        long_exposures = [pd.Series(0.01 * (i + 1), index=idx) for i in range(11)]
        short_exposures = [pd.Series(-0.005 * (i + 1), index=idx) for i in range(11)]

        ax = tr.plot_sector_exposures_longshort(long_exposures, short_exposures)

        assert ax is not None
        assert "Long and short exposures" in ax.get_title()

    def test_plot_sector_exposures_longshort_with_custom_sector_dict(self):
        """Test plot_sector_exposures_longshort with custom sector_dict."""
        from fincore.tearsheets import risk as tr

        idx = pd.date_range("2020-01-01", periods=10, freq="D")
        custom_sectors = {
            1: "Technology",
            2: "Healthcare",
            3: "Finance",
        }
        long_exposures = [pd.Series(0.01 * (i + 1), index=idx) for i in range(3)]
        short_exposures = [pd.Series(-0.005 * (i + 1), index=idx) for i in range(3)]

        ax = tr.plot_sector_exposures_longshort(
            long_exposures,
            short_exposures,
            sector_dict=custom_sectors
        )

        assert ax is not None

    def test_plot_sector_exposures_gross_with_ax_none(self):
        """Test plot_sector_exposures_gross without providing ax."""
        from fincore.tearsheets import risk as tr

        idx = pd.date_range("2020-01-01", periods=10, freq="D")
        gross_exposures = [pd.Series(0.02 * (i + 1), index=idx) for i in range(11)]

        ax = tr.plot_sector_exposures_gross(gross_exposures)

        assert ax is not None
        assert "Gross exposure" in ax.get_title()

    def test_plot_sector_exposures_gross_with_custom_sector_dict(self):
        """Test plot_sector_exposures_gross with custom sector_dict."""
        from fincore.tearsheets import risk as tr

        idx = pd.date_range("2020-01-01", periods=10, freq="D")
        custom_sectors = {
            1: "Tech",
            2: "Bio",
        }
        gross_exposures = [pd.Series(0.02 * (i + 1), index=idx) for i in range(2)]

        ax = tr.plot_sector_exposures_gross(gross_exposures, sector_dict=custom_sectors)

        assert ax is not None

    def test_plot_sector_exposures_net_with_ax_none(self):
        """Test plot_sector_exposures_net without providing ax."""
        from fincore.tearsheets import risk as tr

        idx = pd.date_range("2020-01-01", periods=10, freq="D")
        net_exposures = [pd.Series(0.001 * (i + 1), index=idx) for i in range(11)]

        ax = tr.plot_sector_exposures_net(net_exposures)

        assert ax is not None
        assert "Net exposures" in ax.get_title()

    def test_plot_sector_exposures_net_with_custom_sector_dict(self):
        """Test plot_sector_exposures_net with custom sector_dict."""
        from fincore.tearsheets import risk as tr

        idx = pd.date_range("2020-01-01", periods=10, freq="D")
        custom_sectors = {
            1: "Energy",
            2: "Materials",
        }
        net_exposures = [pd.Series(0.001 * (i + 1), index=idx) for i in range(2)]

        ax = tr.plot_sector_exposures_net(net_exposures, sector_dict=custom_sectors)

        assert ax is not None

    def test_all_sector_plot_functions_with_same_data(self):
        """Test all sector plotting functions with consistent data."""
        from fincore.tearsheets import risk as tr

        idx = pd.date_range("2020-01-01", periods=20, freq="D")

        long_exposures = [pd.Series(0.02 * (i + 1), index=idx) for i in range(11)]
        short_exposures = [pd.Series(-0.01 * (i + 1), index=idx) for i in range(11)]
        gross_exposures = [
            pd.Series((long_exposures[i] - short_exposures[i]).abs(), index=idx)
            for i in range(11)
        ]
        net_exposures = [
            pd.Series(long_exposures[i] + short_exposures[i], index=idx)
            for i in range(11)
        ]

        ax1 = tr.plot_sector_exposures_longshort(long_exposures, short_exposures)
        ax2 = tr.plot_sector_exposures_gross(gross_exposures)
        ax3 = tr.plot_sector_exposures_net(net_exposures)

        assert ax1 is not None
        assert ax2 is not None
        assert ax3 is not None

    def test_plot_with_zero_exposures(self):
        """Test plotting functions with zero exposures."""
        from fincore.tearsheets import risk as tr

        idx = pd.date_range("2020-01-01", periods=10, freq="D")

        zero_long = [pd.Series(0.0, index=idx) for _ in range(11)]
        zero_short = [pd.Series(0.0, index=idx) for _ in range(11)]
        zero_gross = [pd.Series(0.0, index=idx) for _ in range(11)]
        zero_net = [pd.Series(0.0, index=idx) for _ in range(11)]

        ax1 = tr.plot_sector_exposures_longshort(zero_long, zero_short)
        ax2 = tr.plot_sector_exposures_gross(zero_gross)
        ax3 = tr.plot_sector_exposures_net(zero_net)

        assert ax1 is not None
        assert ax2 is not None
        assert ax3 is not None

    def test_plot_with_negative_exposures(self):
        """Test plotting functions with all negative exposures."""
        from fincore.tearsheets import risk as tr

        idx = pd.date_range("2020-01-01", periods=10, freq="D")

        neg_long = [pd.Series(-0.01 * (i + 1), index=idx) for i in range(11)]
        neg_short = [pd.Series(-0.02 * (i + 1), index=idx) for i in range(11)]

        ax = tr.plot_sector_exposures_longshort(neg_long, neg_short)

        assert ax is not None
