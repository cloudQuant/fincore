"""Tests for cap and volume exposure plotting.

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
class TestCapExposurePlots:
    """Test cases for cap exposure plotting."""

    def test_plot_cap_exposures_longshort_with_ax_none(self):
        """Test plot_cap_exposures_longshort without providing ax."""
        from fincore.tearsheets import risk as tr

        idx = pd.date_range("2020-01-01", periods=10, freq="D")
        cap_long = [pd.Series(0.01 * (i + 1), index=idx) for i in range(5)]
        cap_short = [pd.Series(-0.01 * (i + 1), index=idx) for i in range(5)]

        ax = tr.plot_cap_exposures_longshort(cap_long, cap_short)

        assert ax is not None
        assert "market caps" in ax.get_title()

    def test_plot_cap_exposures_gross_with_ax_none(self):
        """Test plot_cap_exposures_gross without providing ax."""
        from fincore.tearsheets import risk as tr

        idx = pd.date_range("2020-01-01", periods=10, freq="D")
        cap_gross = [pd.Series(0.02 * (i + 1), index=idx) for i in range(5)]

        ax = tr.plot_cap_exposures_gross(cap_gross)

        assert ax is not None
        assert "Gross exposure" in ax.get_title()

    def test_plot_cap_exposures_net_with_ax_none(self):
        """Test plot_cap_exposures_net without providing ax."""
        from fincore.tearsheets import risk as tr

        idx = pd.date_range("2020-01-01", periods=10, freq="D")
        cap_net = [pd.Series(0.001 * (i + 1), index=idx) for i in range(5)]

        ax = tr.plot_cap_exposures_net(cap_net)

        assert ax is not None
        assert "Net exposure" in ax.get_title()

    def test_all_cap_plot_functions_with_same_data(self):
        """Test all cap plotting functions with consistent data."""
        from fincore.tearsheets import risk as tr

        idx = pd.date_range("2020-01-01", periods=20, freq="D")

        cap_long = [pd.Series(0.03 * (i + 1), index=idx) for i in range(5)]
        cap_short = [pd.Series(-0.02 * (i + 1), index=idx) for i in range(5)]
        cap_gross = [pd.Series((cap_long[i] - cap_short[i]).abs(), index=idx) for i in range(5)]
        cap_net = [pd.Series(cap_long[i] + cap_short[i], index=idx) for i in range(5)]

        ax1 = tr.plot_cap_exposures_longshort(cap_long, cap_short)
        ax2 = tr.plot_cap_exposures_gross(cap_gross)
        ax3 = tr.plot_cap_exposures_net(cap_net)

        assert ax1 is not None
        assert ax2 is not None
        assert ax3 is not None


@pytest.mark.p2  # Medium: plotting tests
class TestVolumeExposurePlots:
    """Test cases for volume exposure plotting."""

    def test_plot_volume_exposures_longshort_with_ax_none(self):
        """Test plot_volume_exposures_longshort without providing ax."""
        from fincore.tearsheets import risk as tr

        idx = pd.date_range("2020-01-01", periods=10, freq="D")
        longed = pd.Series([0, 1, 2, 3, 4, 2, 1, 0, -1, 0], index=idx, dtype=float)
        shorted = pd.Series([0, -1, -2, -3, -2, -1, 0, 1, 0, 0], index=idx, dtype=float)

        ax = tr.plot_volume_exposures_longshort(longed, shorted, percentile=0.2)

        assert ax is not None
        assert "ill_liquidity" in ax.get_title()

    def test_plot_volume_exposures_gross_with_ax_none(self):
        """Test plot_volume_exposures_gross without providing ax."""
        from fincore.tearsheets import risk as tr

        idx = pd.date_range("2020-01-01", periods=10, freq="D")
        grossed = pd.Series([1, 1.5, 2, 1.8, 1.2, 1, 0.8, 1, 1.2, 1], index=idx, dtype=float)

        ax = tr.plot_volume_exposures_gross(grossed, percentile=0.2)

        assert ax is not None
        assert "Gross exposure" in ax.get_title()

    def test_plot_volume_exposures_different_percentiles(self):
        """Test volume exposures with different percentile values."""
        from fincore.tearsheets import risk as tr

        idx = pd.date_range("2020-01-01", periods=10, freq="D")
        longed = pd.Series([0, 1, 2, 3, 4, 3, 2, 1, 0, 0], index=idx, dtype=float)
        shorted = pd.Series([0, -1, -2, -2, -1, 0, 1, 0, 0, 0], index=idx, dtype=float)
        grossed = pd.Series([1, 2, 3, 4, 5, 4, 3, 2, 1, 1], index=idx, dtype=float)

        ax1 = tr.plot_volume_exposures_longshort(longed, shorted, percentile=0.5)
        assert "50" in ax1.get_ylabel() and "percentile" in ax1.get_ylabel()

        ax2 = tr.plot_volume_exposures_gross(grossed, percentile=0.5)
        assert "50" in ax2.get_ylabel() and "percentile" in ax2.get_ylabel()

        ax3 = tr.plot_volume_exposures_longshort(longed, shorted, percentile=0.95)
        assert "95" in ax3.get_ylabel() and "percentile" in ax3.get_ylabel()

    def test_plot_volume_with_zero_values(self):
        """Test volume exposure plots with zero values."""
        from fincore.tearsheets import risk as tr

        idx = pd.date_range("2020-01-01", periods=10, freq="D")
        zero_longed = pd.Series(0.0, index=idx)
        zero_shorted = pd.Series(0.0, index=idx)
        zero_grossed = pd.Series(0.0, index=idx)

        ax1 = tr.plot_volume_exposures_longshort(zero_longed, zero_shorted, percentile=0.2)
        ax2 = tr.plot_volume_exposures_gross(zero_grossed, percentile=0.2)

        assert ax1 is not None
        assert ax2 is not None
