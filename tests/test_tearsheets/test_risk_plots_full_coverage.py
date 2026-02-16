"""Full coverage tests for fincore.tearsheets.risk plotting helpers.

This file tests the ax=None default parameter branches and other
previously uncovered code paths.
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


class TestRiskPlotsFullCoverage:
    """Test cases for full coverage of tearsheets.risk module."""

    def test_plot_style_factor_exposures_with_ax_none(self):
        """Test plot_style_factor_exposures without providing ax."""
        from fincore.tearsheets import risk as tr

        idx = pd.date_range("2020-01-01", periods=10, freq="D")
        style = pd.Series(np.linspace(-0.2, 0.3, len(idx)), index=idx, name="Momentum")

        # Call without ax parameter - should use gca()
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
        # Should use None as factor_name, which defaults to series.name

    def test_plot_sector_exposures_longshort_with_ax_none(self):
        """Test plot_sector_exposures_longshort without providing ax."""
        from fincore.tearsheets import risk as tr

        idx = pd.date_range("2020-01-01", periods=10, freq="D")
        # 11 sectors expected by default mapping
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

        ax = tr.plot_sector_exposures_longshort(long_exposures, short_exposures, sector_dict=custom_sectors)

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

    def test_plot_cap_exposures_longshort_with_ax_none(self):
        """Test plot_cap_exposures_longshort without providing ax."""
        from fincore.tearsheets import risk as tr

        idx = pd.date_range("2020-01-01", periods=10, freq="D")
        # 5 cap buckets
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

        # Test with 0.5 percentile (format is "50.0th percentile" with decimal)
        ax1 = tr.plot_volume_exposures_longshort(longed, shorted, percentile=0.5)
        assert "50" in ax1.get_ylabel() and "percentile" in ax1.get_ylabel()

        ax2 = tr.plot_volume_exposures_gross(grossed, percentile=0.5)
        assert "50" in ax2.get_ylabel() and "percentile" in ax2.get_ylabel()

        # Test with 0.95 percentile
        ax3 = tr.plot_volume_exposures_longshort(longed, shorted, percentile=0.95)
        assert "95" in ax3.get_ylabel() and "percentile" in ax3.get_ylabel()

    def test_all_sector_plot_functions_with_same_data(self):
        """Test all sector plotting functions with consistent data."""
        from fincore.tearsheets import risk as tr

        idx = pd.date_range("2020-01-01", periods=20, freq="D")

        # Create consistent exposures where long + short = gross, long - short = net
        long_exposures = [pd.Series(0.02 * (i + 1), index=idx) for i in range(11)]
        short_exposures = [pd.Series(-0.01 * (i + 1), index=idx) for i in range(11)]
        gross_exposures = [pd.Series((long_exposures[i] - short_exposures[i]).abs(), index=idx) for i in range(11)]
        net_exposures = [pd.Series(long_exposures[i] + short_exposures[i], index=idx) for i in range(11)]

        # Test all three functions
        ax1 = tr.plot_sector_exposures_longshort(long_exposures, short_exposures)
        ax2 = tr.plot_sector_exposures_gross(gross_exposures)
        ax3 = tr.plot_sector_exposures_net(net_exposures)

        assert ax1 is not None
        assert ax2 is not None
        assert ax3 is not None

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

    def test_plot_with_zero_exposures(self):
        """Test plotting functions with zero exposures."""
        from fincore.tearsheets import risk as tr

        idx = pd.date_range("2020-01-01", periods=10, freq="D")

        # Zero exposures
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

        # All negative exposures
        neg_long = [pd.Series(-0.01 * (i + 1), index=idx) for i in range(11)]
        neg_short = [pd.Series(-0.02 * (i + 1), index=idx) for i in range(11)]

        ax = tr.plot_sector_exposures_longshort(neg_long, neg_short)

        assert ax is not None

    def test_plot_style_factor_with_extreme_values(self):
        """Test plot_style_factor_exposures with extreme values."""
        from fincore.tearsheets import risk as tr

        idx = pd.date_range("2020-01-01", periods=10, freq="D")
        # Very large positive and negative values
        extreme_style = pd.Series(np.concatenate(([10] * 5, [-10] * 5)), index=idx, name="Extreme")

        ax = tr.plot_style_factor_exposures(extreme_style)

        assert ax is not None
        # Y-axis limits should be set based on max absolute value
        # The code sets ylim to (-lim, lim) where lim = max(abs(y1), abs(y2))
        # and adds some margin, so lim should be >= 10
        ylim = ax.get_ylim()
        assert abs(ylim[1]) >= 10  # Upper limit should accommodate the values

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
