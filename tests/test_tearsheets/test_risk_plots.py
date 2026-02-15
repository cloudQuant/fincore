"""Smoke tests for fincore.tearsheets.risk plotting helpers."""

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


def test_risk_tearsheet_plot_helpers_smoke():
    import matplotlib.pyplot as plt

    from fincore.tearsheets import risk as tr

    idx = pd.date_range("2020-01-01", periods=5, freq="D")

    style = pd.Series(np.linspace(-0.1, 0.1, len(idx)), index=idx, name="Value")
    _, ax = plt.subplots()
    ax = tr.plot_style_factor_exposures(style, ax=ax)
    assert ax is not None

    # 11 sectors expected by default mapping.
    long_exposures = [pd.Series(0.01 * (i + 1), index=idx) for i in range(11)]
    short_exposures = [pd.Series(-0.005 * (i + 1), index=idx) for i in range(11)]
    gross_exposures = [pd.Series(0.02 * (i + 1), index=idx) for i in range(11)]
    net_exposures = [pd.Series(0.001 * (i + 1), index=idx) for i in range(11)]

    _, ax2 = plt.subplots()
    ax2 = tr.plot_sector_exposures_longshort(long_exposures, short_exposures, ax=ax2)
    assert ax2 is not None
    _, ax3 = plt.subplots()
    ax3 = tr.plot_sector_exposures_gross(gross_exposures, ax=ax3)
    assert ax3 is not None
    _, ax4 = plt.subplots()
    ax4 = tr.plot_sector_exposures_net(net_exposures, ax=ax4)
    assert ax4 is not None

    # 5 cap buckets.
    cap_long = [pd.Series(0.01 * (i + 1), index=idx) for i in range(5)]
    cap_short = [pd.Series(-0.01 * (i + 1), index=idx) for i in range(5)]
    cap_gross = [pd.Series(0.02 * (i + 1), index=idx) for i in range(5)]
    cap_net = [pd.Series(0.001 * (i + 1), index=idx) for i in range(5)]

    _, ax5 = plt.subplots()
    ax5 = tr.plot_cap_exposures_longshort(cap_long, cap_short, ax=ax5)
    assert ax5 is not None
    _, ax6 = plt.subplots()
    ax6 = tr.plot_cap_exposures_gross(cap_gross, ax=ax6)
    assert ax6 is not None
    _, ax7 = plt.subplots()
    ax7 = tr.plot_cap_exposures_net(cap_net, ax=ax7)
    assert ax7 is not None

    longed = pd.Series([0, 1, 2, 3, 4], index=idx, dtype=float)
    shorted = pd.Series([0, -1, -2, -3, -4], index=idx, dtype=float)
    grossed = pd.Series([1, 1, 1, 1, 1], index=idx, dtype=float)

    _, ax8 = plt.subplots()
    ax8 = tr.plot_volume_exposures_longshort(longed, shorted, percentile=0.2, ax=ax8)
    assert ax8 is not None
    _, ax9 = plt.subplots()
    ax9 = tr.plot_volume_exposures_gross(grossed, percentile=0.2, ax=ax9)
    assert ax9 is not None
