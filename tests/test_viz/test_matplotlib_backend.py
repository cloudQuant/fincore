"""Tests for fincore.viz.matplotlib_backend.

These are smoke tests ensuring plots render without requiring a GUI backend.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _make_date_series(values, start="2020-01-01"):
    idx = pd.date_range(start, periods=len(values), freq="D")
    return pd.Series(values, index=idx)


def test_matplotlib_backend_plots_smoke(monkeypatch):
    import matplotlib

    matplotlib.use("Agg", force=True)

    from fincore.viz.matplotlib_backend import MatplotlibBackend

    be = MatplotlibBackend()

    cum = _make_date_series([0.0, 0.1, 0.2])
    ax1 = be.plot_returns(cum, title="X")
    assert ax1.get_title() == "X"

    dd = _make_date_series([0.0, -0.1, -0.05])
    ax2 = be.plot_drawdown(dd, title="Y")
    assert ax2.get_title() == "Y"

    s = _make_date_series([0.5, 0.6, 0.7])
    b = _make_date_series([0.4, 0.5, 0.55])
    ax3 = be.plot_rolling_sharpe(s, benchmark_sharpe=b, window=20, title=None)
    assert "Rolling Sharpe Ratio" in ax3.get_title()


def test_matplotlib_backend_monthly_heatmap_series_and_vmin_vmax_branches():
    import matplotlib

    matplotlib.use("Agg", force=True)

    from fincore.viz.matplotlib_backend import MatplotlibBackend

    be = MatplotlibBackend()

    # Series path: covers resample/pivot logic.
    s = pd.Series(
        [0.01] * 40,
        index=pd.date_range("2020-01-01", periods=40, freq="D"),
    )
    ax = be.plot_monthly_heatmap(s, title="H")
    assert ax.get_title() == "H"

    # DataFrame path with non-finite min/max: vmin/vmax fallback branch.
    df_nan = pd.DataFrame([[np.nan, np.nan], [np.nan, np.nan]], index=[2020, 2021], columns=["Jan", "Feb"])
    ax2 = be.plot_monthly_heatmap(df_nan)
    assert ax2.get_title() == "Monthly Returns (%)"

    # DataFrame path with constant min/max: equal-range branch.
    df_const = pd.DataFrame([[0.0, 0.0], [0.0, 0.0]], index=[2020, 2021], columns=["Jan", "Feb"])
    ax3 = be.plot_monthly_heatmap(df_const)
    assert ax3.get_title() == "Monthly Returns (%)"
