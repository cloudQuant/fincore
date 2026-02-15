from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import fincore.tearsheets.positions as pos_ts


class _DummyEmpyrical:
    def gross_lev(self, positions: pd.DataFrame) -> pd.Series:
        pos_no_cash = positions.drop("cash", axis="columns")
        gross = pos_no_cash.abs().sum(axis=1)
        nav = positions.sum(axis=1).replace(0.0, np.nan)
        return gross / nav

    def get_max_median_position_concentration(self, positions: pd.DataFrame) -> pd.DataFrame:
        pos_no_cash = positions.drop("cash", axis="columns")
        long = pos_no_cash.where(pos_no_cash > 0).fillna(0.0)
        short = (-pos_no_cash.where(pos_no_cash < 0)).fillna(0.0)
        out = pd.DataFrame(
            {
                "max_long": long.max(axis=1),
                "median_long": long.median(axis=1),
                "max_short": short.max(axis=1),
                "median_short": short.median(axis=1),
            },
            index=positions.index,
        )
        return out

    def get_top_long_short_abs(self, positions_alloc: pd.DataFrame):
        pos_no_cash = positions_alloc.drop("cash", axis="columns", errors="ignore")
        mx = pos_no_cash.max()
        mn = pos_no_cash.min()
        abs_mx = pos_no_cash.abs().max()
        # Return Series indexed by tickers.
        top_long = mx.sort_values(ascending=False).head(10)
        top_short = mn.sort_values(ascending=True).head(10)
        top_abs = abs_mx.sort_values(ascending=False).head(10)
        return top_long, top_short, top_abs


def _make_inputs(periods: int = 20):
    idx = pd.date_range("2023-01-02", periods=periods, freq="B", tz="UTC")
    returns = pd.Series(np.sin(np.linspace(0, 6, len(idx))) * 0.01, index=idx)
    positions = pd.DataFrame(
        {
            "AAA": np.where(np.arange(len(idx)) % 2 == 0, 10.0, 0.0),
            "BBB": np.where(np.arange(len(idx)) % 3 == 0, -5.0, 0.0),
            "cash": 100.0,
        },
        index=idx,
    )
    positions_alloc = positions.div(positions.sum(axis=1), axis=0)
    sector_alloc = pd.DataFrame(
        {"Tech": np.linspace(0.2, 0.4, len(idx)), "Energy": np.linspace(0.1, 0.05, len(idx))},
        index=idx,
    )
    return returns, positions, positions_alloc, sector_alloc


def test_positions_plots_cover_ax_none_branches(monkeypatch):
    emp = _DummyEmpyrical()
    returns, positions, positions_alloc, sector_alloc = _make_inputs()

    # Avoid printing during tests.
    monkeypatch.setattr(pos_ts, "print_table", lambda *_args, **_kwargs: None)

    assert pos_ts.plot_holdings(emp, returns, positions, ax=None) is not None
    assert pos_ts.plot_long_short_holdings(returns, positions, ax=None) is not None
    assert pos_ts.plot_exposures(returns, positions, ax=None) is not None
    assert pos_ts.plot_gross_leverage(emp, returns, positions, ax=None) is not None
    assert pos_ts.plot_max_median_position_concentration(emp, positions, ax=None) is not None
    assert pos_ts.plot_sector_allocations(returns, sector_alloc, ax=None) is not None

    # Cover show_and_plot_top_positions plotting branch with non-default legend_loc.
    ax = pos_ts.show_and_plot_top_positions(
        emp,
        returns,
        positions_alloc,
        show_and_plot=0,  # plot-only
        legend_loc="upper right",  # cover the non-"real_best" branch
        ax=None,
        hide_positions=False,
    )
    assert ax is not None

    plt.close("all")
