from __future__ import annotations

from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure
from pandas.testing import assert_frame_equal, assert_series_equal

from fincore.exceptions import ValidationError
from fincore.pyfolio import Pyfolio


@pytest.fixture(autouse=True)
def _close_figures_after_test() -> None:
    yield
    plt.close("all")


def _volume_oracle(
    shares_held: pd.DataFrame,
    volumes: pd.DataFrame,
    percentile: float,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Independent expression of the documented shares/volume contract."""

    shares, aligned_volumes = shares_held.align(volumes, join="inner")
    shares = shares.replace(0, np.nan)

    def percentile_series(fraction: pd.DataFrame) -> pd.Series:
        return 100.0 * fraction.apply(
            lambda row: np.nanpercentile(row.to_numpy(dtype=float), 100.0 * percentile),
            axis="columns",
        )

    long = percentile_series(shares.where(shares > 0).divide(aligned_volumes))
    short = percentile_series((-shares.where(shares < 0)).divide(aligned_volumes))
    gross = percentile_series(shares.abs().divide(aligned_volumes))
    return long, short, gross


def test_real_risk_tear_sheet_runs_compute_plot_sheet_chain_with_eight_axes(
    pyfolio_risk_inputs: Any,
) -> None:
    positions_before = pyfolio_risk_inputs.positions.copy(deep=True)
    sectors_before = pyfolio_risk_inputs.sectors.copy(deep=True)
    caps_before = pyfolio_risk_inputs.caps.copy(deep=True)
    shares_before = pyfolio_risk_inputs.shares_held.copy(deep=True)
    volumes_before = pyfolio_risk_inputs.volumes.copy(deep=True)
    returns_before = pyfolio_risk_inputs.returns.copy(deep=True)

    fig = Pyfolio().create_risk_tear_sheet(
        positions=pyfolio_risk_inputs.positions,
        sectors=pyfolio_risk_inputs.sectors,
        caps=pyfolio_risk_inputs.caps,
        shares_held=pyfolio_risk_inputs.shares_held,
        volumes=pyfolio_risk_inputs.volumes,
        percentile=pyfolio_risk_inputs.percentile,
        returns=pyfolio_risk_inputs.returns,
        estimate_intraday=False,
        run_flask_app=True,
    )

    assert matplotlib.get_backend().lower() == "agg"
    assert isinstance(fig, Figure)
    # Eleven GridSpec rows render as sector 3 + cap 3 + volume 2 axes.
    assert len(fig.axes) == 8
    assert {axis.get_title() for axis in fig.axes} == {
        "Long and short exposures to sectors",
        "Gross exposure to sectors",
        "Net exposures to sectors",
        "Long and short exposures to market caps",
        "Gross exposure to market caps",
        "Net exposure to market caps",
        "Long and short exposures to ill_liquidity",
        "Gross exposure to ill_liquidity",
    }

    assert_frame_equal(pyfolio_risk_inputs.positions, positions_before)
    assert_frame_equal(pyfolio_risk_inputs.sectors, sectors_before)
    assert_frame_equal(pyfolio_risk_inputs.caps, caps_before)
    assert_frame_equal(pyfolio_risk_inputs.shares_held, shares_before)
    assert_frame_equal(pyfolio_risk_inputs.volumes, volumes_before)
    assert_series_equal(pyfolio_risk_inputs.returns, returns_before)


def test_real_volume_sheet_uses_shares_held_and_survives_three_date_false_unpack(
    pyfolio_risk_inputs: Any,
) -> None:
    assert len(pyfolio_risk_inputs.positions.index) == 3
    expected_long, expected_short, expected_gross = _volume_oracle(
        pyfolio_risk_inputs.shares_held,
        pyfolio_risk_inputs.volumes,
        pyfolio_risk_inputs.percentile,
    )

    fig = Pyfolio().create_risk_tear_sheet(
        positions=pyfolio_risk_inputs.positions,
        shares_held=pyfolio_risk_inputs.shares_held,
        volumes=pyfolio_risk_inputs.volumes,
        percentile=pyfolio_risk_inputs.percentile,
        returns=pyfolio_risk_inputs.returns,
        estimate_intraday=False,
        run_flask_app=True,
    )

    assert isinstance(fig, Figure)
    assert len(fig.axes) == 2
    long_short_axis, gross_axis = fig.axes
    np.testing.assert_allclose(long_short_axis.lines[0].get_ydata(), expected_long)
    np.testing.assert_allclose(long_short_axis.lines[1].get_ydata(), expected_short)
    np.testing.assert_allclose(gross_axis.lines[0].get_ydata(), expected_gross)


def test_volume_panel_without_shares_held_is_rejected_instead_of_using_dollar_positions(
    pyfolio_risk_inputs: Any,
) -> None:
    with pytest.raises(ValidationError, match="shares_held"):
        Pyfolio().create_risk_tear_sheet(
            positions=pyfolio_risk_inputs.positions,
            shares_held=None,
            volumes=pyfolio_risk_inputs.volumes,
            percentile=pyfolio_risk_inputs.percentile,
            returns=pyfolio_risk_inputs.returns,
            estimate_intraday=False,
            run_flask_app=True,
        )


def test_real_risk_sheet_intersects_dates_across_all_panels(
    pyfolio_risk_inputs: Any,
) -> None:
    common_date = pyfolio_risk_inputs.positions.index[1]

    fig = Pyfolio().create_risk_tear_sheet(
        positions=pyfolio_risk_inputs.positions,
        sectors=pyfolio_risk_inputs.sectors.iloc[1:],
        caps=pyfolio_risk_inputs.caps.iloc[:2],
        shares_held=pyfolio_risk_inputs.shares_held.iloc[1:2],
        volumes=pyfolio_risk_inputs.volumes,
        percentile=pyfolio_risk_inputs.percentile,
        returns=pyfolio_risk_inputs.returns,
        estimate_intraday=False,
        run_flask_app=True,
    )

    assert isinstance(fig, Figure)
    assert len(fig.axes) == 8
    # The first two lines of the volume axis are the long/short series.  Both
    # must be computed for the one common date rather than by array position.
    long_line, short_line = fig.axes[-2].lines[:2]
    assert len(long_line.get_xdata()) == 1
    assert len(short_line.get_xdata()) == 1
    assert pd.Timestamp(long_line.get_xdata()[0]) == common_date


def test_real_risk_sheet_with_no_overlapping_dates_warns_and_returns_none(
    pyfolio_risk_inputs: Any,
) -> None:
    non_overlapping = pyfolio_risk_inputs.sectors.copy()
    non_overlapping.index = non_overlapping.index + pd.DateOffset(years=1)

    with pytest.warns(UserWarning, match="No overlapping index"):
        result = Pyfolio().create_risk_tear_sheet(
            positions=pyfolio_risk_inputs.positions,
            sectors=non_overlapping,
            returns=pyfolio_risk_inputs.returns,
            estimate_intraday=False,
            run_flask_app=True,
        )

    assert result is None


@pytest.mark.parametrize("case", ["zero_assets", "all_cash"])
def test_real_style_sheet_handles_zero_gross_and_all_cash_portfolios(case: str) -> None:
    index = pd.date_range("2024-07-01", periods=2, freq="B", tz="UTC")
    if case == "zero_assets":
        positions = pd.DataFrame({"AAA": [0.0, 0.0], "cash": [100.0, 100.0]}, index=index)
        style = pd.DataFrame({"AAA": [1.0, -1.0]}, index=index)
    else:
        positions = pd.DataFrame({"cash": [100.0, 100.0]}, index=index)
        style = pd.DataFrame(index=index)

    fig = Pyfolio().create_risk_tear_sheet(
        positions=positions,
        style_factor_panel={"Momentum": style},
        estimate_intraday=False,
        run_flask_app=True,
    )

    assert isinstance(fig, Figure)
    assert len(fig.axes) == 1
    np.testing.assert_allclose(fig.axes[0].lines[0].get_ydata(), [0.0, 0.0])
