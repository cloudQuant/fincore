"""Contract tests for the standalone factor-portfolio kernel (Task 5)."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest
from pandas.tseries.offsets import BDay, CustomBusinessDay

from fincore.factor_analysis.portfolio import (
    FactorPortfolioInputs,
    build_factor_portfolio_inputs,
    factor_cumulative_returns,
    factor_positions,
    positions,
)


def _factor_data(*, timezone: str | None = None, intraday: bool = False) -> pd.DataFrame:
    """Return a small, fully specified factor table without fixture imports."""

    dates = pd.bdate_range("2024-01-02", periods=4, name="date")
    if intraday:
        dates = dates + pd.Timedelta(hours=9, minutes=30)
    if timezone is not None:
        dates = dates.tz_localize(timezone)
    assets = pd.Index(["A", "B", "C", "D"], name="asset")
    index = pd.MultiIndex.from_product((dates, assets), names=("date", "asset"))
    return pd.DataFrame(
        {
            "factor": np.tile([1.0, 2.0, 3.0, 4.0], len(dates)),
            "1D": np.asarray(
                [0.10, 0.02, -0.01, 0.03, 0.00, 0.04, 0.01, -0.02, 0.02, -0.02, 0.03, 0.00, 0.01, 0.03, -0.01, 0.02]
            ),
            "5D": np.asarray(
                [0.20, 0.10, -0.05, 0.15, 0.00, 0.08, 0.02, -0.04, 0.04, -0.04, 0.06, 0.00, 0.02, 0.06, -0.02, 0.04]
            ),
            "factor_quantile": np.tile([1, 1, 2, 2], len(dates)),
            "group": np.tile(["tech", "tech", "finance", "finance"], len(dates)),
        },
        index=index,
    )


def _weights() -> pd.Series:
    dates = pd.DatetimeIndex([pd.Timestamp("2024-01-02 09:30"), pd.Timestamp("2024-01-02 12:30")], name="date")
    index = pd.MultiIndex.from_product((dates, pd.Index(["A", "B"], name="asset")))
    return pd.Series([0.75, -0.25, -0.25, 0.75], index=index, name="factor")


def test_positions_holds_intraday_weights_and_expires_on_calendar_boundary() -> None:
    source = _weights()
    original = source.copy(deep=True)

    actual = positions(source, "4h", freq=BDay())

    expected_index = pd.DatetimeIndex(
        ["2024-01-02 09:30", "2024-01-02 12:30", "2024-01-02 13:30", "2024-01-02 16:30"],
        name="date",
    )
    pd.testing.assert_index_equal(actual.index, expected_index)
    np.testing.assert_allclose(actual.loc[pd.Timestamp("2024-01-02 09:30")].to_numpy(), [0.75, -0.25])
    np.testing.assert_allclose(actual.loc[pd.Timestamp("2024-01-02 12:30")].to_numpy(), [0.5, 0.5])
    np.testing.assert_allclose(actual.abs().sum(axis=1).iloc[:3].to_numpy(), np.ones(3))
    np.testing.assert_allclose(actual.iloc[-1].to_numpy(), np.zeros(2))
    pd.testing.assert_series_equal(source, original)


def test_positions_uses_implicit_business_calendar_and_explicit_custom_calendar() -> None:
    source = _weights()
    with pytest.warns(UserWarning, match="freq.*business day"):
        implicit = positions(source, "1D")
    custom = positions(source, "1D", freq=CustomBusinessDay(weekmask="Tue Wed Thu Fri"))

    assert implicit.index[-1] == pd.Timestamp("2024-01-03 12:30")
    assert custom.index[-1] == pd.Timestamp("2024-01-03 12:30")
    assert implicit.index.tz is None


def test_factor_portfolios_cover_filters_weighting_and_calendar_without_input_mutation() -> None:
    source = _factor_data()
    original = source.copy(deep=True)

    cumulative = factor_cumulative_returns(
        source,
        "1D",
        long_short=False,
        equal_weight=True,
        quantiles=[2],
        groups=["finance"],
    )
    positions_frame = factor_positions(
        source,
        "5D",
        long_short=True,
        group_neutral=True,
        equal_weight=True,
    )

    # Quantile-two finance contains C/D, whose equal-weight first-day return
    # is (-0.01 + 0.03) / 2 = 0.01.  This guards the actual filter/weight path.
    assert cumulative.iloc[0] == pytest.approx(1.01)
    assert cumulative.index.name == "date"
    assert positions_frame.index[-1] == source.index.get_level_values("date").max() + pd.offsets.BDay(5)
    np.testing.assert_allclose(
        positions_frame.abs().sum(axis=1).iloc[:-1].to_numpy(),
        np.ones(len(positions_frame) - 1),
    )
    pd.testing.assert_frame_equal(source, original)


def test_factor_portfolio_inputs_are_frozen_and_preserve_daily_alignment_timezone_and_cash() -> None:
    source = _factor_data(timezone="UTC", intraday=True)
    original = source.copy(deep=True)

    output = build_factor_portfolio_inputs(
        source,
        "1D",
        capital=1_000_000,
        long_short=False,
        equal_weight=True,
        benchmark_period="5D",
    )

    assert isinstance(output, FactorPortfolioInputs)
    assert output.returns.index.tz is not None
    assert output.positions.index.tz is not None
    assert output.returns.index.isin(output.positions.index).all()
    assert output.positions.columns[-1] == "cash"
    assert output.benchmark_returns is not None
    assert output.benchmark_returns.name == "benchmark"
    assert output.positions.abs().sum(axis=1).max() > 1.0
    assert output.positions.drop(columns="cash").abs().sum(axis=1).iloc[0] == pytest.approx(1_000_000 * 1.035)
    with pytest.raises(FrozenInstanceError):
        output.returns = pd.Series(dtype=float)  # type: ignore[misc]
    pd.testing.assert_frame_equal(source, original)


def test_factor_portfolio_inputs_use_none_for_a_missing_benchmark_period_and_keep_gross_net_cash() -> None:
    output = build_factor_portfolio_inputs(
        _factor_data(),
        "1D",
        capital=None,
        long_short=True,
        group_neutral=True,
        equal_weight=True,
        benchmark_period="10D",
    )

    assert output.benchmark_returns is None
    gross = output.positions.drop(columns="cash").abs().sum(axis=1)
    net = output.positions.drop(columns="cash").sum(axis=1)
    pd.testing.assert_series_equal(output.positions["cash"], 1.0 - net, check_names=False)
    np.testing.assert_allclose(gross[gross > 0].to_numpy(), np.ones((gross > 0).sum()))


def test_factor_portfolio_capital_positions_fill_the_full_five_day_holding_horizon() -> None:
    """Capital scaling must not turn active post-return-horizon rows into NaN."""

    source = _factor_data()
    output = build_factor_portfolio_inputs(
        source,
        "5D",
        capital=100_000,
        long_short=False,
        equal_weight=True,
        benchmark_period="missing",
    )

    assert output.positions.index[-1] == pd.Timestamp("2024-01-12")
    assert output.returns.index[-1] == pd.Timestamp("2024-01-05")
    assert not output.positions.isna().any().any()
    assert output.positions.loc[pd.Timestamp("2024-01-12")].abs().sum() > 0
