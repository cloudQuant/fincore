"""Exact contracts for the bounded float64 active-window positions path."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pandas.tseries.offsets import BDay, CustomBusinessDay

from fincore.factor_analysis import portfolio


def _legacy_positions(
    weights: pd.Series, period: pd.Timedelta | object, freq: object = None
) -> pd.DataFrame:
    """The pre-G11 positions algorithm, retained as an independent oracle."""

    copied = portfolio._require_weight_series(weights)
    weights_frame = copied.unstack()
    holding_period = period if isinstance(period, pd.Timedelta) else pd.Timedelta(period)
    if not isinstance(weights_frame.index, pd.DatetimeIndex):
        raise ValueError("weights date level must be a DatetimeIndex")
    calendar = freq if freq is not None else weights_frame.index.freq
    if calendar is None:
        calendar = BDay()
    trades_index = weights_frame.index.copy()
    return_index = portfolio.add_custom_calendar_timedelta(trades_index, holding_period, calendar)
    position_index = trades_index.union(return_index)
    result = pd.DataFrame(0.0, index=position_index, columns=weights_frame.columns, dtype=float)
    active: list[tuple[pd.Timestamp, pd.Series]] = []

    for timestamp in position_index:
        if timestamp in weights_frame.index:
            assets_weights = weights_frame.loc[timestamp].copy(deep=True)
            if not isinstance(assets_weights, pd.Series):
                raise ValueError("weights must contain at most one row per date and asset")
            expires_at = portfolio.add_custom_calendar_timedelta(
                timestamp, holding_period, calendar
            )
            active.append((expires_at, assets_weights))

        if active and active[0][0] <= timestamp:
            active.pop(0)
        if not active:
            continue

        total = pd.concat([entry[1] for entry in active], axis=1).sum(axis=1)
        gross = total.abs().sum()
        if gross != 0 and not pd.isna(gross):
            result.loc[timestamp, total.index] = total / gross

    return result.fillna(0.0)


def _weights(
    dates: pd.DatetimeIndex, assets: list[object], values: np.ndarray
) -> pd.Series:
    index = pd.MultiIndex.from_product((dates, assets), names=("date", "asset"))
    return pd.Series(np.asarray(values).reshape(-1), index=index)


def _assert_frame_bits_equal(actual: pd.DataFrame, expected: pd.DataFrame) -> None:
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    assert actual.dtypes.equals(expected.dtypes)
    np.testing.assert_array_equal(
        actual.to_numpy(copy=False).view(np.uint64),
        expected.to_numpy(copy=False).view(np.uint64),
    )


def _assert_fastpath_supported(weights: pd.Series, frame: pd.DataFrame) -> None:
    actual = portfolio._float64_position_frame_values(weights, frame)
    supports_audited_layout = pd.__version__ == "2.2.3"
    assert (actual is not None) is supports_audited_layout


def test_float64_positions_match_legacy_bits_for_wide_cancellation_panel() -> None:
    dates = pd.date_range("2024-01-05 15:30", periods=18, freq=BDay(), tz="UTC")
    assets = [f"A{position:03d}" for position in range(500)]
    sequence = np.array(
        [1e16, 1.0, -1e16, 0.5, 2.0, -3.0, 1e-8, -1e-8, 7.0, -7.0],
        dtype=np.float64,
    )
    asset_scale = 1.0 + (np.arange(len(assets), dtype=np.float64) % 13) / 8.0
    values = sequence[np.arange(len(dates)) % len(sequence), None] * asset_scale[None, :]
    values = np.broadcast_to(values, (len(dates), len(assets))).copy()
    values[::4, ::17] = -0.0
    weights = _weights(dates, assets, values)
    frame = portfolio._require_weight_series(weights).unstack()
    _assert_fastpath_supported(weights, frame)

    actual = portfolio.positions(weights, "8D", freq=BDay())
    expected = _legacy_positions(weights, "8D", freq=BDay())

    _assert_frame_bits_equal(actual, expected)


def test_float64_positions_match_legacy_bits_with_missing_and_infinite_values() -> None:
    dates = pd.date_range("2024-02-01 10:15", periods=12, freq=BDay(), tz="UTC")
    assets = ["A", "B", "C", "D", "E"]
    rows = np.array(
        [
            [1e16, 1.0, -1e16, 0.0, -0.0],
            [2.0, np.nan, 4.0, -2.0, 1.0],
            [-3.0, 5.0, np.inf, 1.0, np.nan],
            [0.5, 2.0, -np.inf, np.nan, -1.0],
            [np.nan, -4.0, 2.0, 8.0, 0.0],
            [7.0, 3.0, 1.0, -8.0, np.nan],
            [-1e16, 6.0, 2.0, 0.0, 2.0],
            [1.0, -2.0, -2.0, np.inf, 0.0],
            [3.0, np.nan, 0.25, -np.inf, -3.0],
            [-7.0, 1.0, -0.25, 2.0, 4.0],
            [4.0, 0.0, np.nan, 1.0, 1.0],
            [2.0, -1.0, 5.0, -3.0, 0.0],
        ],
        dtype=np.float64,
    )
    rows = np.column_stack((rows, np.full(len(dates), np.nan, dtype=np.float64)))
    weights = _weights(dates, [*assets, "F"], rows)
    frame = portfolio._require_weight_series(weights).unstack()
    _assert_fastpath_supported(weights, frame)

    with np.errstate(invalid="ignore"):
        actual = portfolio.positions(weights, "7D", freq=BDay())
        expected = _legacy_positions(weights, "7D", freq=BDay())

    _assert_frame_bits_equal(actual, expected)


def test_positions_keep_one_expiry_pop_for_colliding_weekend_expiries() -> None:
    dates = pd.DatetimeIndex(
        ["2024-03-01 15:00:00+00:00", "2024-03-02 15:00:00+00:00"]
    )
    index = pd.MultiIndex.from_tuples(
        [(dates[0], "A"), (dates[1], "B")], names=("date", "asset")
    )
    weights = pd.Series([1.0, 2.0], index=index, dtype=np.float64)

    actual = portfolio.positions(weights, "1D", freq=BDay())
    expected = _legacy_positions(weights, "1D", freq=BDay())

    _assert_frame_bits_equal(actual, expected)
    monday = pd.Timestamp("2024-03-04 15:00:00+00:00")
    assert actual.loc[monday, "A"].tolist() == [0.0, 0.0]
    assert actual.loc[monday, "B"].tolist() == [0.0, 0.0]


def test_float64_positions_match_legacy_bits_with_custom_holiday_calendar() -> None:
    calendar = CustomBusinessDay(holidays=["2024-08-06"])
    dates = pd.DatetimeIndex(
        [
            "2024-08-02 13:30:00+00:00",
            "2024-08-05 13:30:00+00:00",
            "2024-08-07 13:30:00+00:00",
            "2024-08-08 13:30:00+00:00",
            "2024-08-09 13:30:00+00:00",
        ]
    )
    weights = _weights(
        dates,
        ["A", "B", "C"],
        np.array(
            [
                [2.0, -3.0, 1.0],
                [5.0, 1.0, -2.0],
                [-4.0, 3.0, 0.5],
                [1.0, 6.0, -7.0],
                [3.0, -2.0, 4.0],
            ]
        ),
    )

    actual = portfolio.positions(weights, "2D3h", freq=calendar)
    expected = _legacy_positions(weights, "2D3h", freq=calendar)

    _assert_frame_bits_equal(actual, expected)


def test_float64_positions_match_legacy_bits_for_sparse_unsorted_assets() -> None:
    timestamps = pd.DatetimeIndex(
        [
            "2024-09-04 14:00:00+00:00",
            "2024-09-02 14:00:00+00:00",
            "2024-09-03 14:00:00+00:00",
            "2024-09-06 14:00:00+00:00",
        ]
    )
    pairs = [
        (timestamps[0], "Z"),
        (timestamps[0], "A"),
        (timestamps[1], "M"),
        (timestamps[2], "Z"),
        (timestamps[2], "M"),
        (timestamps[3], "A"),
        (timestamps[3], "Z"),
    ]
    weights = pd.Series(
        [1e16, -1e16, 2.0, 3.0, -2.0, 0.5, -1.0],
        index=pd.MultiIndex.from_tuples(pairs, names=("date", "asset")),
        dtype=np.float64,
    )
    frame = portfolio._require_weight_series(weights).unstack()
    _assert_fastpath_supported(weights, frame)

    actual = portfolio.positions(weights, "2D", freq=BDay())
    expected = _legacy_positions(weights, "2D", freq=BDay())

    _assert_frame_bits_equal(actual, expected)
    assert actual.columns.tolist() == ["A", "M", "Z"]


def test_float64_positions_keep_zero_and_negative_zero_output_bits() -> None:
    dates = pd.date_range("2024-10-01", periods=4, freq=BDay(), tz="UTC")
    values = np.array(
        [[0.0, -0.0], [-0.0, 0.0], [0.0, 0.0], [-0.0, -0.0]], dtype=np.float64
    )
    weights = _weights(dates, ["A", "B"], values)

    actual = portfolio.positions(weights, "2D", freq=BDay())
    expected = _legacy_positions(weights, "2D", freq=BDay())

    _assert_frame_bits_equal(actual, expected)


@pytest.mark.parametrize("dtype", [np.float32, object])
def test_positions_keep_legacy_fallback_for_non_float64_values(dtype: object) -> None:
    dates = pd.date_range("2024-04-01", periods=5, freq=BDay(), tz="UTC")
    values = np.array(
        [[1.0, 0.0], [-2.0, 3.0], [np.nan, 4.0], [5.0, -6.0], [0.0, 1.0]],
        dtype=dtype,
    )
    weights = _weights(dates, ["A", "B"], values)
    frame = portfolio._require_weight_series(weights).unstack()
    assert portfolio._float64_position_frame_values(weights, frame) is None

    actual = portfolio.positions(weights, "3D", freq=BDay())
    expected = _legacy_positions(weights, "3D", freq=BDay())

    _assert_frame_bits_equal(actual, expected)


def test_positions_fall_back_for_non_string_asset_labels() -> None:
    dates = pd.date_range("2024-04-15", periods=5, freq=BDay(), tz="UTC")
    values = np.array([[1.0, -2.0], [3.0, 0.5], [-4.0, 5.0], [2.0, 1.0], [0.0, 3.0]])
    weights = _weights(dates, [1, 2], values)
    frame = portfolio._require_weight_series(weights).unstack()
    assert portfolio._float64_position_frame_values(weights, frame) is None

    actual = portfolio.positions(weights, "3D", freq=BDay())
    expected = _legacy_positions(weights, "3D", freq=BDay())

    _assert_frame_bits_equal(actual, expected)


def test_positions_budget_fallback_matches_legacy_float64_bits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dates = pd.date_range("2024-05-01", periods=7, freq=BDay(), tz="UTC")
    values = np.array(
        [[1e16, 1.0, -1e16], [2.0, -3.0, 1.0], [-4.0, 0.5, 7.0]] * 3,
        dtype=np.float64,
    )[: len(dates)]
    weights = _weights(dates, ["A", "B", "C"], values)
    monkeypatch.setattr(portfolio, "_POSITIONS_ACTIVE_SCRATCH_BUDGET", 0)

    actual = portfolio.positions(weights, "3D", freq=BDay())
    expected = _legacy_positions(weights, "3D", freq=BDay())

    _assert_frame_bits_equal(actual, expected)


def test_positions_fall_back_for_series_subclasses() -> None:
    class PositionFrame(pd.DataFrame):
        @property
        def _constructor(self):
            return PositionFrame

    class PositionSeries(pd.Series):
        @property
        def _constructor(self):
            return PositionSeries

        @property
        def _constructor_expanddim(self):
            return PositionFrame

    dates = pd.date_range("2024-06-03", periods=5, freq=BDay(), tz="UTC")
    values = np.arange(15, dtype=np.float64).reshape(5, 3) - 7.0
    weights = PositionSeries(_weights(dates, ["A", "B", "C"], values))
    frame = portfolio._require_weight_series(weights).unstack()
    assert type(frame) is PositionFrame
    assert portfolio._float64_position_frame_values(weights, frame) is None

    actual = portfolio.positions(weights, "2D", freq=BDay())
    expected = _legacy_positions(weights, "2D", freq=BDay())

    _assert_frame_bits_equal(actual, expected)


def test_positions_duplicate_date_asset_keeps_unstack_error_contract() -> None:
    timestamp = pd.Timestamp("2024-07-01", tz="UTC")
    index = pd.MultiIndex.from_tuples(
        [(timestamp, "A"), (timestamp, "A")], names=("date", "asset")
    )
    weights = pd.Series([1.0, 2.0], index=index, dtype=np.float64)

    with pytest.raises(ValueError) as actual_error:
        portfolio.positions(weights, "1D", freq=BDay())
    with pytest.raises(type(actual_error.value)) as expected_error:
        _legacy_positions(weights, "1D", freq=BDay())

    assert str(actual_error.value) == str(expected_error.value)
