"""Focused contracts for the standalone factor-calendar primitives."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from pandas.tseries.offsets import BusinessDay, CustomBusinessDay, Day


def test_calendar_primitives_cover_day_business_and_custom_days() -> None:
    """Calendar arithmetic has one public, pandas-version-safe projection."""

    from fincore.factor_analysis.calendar import (
        add_custom_calendar_timedelta,
        diff_custom_calendar_timedeltas,
        infer_trading_calendar,
    )

    start = pd.Timestamp("2024-01-05 09:30")
    assert add_custom_calendar_timedelta(start, pd.Timedelta("2D 1h"), Day()) == pd.Timestamp("2024-01-07 10:30")
    assert add_custom_calendar_timedelta(start, pd.Timedelta("1D"), BusinessDay()) == pd.Timestamp("2024-01-08 09:30")

    custom = CustomBusinessDay(holidays=(pd.Timestamp("2024-01-08"),))
    assert add_custom_calendar_timedelta(start, pd.Timedelta("1D"), custom) == pd.Timestamp("2024-01-09 09:30")
    assert diff_custom_calendar_timedeltas(start, pd.Timestamp("2024-01-09 10:30"), custom) == pd.Timedelta("1D 1h")

    factor_dates = pd.DatetimeIndex(("2024-01-05", "2024-01-09"))
    price_dates = pd.DatetimeIndex(("2024-01-05", "2024-01-08", "2024-01-09"))
    inferred = infer_trading_calendar(factor_dates, price_dates)
    assert isinstance(inferred, CustomBusinessDay)


def test_calendar_labels_columns_and_backshift_are_stable() -> None:
    """Return labels and MultiIndex shifting do not rely on pandas private APIs."""

    from fincore.factor_analysis.calendar import (
        backshift_returns_series,
        get_forward_returns_columns,
        timedelta_strings_to_integers,
        timedelta_to_string,
    )

    assert timedelta_to_string(pd.Timedelta("1D 3h")) == "1D3h"
    assert timedelta_to_string(pd.Timedelta(0)) == ""
    assert timedelta_strings_to_integers(("1D", "3h", "1D3h")) == [1, 0, 1]
    columns = pd.Index(("1D", "2D", "3h", "factor"))
    pd.testing.assert_index_equal(get_forward_returns_columns(columns), pd.Index(("1D", "2D", "3h")))
    pd.testing.assert_index_equal(
        get_forward_returns_columns(columns, require_exact_day_multiple=True), pd.Index(("1D", "2D"))
    )

    index = pd.MultiIndex.from_product((pd.date_range("2024-01-01", periods=3), ("A",)), names=("date", "asset"))
    returns = pd.Series((0.1, 0.2, 0.3), index=index, name="1D")
    shifted = backshift_returns_series(returns, 1)
    expected_index = pd.MultiIndex.from_product(
        (pd.date_range("2024-01-01", periods=2), ("A",)), names=("date", "asset")
    )
    pd.testing.assert_series_equal(shifted, pd.Series((0.2, 0.3), index=expected_index, name="1D"))


def test_calendar_ignores_unknown_forward_return_labels() -> None:
    """Non-duration data columns are excluded without changing strict legacy behavior."""

    from fincore.factor_analysis.calendar import get_forward_returns_columns

    assert get_forward_returns_columns(pd.Index(("not-a-duration",))).empty


def _pinned_busday_difference(
    start: pd.Timestamp, end: pd.Timestamp, freq: BusinessDay | CustomBusinessDay
) -> pd.Timedelta:
    """Public-API oracle for the pinned 3fa17ad utility implementation."""

    weekmask = getattr(freq, "weekmask", None) or "Mon Tue Wed Thu Fri"
    holidays = getattr(freq, "holidays", None)
    if holidays is None:
        holidays = []
    actual_days = np.busday_count(
        np.array(start).astype("datetime64[D]"),
        np.array(end).astype("datetime64[D]"),
        weekmask=weekmask,
        holidays=holidays,
    )
    elapsed = end - start
    return elapsed - pd.Timedelta(days=elapsed.components.days - actual_days)


def test_business_calendar_difference_matches_pinned_half_open_busday_matrix() -> None:
    """BusinessDay/CustomBusinessDay retain the pinned half-open endpoint rule."""

    from fincore.factor_analysis.calendar import diff_custom_calendar_timedeltas

    endpoints = pd.DatetimeIndex(
        (
            "2024-01-05 09:30",  # Friday
            "2024-01-06 09:30",  # Saturday, off session
            "2024-01-07 10:30",  # Sunday, off session
            "2024-01-08 09:30",  # Monday / custom holiday
            "2024-01-09 10:30",
            "2024-01-10 09:30",
            "2024-01-11 10:30",
            "2024-01-12 09:30",
            "2024-01-13 10:30",
            "2024-01-15 09:30",
        )
    )
    for frequency in (BusinessDay(), CustomBusinessDay(holidays=(pd.Timestamp("2024-01-08"),))):
        assert diff_custom_calendar_timedeltas(endpoints[1], endpoints[3], frequency) == pd.Timedelta(0)
        for start in endpoints:
            for end in endpoints:
                assert diff_custom_calendar_timedeltas(start, end, frequency) == _pinned_busday_difference(
                    start, end, frequency
                )


def test_calendar_difference_accepts_timezone_aware_endpoints_without_numpy_parse_warning() -> None:
    """Timezone-aware public calls do not rely on NumPy's deprecated parser path."""

    from fincore.factor_analysis.calendar import diff_custom_calendar_timedeltas

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        result = diff_custom_calendar_timedeltas(
            pd.Timestamp("2024-01-05 09:30", tz="UTC"),
            pd.Timestamp("2024-01-08 09:30", tz="UTC"),
            BusinessDay(),
        )
    assert result == pd.Timedelta("1D")
    assert not any(item.category is DeprecationWarning for item in recorded)


def test_calendar_difference_uses_local_wall_dates_and_inverts_calendar_addition() -> None:
    """Timezone-aware business days are counted in their local calendar, not UTC dates."""

    from fincore.factor_analysis.calendar import add_custom_calendar_timedelta, diff_custom_calendar_timedeltas

    shanghai_start = pd.Timestamp("2024-01-05 00:30", tz="Asia/Shanghai")
    shanghai_end = pd.Timestamp("2024-01-08 00:30", tz="Asia/Shanghai")
    assert diff_custom_calendar_timedeltas(shanghai_start, shanghai_end, BusinessDay()) == pd.Timedelta("1D")

    # The Friday immediately before the DST fallback keeps the local Friday ->
    # Saturday half-open business-day count even though both UTC endpoints are
    # Saturday.  This detects accidental UTC-date conversion for negative offsets.
    new_york_start = pd.Timestamp("2024-11-01 23:30", tz="America/New_York")
    new_york_end = pd.Timestamp("2024-11-02 00:30", tz="America/New_York")
    assert diff_custom_calendar_timedeltas(new_york_start, new_york_end, BusinessDay()) == pd.Timedelta("1D 1h")

    start = pd.Timestamp("2024-01-10 09:30", tz="America/New_York")
    for delta in (pd.Timedelta("1D"), pd.Timedelta("-1D")):
        end = add_custom_calendar_timedelta(start, delta, BusinessDay())
        assert diff_custom_calendar_timedeltas(start, end, BusinessDay()) == delta


@pytest.mark.parametrize(
    ("start", "holiday"),
    [
        (pd.Timestamp("2024-03-08 09:30", tz="America/New_York"), pd.Timestamp("2024-03-11")),
        (pd.Timestamp("2024-11-01 09:30", tz="America/New_York"), pd.Timestamp("2024-11-04")),
    ],
    ids=("spring-forward", "fall-back"),
)
def test_calendar_diff_uses_wall_clock_remainder_across_new_york_dst(
    start: pd.Timestamp, holiday: pd.Timestamp
) -> None:
    """Calendar add/diff remains inverse through both New York DST transitions."""

    from fincore.factor_analysis.calendar import add_custom_calendar_timedelta, diff_custom_calendar_timedeltas

    expected = pd.Timedelta("1D 2h")
    for frequency in (BusinessDay(), CustomBusinessDay(holidays=(holiday,))):
        end = add_custom_calendar_timedelta(start, expected, frequency)
        assert diff_custom_calendar_timedeltas(start, end, frequency) == expected


def test_backshift_preserves_unnamed_unused_level_positions() -> None:
    """Backshifting uses MultiIndex codes, including unobserved date and asset levels."""

    from fincore.factor_analysis.calendar import backshift_returns_series

    dates = pd.DatetimeIndex(("2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"))
    assets = pd.Index(("A", "B", "UNUSED"))
    index = pd.MultiIndex(
        levels=(dates, assets),
        codes=((1, 1, 3), (0, 1, 1)),
        names=(None, None),
    )
    returns = pd.Series((0.1, 0.2, 0.4), index=index, name="1D")
    expected_index = pd.MultiIndex(
        levels=(dates[:-1], assets),
        codes=((0, 0, 2), (0, 1, 1)),
        names=(None, None),
    )

    shifted = backshift_returns_series(returns, 1)
    pd.testing.assert_series_equal(shifted, pd.Series((0.1, 0.2, 0.4), index=expected_index, name="1D"))
    pd.testing.assert_index_equal(shifted.index.levels[0], dates[:-1])
    pd.testing.assert_index_equal(shifted.index.levels[1], assets)
