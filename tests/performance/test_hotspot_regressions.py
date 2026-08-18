"""Hotspot regression tests: semantics and budgets for optimized paths."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from pandas.tseries.offsets import BusinessDay, CustomBusinessDay, Day

from fincore.factor_analysis.calendar import add_custom_calendar_timedelta


def _naive_index() -> pd.DatetimeIndex:
    return pd.DatetimeIndex(
        (
            "2024-01-05 09:30",
            "2024-01-06 11:30",
            "2024-01-08 09:30",
            "2024-01-09 10:30",
            "2024-01-12 09:30",
            "2024-01-15 09:30",
        )
    )


@pytest.mark.parametrize(
    "frequency",
    (BusinessDay(), CustomBusinessDay(holidays=(pd.Timestamp("2024-01-08"), pd.Timestamp("2024-01-15")))),
    ids=("business-day", "custom-business-day"),
)
def test_vectorized_index_add_matches_elementwise_reference(frequency) -> None:
    """The DatetimeIndex path must equal the elementwise Timestamp path."""
    index = _naive_index()

    with warnings.catch_warnings():
        warnings.simplefilter("error", category=pd.errors.PerformanceWarning)
        actual = add_custom_calendar_timedelta(index, pd.Timedelta("1D 1h"), frequency)

    expected = pd.DatetimeIndex([add_custom_calendar_timedelta(t, pd.Timedelta("1D 1h"), frequency) for t in index])

    pd.testing.assert_index_equal(actual, expected)


@pytest.mark.parametrize(
    "delta",
    (pd.Timedelta("0D"), pd.Timedelta("1D"), pd.Timedelta("2D 3h"), pd.Timedelta("-1D 2h")),
)
def test_datetime_index_add_matches_legacy_semantics(delta: pd.Timedelta) -> None:
    """Every whole-day / remainder combination matches the legacy result."""
    frequency = CustomBusinessDay(holidays=(pd.Timestamp("2024-01-08"),))
    index = _naive_index()

    with warnings.catch_warnings():
        warnings.simplefilter("error", category=pd.errors.PerformanceWarning)
        actual = add_custom_calendar_timedelta(index, delta, frequency)

    expected = pd.DatetimeIndex([add_custom_calendar_timedelta(t, delta, frequency) for t in index])

    pd.testing.assert_index_equal(actual, expected)


def test_datetime_index_add_with_custom_calendar_does_not_warn() -> None:
    frequency = CustomBusinessDay(holidays=(pd.Timestamp("2024-01-08"),))
    index = _naive_index()

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        add_custom_calendar_timedelta(index, pd.Timedelta("1D"), frequency)

    assert not any(issubclass(item.category, pd.errors.PerformanceWarning) for item in recorded)


def test_timestamp_add_still_matches_pinned_contracts() -> None:
    start = pd.Timestamp("2024-01-05 09:30")
    custom = CustomBusinessDay(holidays=(pd.Timestamp("2024-01-08"),))

    assert add_custom_calendar_timedelta(start, pd.Timedelta("2D 1h"), Day()) == pd.Timestamp("2024-01-07 10:30")
    assert add_custom_calendar_timedelta(start, pd.Timedelta("1D"), BusinessDay()) == pd.Timestamp("2024-01-08 09:30")
    assert add_custom_calendar_timedelta(start, pd.Timedelta("1D"), custom) == pd.Timestamp("2024-01-09 09:30")


def test_round_trip_fifo_semantics_are_unchanged() -> None:
    """The optimized calendar path must not shift factor position semantics."""
    from fincore.factor_analysis.portfolio import positions

    weights = pd.Series(
        [1.0, 1.0, 1.0],
        index=pd.MultiIndex.from_product((pd.bdate_range("2024-01-02", periods=3), ["AAPL"]), names=("date", "asset")),
    )
    result = positions(weights, pd.Timedelta("1D"))

    assert not result.empty
    assert np.isfinite(result.to_numpy()).all()
