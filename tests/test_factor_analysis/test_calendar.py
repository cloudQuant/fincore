"""Focused contracts for the standalone factor-calendar primitives."""

from __future__ import annotations

import pandas as pd
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
