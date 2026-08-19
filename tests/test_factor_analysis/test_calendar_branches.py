"""Branch-completion tests for factor_analysis.calendar."""

from __future__ import annotations

import pandas as pd
import pytest
from pandas.tseries.offsets import BDay, BusinessDay, CustomBusinessDay, Day

from fincore.factor_analysis.calendar import (
    backshift_returns_series,
    diff_custom_calendar_timedeltas,
    infer_trading_calendar,
)


def test_infer_trading_calendar_empty_indexes() -> None:
    with pytest.raises(ValueError, match="empty"):
        infer_trading_calendar([], [])


def test_diff_custom_calendar_timedeltas_day_offset() -> None:
    result = diff_custom_calendar_timedeltas("2024-01-01", "2024-01-05", Day())
    assert result == pd.Timedelta(days=4)


def test_diff_custom_calendar_timedeltas_business_day() -> None:
    result = diff_custom_calendar_timedeltas("2024-01-01", "2024-01-08", BusinessDay())
    assert result == pd.Timedelta(days=5)


def test_diff_custom_calendar_timedeltas_aware_negative() -> None:
    start = pd.Timestamp("2024-01-08", tz="UTC")
    end = pd.Timestamp("2024-01-01", tz="UTC")
    result = diff_custom_calendar_timedeltas(start, end, BDay())
    assert result < pd.Timedelta(0)


def test_diff_custom_calendar_timedeltas_custom_business_day() -> None:
    cbd = CustomBusinessDay(weekmask="Mon Tue Wed Thu Fri")
    result = diff_custom_calendar_timedeltas("2024-01-01", "2024-01-08", cbd)
    assert result == pd.Timedelta(days=5)


def test_backshift_rejects_non_series() -> None:
    with pytest.raises(ValueError, match="MultiIndex"):
        backshift_returns_series(pd.Series([1.0, 2.0]), 1)  # type: ignore[arg-type]


def test_backshift_rejects_non_positive_n() -> None:
    idx = pd.MultiIndex.from_product([["a", "b"], ["x", "y"]])
    series = pd.Series([1.0, 2.0, 3.0, 4.0], index=idx)
    with pytest.raises(ValueError, match="positive integer"):
        backshift_returns_series(series, 0)


def test_backshift_rejects_non_int_n() -> None:
    idx = pd.MultiIndex.from_product([["a", "b"], ["x", "y"]])
    series = pd.Series([1.0, 2.0, 3.0, 4.0], index=idx)
    with pytest.raises(ValueError, match="positive integer"):
        backshift_returns_series(series, "1")  # type: ignore[arg-type]


def test_backshift_empty_when_n_exceeds_dates() -> None:
    idx = pd.MultiIndex.from_product([["a", "b"], ["x", "y"]])
    series = pd.Series([1.0, 2.0, 3.0, 4.0], index=idx)
    result = backshift_returns_series(series, 5)
    assert len(result) == 0


def test_backshift_normal_shift() -> None:
    idx = pd.MultiIndex.from_product([["a", "b", "c"], ["x", "y"]])
    series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], index=idx)
    result = backshift_returns_series(series, 1)
    assert len(result) == 4
