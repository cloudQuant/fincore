"""Explicit index and timezone policies for enhanced time-series APIs."""

from __future__ import annotations

from typing import Literal

import pandas as pd

from fincore.exceptions import DataAlignmentError

AlignmentPolicy = Literal["strict", "inner", "outer_dropna"]
TimeSeries = pd.Series | pd.DataFrame

__all__ = ["AlignmentPolicy", "align_time_series"]


def _normalize_datetime_index(index: pd.Index, normalize_tz: str | None) -> pd.Index:
    if not isinstance(index, pd.DatetimeIndex) or normalize_tz is None:
        return index
    if index.tz is None:
        return index.tz_localize("UTC")
    return index.tz_convert("UTC")


def _normalize_object(value: TimeSeries, normalize_tz: str | None) -> TimeSeries:
    result = value.copy()
    result.index = _normalize_datetime_index(result.index, normalize_tz)
    return result


def _validate_normalize_tz(normalize_tz: str | None) -> None:
    if normalize_tz is not None and normalize_tz.upper() != "UTC":
        raise ValueError("normalize_tz currently supports only 'UTC'")


def validate_time_series_timezones(*values: TimeSeries) -> None:
    """Reject mixed datetime-index timezone policies without aligning data."""

    datetime_indices = [value.index for value in values if isinstance(value.index, pd.DatetimeIndex)]
    awareness = {index.tz is not None for index in datetime_indices}
    timezones = {str(index.tz) for index in datetime_indices if index.tz is not None}
    if len(awareness) > 1 or len(timezones) > 1:
        raise DataAlignmentError("timezone mismatch; pass normalize_tz='UTC' explicitly")


def normalize_time_series_timezone(value: TimeSeries, normalize_tz: str) -> TimeSeries:
    """Normalize one object's datetime index without imposing label uniqueness."""

    _validate_normalize_tz(normalize_tz)
    return _normalize_object(value, normalize_tz)


def _validate_indices(values: tuple[TimeSeries, ...], normalize_tz: str | None) -> None:
    if any(value.index.has_duplicates for value in values):
        raise DataAlignmentError("duplicate time-series labels are ambiguous")
    if normalize_tz is None:
        validate_time_series_timezones(*values)


def align_time_series(
    *values: TimeSeries,
    policy: AlignmentPolicy,
    normalize_tz: str | None = None,
) -> tuple[TimeSeries, ...]:
    """Return copies aligned under one explicit label and timezone policy."""

    _validate_normalize_tz(normalize_tz)
    if not values:
        return ()
    if policy not in {"strict", "inner", "outer_dropna"}:
        raise ValueError(f"unknown alignment policy: {policy!r}")

    normalized = tuple(_normalize_object(value, normalize_tz) for value in values)
    _validate_indices(normalized, normalize_tz)

    if policy == "strict":
        reference = normalized[0].index
        if any(not value.index.equals(reference) for value in normalized[1:]):
            raise DataAlignmentError("strict alignment requires identical indices")
        return normalized

    if policy == "inner":
        common = normalized[0].index
        for value in normalized[1:]:
            common = common.intersection(value.index, sort=False)
        common = common.sort_values()
        return tuple(value.loc[common] for value in normalized)

    combined = pd.concat(normalized, axis="columns", keys=range(len(normalized)), sort=True)
    valid = combined.notna().all(axis="columns")
    common = combined.index[valid]
    return tuple(value.loc[common] for value in normalized)
