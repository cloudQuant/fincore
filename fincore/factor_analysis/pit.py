"""Causal point-in-time (PIT) factor-input contracts.

The strict Alphalens facade remains source-compatible and accepts its legacy
factor Series. Enhanced callers can instead materialize a factor Series from
an observation ledger containing ``as_of``, ``known_at``,
``effective_from``, and an explicit universe flag. A value becomes selectable
only on or after both ``known_at`` and ``effective_from``.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd

__all__ = ["PITPoint", "materialize_pit_factor", "validate_pit_alignment"]


@dataclass(frozen=True, slots=True)
class PITPoint:
    """One validated point-in-time factor observation.

    ``as_of <= known_at <= effective_from`` is required so a value cannot be
    selected before it was both observed and available for use.
    """

    as_of: pd.Timestamp
    known_at: pd.Timestamp
    effective_from: pd.Timestamp
    value: float

    def __post_init__(self) -> None:
        timestamps = {
            "as_of": self.as_of,
            "known_at": self.known_at,
            "effective_from": self.effective_from,
        }
        if any(not isinstance(value, pd.Timestamp) or pd.isna(value) for value in timestamps.values()):
            raise TypeError("as_of, known_at, and effective_from must be non-missing pandas Timestamps")
        timezones = {timestamp.tz for timestamp in timestamps.values()}
        if len(timezones) != 1:
            raise ValueError("as_of, known_at, and effective_from must use the same timezone")
        if self.known_at < self.as_of:
            raise ValueError(f"look-ahead bias: known_at {self.known_at} precedes as_of {self.as_of}")
        if self.effective_from < self.known_at:
            raise ValueError(f"look-ahead bias: effective_from {self.effective_from} precedes known_at {self.known_at}")
        if isinstance(self.value, bool):
            raise ValueError("PIT value must be a finite real number")
        try:
            normalized_value = float(self.value)
        except (TypeError, ValueError) as error:
            raise ValueError("PIT value must be a finite real number") from error
        if not np.isfinite(normalized_value):
            raise ValueError("PIT value must be a finite real number")
        object.__setattr__(self, "value", normalized_value)


def materialize_pit_factor(
    observations: pd.DataFrame,
    evaluation_dates: pd.DatetimeIndex | Sequence[object],
) -> pd.Series:
    """Materialize a causal ``(date, asset)`` factor Series from a PIT ledger.

    ``observations`` must contain ``asset``, ``as_of``, ``known_at``,
    ``effective_from``, ``value``, and ``in_universe`` columns. For every
    requested date and asset, the latest effective observation that was known
    by that date is selected. A selected ``in_universe=False`` record removes
    the asset instead of allowing a stale in-universe observation to leak
    forward.
    """
    dates = _require_evaluation_dates(evaluation_dates)
    ledger = _validated_observation_ledger(observations, dates)
    if dates.empty:
        return _empty_factor(dates)

    index_entries: list[tuple[pd.Timestamp, Hashable]] = []
    values: list[float] = []
    for evaluation_date in dates:
        eligible = ledger.loc[(ledger["known_at"] <= evaluation_date) & (ledger["effective_from"] <= evaluation_date)]
        if eligible.empty:
            continue
        latest = (
            eligible.sort_values(["effective_from", "known_at", "as_of"], kind="stable")
            .groupby("asset", sort=False, observed=True)
            .tail(1)
        )
        latest = latest.loc[latest["in_universe"]]
        for asset, value in zip(latest["asset"].to_list(), latest["value"].to_list(), strict=True):
            index_entries.append((evaluation_date, cast("Hashable", asset)))
            values.append(float(cast("Any", value)))

    if not index_entries:
        return _empty_factor(dates)
    index = pd.MultiIndex.from_tuples(index_entries, names=("date", "asset"))
    return pd.Series(values, index=index, name="factor", dtype=float)


def validate_pit_alignment(points: Sequence[PITPoint]) -> None:
    """Validate a sequence of already-typed PIT points without weakening its contract."""
    for point in points:
        if not isinstance(point, PITPoint):
            raise TypeError("points must contain PITPoint instances")
        # Reconstructing runs the complete temporal and finite-value contract,
        # including defensive validation if a caller bypassed frozen dataclass
        # construction through low-level object mutation.
        PITPoint(point.as_of, point.known_at, point.effective_from, point.value)


def _require_evaluation_dates(evaluation_dates: pd.DatetimeIndex | Sequence[object]) -> pd.DatetimeIndex:
    try:
        dates = pd.DatetimeIndex(cast("Any", evaluation_dates), name="date")
    except (TypeError, ValueError) as error:
        raise ValueError("evaluation_dates must contain datetimes with one timezone") from error
    if dates.hasnans:
        raise ValueError("evaluation_dates must not contain missing timestamps")
    if dates.has_duplicates or not dates.is_monotonic_increasing:
        raise ValueError("evaluation_dates must be sorted and duplicate-free")
    return dates


def _validated_observation_ledger(observations: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    required = ("asset", "as_of", "known_at", "effective_from", "value", "in_universe")
    if not isinstance(observations, pd.DataFrame):
        raise TypeError("observations must be a pandas DataFrame")
    missing = [column for column in required if column not in observations.columns]
    if missing:
        raise ValueError(f"observations are missing required columns: {missing!r}")
    if observations.empty:
        raise ValueError("observations must contain at least one row")

    ledger = cast("pd.DataFrame", observations.loc[:, list(required)].copy(deep=True))
    if ledger["asset"].isna().any() or not all(isinstance(asset, Hashable) for asset in ledger["asset"]):
        raise ValueError("observation assets must be non-missing hashable values")
    for column in ("as_of", "known_at", "effective_from"):
        try:
            converted = pd.DatetimeIndex(ledger[column])
        except (TypeError, ValueError) as error:
            raise ValueError(f"observations {column} must contain datetimes with one timezone") from error
        if converted.hasnans:
            raise ValueError(f"observations {column} must not contain missing timestamps")
        if converted.tz != dates.tz:
            raise ValueError(f"observations {column} timezone must match evaluation_dates")
        ledger[column] = converted

    if (ledger["known_at"] < ledger["as_of"]).any():
        raise ValueError("known_at must not precede as_of")
    if (ledger["effective_from"] < ledger["known_at"]).any():
        raise ValueError("effective_from must not precede known_at")
    if ledger.duplicated(["asset", "as_of", "known_at", "effective_from"], keep=False).any():
        raise ValueError("PIT observations must not contain duplicate revisions")

    try:
        values = pd.to_numeric(ledger["value"], errors="raise").astype(float)
    except (TypeError, ValueError) as error:
        raise ValueError("PIT observation values must be numeric") from error
    if not np.isfinite(values.to_numpy()).all():
        raise ValueError("PIT observation values must be finite")
    ledger["value"] = values

    universe = ledger["in_universe"]
    if universe.isna().any() or not all(isinstance(value, (bool, np.bool_)) for value in universe):
        raise ValueError("in_universe must contain non-missing boolean values")
    ledger["in_universe"] = universe.astype(bool)
    return cast("pd.DataFrame", ledger)


def _empty_factor(dates: pd.DatetimeIndex) -> pd.Series:
    index = pd.MultiIndex.from_arrays(
        (pd.DatetimeIndex([], tz=dates.tz), pd.Index([], dtype=object)), names=("date", "asset")
    )
    return pd.Series([], index=index, name="factor", dtype=float)
