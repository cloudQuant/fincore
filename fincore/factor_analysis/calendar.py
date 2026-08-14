"""Trading-calendar primitives for the standalone factor-analysis kernel.

The implementation intentionally stays on public pandas APIs.  In
particular, it does not use the removed ``MultiIndex.labels`` or mutate index
internals, both of which would make the compatibility surface fragile across
pandas releases.
"""

from __future__ import annotations

import re
import warnings
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BusinessDay, CustomBusinessDay, Day

if TYPE_CHECKING:
    from collections.abc import Iterable

_FORWARD_RETURN_RE = re.compile(r"^(?:\d+(?:D|h|m|s|ms|us|ns))+$", re.IGNORECASE)
_EXACT_DAY_RE = re.compile(r"^(?:\d+D)+$", re.IGNORECASE)


def _require_calendar_offset(freq: Any) -> Day | BusinessDay | CustomBusinessDay:
    if not isinstance(freq, (Day, BusinessDay, CustomBusinessDay)):
        raise ValueError("freq must be Day, BDay or CustomBusinessDay")
    return freq


def infer_trading_calendar(factor_idx: Iterable[object], prices_idx: Iterable[object]) -> CustomBusinessDay:
    """Infer active weekdays and missing-session holidays from two date indexes.

    Both inputs are copied into a normalized ``DatetimeIndex`` first, so the
    function has no side effect on caller-owned index frequency metadata.
    """

    factor_dates = pd.DatetimeIndex(list(factor_idx))
    price_dates = pd.DatetimeIndex(list(prices_idx))
    full_idx = factor_dates.union(price_dates).sort_values()
    if full_idx.empty:
        raise ValueError("cannot infer a trading calendar from empty indexes")

    traded_weekdays: list[str] = []
    holidays: list[object] = []
    weekday_names = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
    for weekday, weekday_name in enumerate(weekday_names):
        used = full_idx[full_idx.dayofweek == weekday].normalize().unique()
        if len(used) == 0:
            continue
        traded_weekdays.append(weekday_name)
        potential = pd.date_range(
            full_idx.min().normalize(),
            full_idx.max().normalize(),
            freq=CustomBusinessDay(weekmask=weekday_name),  # type: ignore[call-arg]
        )
        holidays.extend(timestamp.date() for timestamp in potential.difference(used))
    return CustomBusinessDay(weekmask=" ".join(traded_weekdays), holidays=holidays)  # type: ignore[call-arg]


def add_custom_calendar_timedelta(
    input: pd.Timestamp | pd.DatetimeIndex, timedelta: pd.Timedelta | object, freq: Any
) -> pd.Timestamp | pd.DatetimeIndex:
    """Add a timedelta while counting whole days through a trading calendar."""

    offset = _require_calendar_offset(freq)
    delta = pd.Timedelta(cast("Any", timedelta))
    whole_days = delta.components.days
    remainder = delta - pd.Timedelta(days=whole_days)
    return input + offset * whole_days + remainder


def _negative_aware_calendar_inverse(
    start: pd.Timestamp,
    end: pd.Timestamp,
    offset: BusinessDay | CustomBusinessDay,
    local_end: pd.Timestamp,
    busday_count: int,
) -> pd.Timedelta | None:
    """Recover a normalized negative add delta for same-zone aware endpoints.

    ``Timedelta.components`` represents a negative value with a negative
    whole-day component and a non-negative remainder.  Adding that remainder
    can move the calendar-offset anchor into the next local date, while NumPy's
    half-open reverse count sees one fewer business day.  The only possible
    normalized candidates are the half-open count and up to two preceding
    offsets: one for the negative local remainder and one for an off-session
    ``DateOffset(0)`` roll-forward.  Accept exactly one public-add
    reconstruction, never an arbitrary nearby date.  The bounded choice
    preserves the pinned naive half-open semantics because it applies only to
    same-zone aware endpoints.
    """

    if start.tz is None or end.tz is None or start.tz != end.tz or end >= start:
        return None
    candidates: list[pd.Timedelta] = []
    for whole_days in range(busday_count - 2, busday_count + 1):
        anchor = start + offset * whole_days
        local_anchor = anchor.tz_localize(None)
        remainder = local_end - local_anchor
        if not pd.Timedelta(0) <= remainder < pd.Timedelta(days=1):
            continue
        if anchor + remainder == end:
            candidates.append(pd.Timedelta(days=whole_days) + remainder)
    return candidates[0] if len(candidates) == 1 else None


def diff_custom_calendar_timedeltas(start: object, end: object, freq: Any) -> pd.Timedelta:
    """Return ``end - start`` with whole-day components measured by ``freq``."""

    offset = _require_calendar_offset(freq)
    start_timestamp = pd.Timestamp(cast("Any", start))
    end_timestamp = pd.Timestamp(cast("Any", end))
    weekmask = getattr(offset, "weekmask", None)
    holidays = getattr(offset, "holidays", None)
    if weekmask is None and holidays is None:
        if isinstance(offset, Day):
            weekmask, holidays = "Mon Tue Wed Thu Fri Sat Sun", []
        else:
            weekmask, holidays = "Mon Tue Wed Thu Fri", []

    # ``busday_count`` is the public NumPy implementation used by the pinned
    # source.  Its [start, end) treatment is significant for off-session
    # endpoints (for example Saturday -> Monday), and unlike date_range does
    # not accidentally count a closed endpoint as a session.
    # A trading calendar is defined by local wall dates.  ``to_datetime64``
    # converts aware timestamps to UTC, which can shift a Friday/Saturday
    # boundary for positive or negative offsets.  Drop only the timezone
    # metadata before asking NumPy to count local calendar days.
    local_start = start_timestamp.tz_localize(None) if start_timestamp.tz is not None else start_timestamp
    local_end = end_timestamp.tz_localize(None) if end_timestamp.tz is not None else end_timestamp
    actual_days = int(
        np.busday_count(
            local_start.to_datetime64().astype("datetime64[D]"),
            local_end.to_datetime64().astype("datetime64[D]"),
            weekmask=cast("str", weekmask),
            holidays=cast("Any", holidays),
        )
    )
    if isinstance(offset, (BusinessDay, CustomBusinessDay)):
        inverse = _negative_aware_calendar_inverse(start_timestamp, end_timestamp, offset, local_end, actual_days)
        if inverse is not None:
            return inverse
    # Preserve the local wall-clock remainder as well.  Subtracting two aware
    # timestamps measures elapsed UTC time, which changes by an hour across a
    # DST boundary.  Calendar arithmetic above instead advances local trading
    # dates, so use the timezone-naive local values for the complementary
    # remainder calculation too.
    wall_timediff = local_end - local_start
    return wall_timediff - pd.Timedelta(days=wall_timediff.components.days - actual_days)


def timedelta_to_string(timedelta: pd.Timedelta | object) -> str:
    """Format a ``Timedelta`` using the pinned forward-return label grammar."""

    delta = pd.Timedelta(cast("Any", timedelta))
    components = delta.components
    pieces: list[str] = []
    if components.days:
        pieces.append(f"{components.days}D")
    for attribute, suffix in (
        ("hours", "h"),
        ("minutes", "m"),
        ("seconds", "s"),
        ("milliseconds", "ms"),
        ("microseconds", "us"),
        ("nanoseconds", "ns"),
    ):
        value = getattr(components, attribute)
        if value:
            pieces.append(f"{value}{suffix}")
    return "".join(pieces)


def timedelta_strings_to_integers(sequence: Iterable[str]) -> list[int]:
    """Return the whole-day portion of each forward-return label."""

    return [pd.Timedelta(value).days for value in sequence]


def get_forward_returns_columns(columns: pd.Index, require_exact_day_multiple: bool = False) -> pd.Index:
    """Return columns matching the pinned forward-return timedelta grammar."""

    if not isinstance(columns, pd.Index):
        columns = pd.Index(columns)
    pattern = _EXACT_DAY_RE if require_exact_day_multiple else _FORWARD_RETURN_RE
    valid: list[bool] = []
    for column in columns:
        is_valid = isinstance(column, str) and pattern.fullmatch(column) is not None
        valid.append(is_valid)
    if require_exact_day_multiple and any(not item for item in valid):
        warnings.warn("Skipping return periods that aren't exact multiples of days.", stacklevel=2)
    return columns[np.asarray(valid, dtype=bool)]


def backshift_returns_series(series: pd.Series, N: int) -> pd.Series:
    """Move a backward-looking MultiIndex return series ``N`` sessions earlier."""

    if not isinstance(series, pd.Series) or not isinstance(series.index, pd.MultiIndex) or series.index.nlevels != 2:
        raise ValueError("series must use a two-level MultiIndex")
    if not isinstance(N, int) or N <= 0:
        raise ValueError("N must be a positive integer")
    index = series.index
    date_levels, asset_levels = index.levels
    date_codes, asset_codes = (np.asarray(codes) for codes in index.codes)
    if len(date_levels) <= N:
        empty_index = pd.MultiIndex(
            levels=cast("Any", (date_levels[:0], asset_levels)),
            codes=cast("Any", (np.asarray([], dtype=int), np.asarray([], dtype=int))),
            names=index.names,
            verify_integrity=False,
        )
        return pd.Series(index=empty_index, dtype=series.dtype, name=series.name)

    # Shift level *positions*, not observed values.  This keeps gaps and
    # unused levels intact and also works when level names are ``None``.
    retained = date_codes >= N
    shifted_index = pd.MultiIndex(
        levels=cast("Any", (date_levels[:-N], asset_levels)),
        codes=cast("Any", (date_codes[retained] - N, asset_codes[retained])),
        names=index.names,
        verify_integrity=False,
    )
    return pd.Series(series.to_numpy(copy=True)[retained], index=shifted_index, name=series.name)


__all__ = [
    "add_custom_calendar_timedelta",
    "backshift_returns_series",
    "diff_custom_calendar_timedeltas",
    "get_forward_returns_columns",
    "infer_trading_calendar",
    "timedelta_strings_to_integers",
    "timedelta_to_string",
]
