"""Standalone pre-cleaned factor-performance kernels.

The enhanced API in this module is deliberately profile-free.  Strict
Alphalens output projection is kept in :mod:`fincore.alphalens.performance`.
Task 4 is built in small, independently characterized function families; the
information-coefficient family is defined first so it can also be reused by
later model/report work without importing optional plotting dependencies.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd

from fincore.factor_analysis.calendar import get_forward_returns_columns


def _copy_factor_data(factor_data: pd.DataFrame) -> pd.DataFrame:
    """Validate and deep-copy a pre-cleaned two-level factor table."""

    if not isinstance(factor_data, pd.DataFrame):
        raise TypeError("factor_data must be a pandas DataFrame")
    if not isinstance(factor_data.index, pd.MultiIndex) or factor_data.index.nlevels != 2:
        raise ValueError("factor_data must use a two-level (date, asset) MultiIndex")
    if "factor" not in factor_data.columns:
        raise ValueError("factor_data must contain a 'factor' column")
    copied = factor_data.copy(deep=True)
    copied.index = copied.index.set_names(("date", "asset"))
    return copied


def _forward_columns(factor_data: pd.DataFrame) -> pd.Index:
    """Return the canonical forward-return columns without reading facade fixtures."""

    return get_forward_returns_columns(factor_data.columns)


def _date_key(factor_data: pd.DataFrame) -> pd.DatetimeIndex:
    """Use the first MultiIndex level as a named, stable date grouping key."""

    return pd.DatetimeIndex(factor_data.index.get_level_values(0), name="date")


def _demean_forward_returns(factor_data: pd.DataFrame, *, by_group: bool = False) -> pd.DataFrame:
    """Demean each forward period by date, optionally within group.

    This intentionally mirrors only the numerical transformation needed by
    Task 4.  It returns a fresh frame and never changes caller-owned data.
    """

    copied = _copy_factor_data(factor_data)
    columns = _forward_columns(copied)
    if not len(columns):
        return copied
    keys: list[object] = [_date_key(copied)]
    if by_group:
        if "group" not in copied.columns:
            raise ValueError("factor_data must contain a 'group' column when group_adjust=True")
        keys.append(copied["group"])
    means = copied.groupby(keys, observed=False, sort=True)[list(columns)].transform("mean")
    # Pandas 3 refuses an in-place float assignment into an int64 column.
    # Replace each forward-return column after an explicit float projection so
    # group adjustment remains non-mutating and valid for pinned integer data.
    demeaned = copied.loc[:, columns].astype(float).subtract(means.astype(float))
    for column in columns:
        copied[column] = demeaned[column]
    return copied


def factor_information_coefficient(
    factor_data: pd.DataFrame,
    group_adjust: bool = False,
    by_group: bool = False,
) -> pd.DataFrame:
    """Compute per-date Spearman IC values for every forward-return period."""

    copied = _copy_factor_data(factor_data)
    columns = _forward_columns(copied)
    if group_adjust:
        copied = _demean_forward_returns(copied, by_group=True)
    if by_group and "group" not in copied.columns:
        raise ValueError("factor_data must contain a 'group' column when by_group=True")

    keys: list[object] = [_date_key(copied)]
    names = ["date"]
    if by_group:
        keys.append(copied["group"])
        names.append("group")

    rows: list[dict[object, float]] = []
    index_values: list[object] = []
    for key, group in copied.groupby(keys, observed=False, sort=True):
        rows.append({column: group["factor"].corr(group[column], method="spearman") for column in columns})
        # Pandas 3 yields a one-tuple when the grouper is supplied as a
        # one-element list.  The strict output contract is a DatetimeIndex,
        # not an object index of singleton tuples.
        index_values.append(key if by_group else key[0] if isinstance(key, tuple) else key)

    index: pd.Index
    if by_group:
        tuples: list[tuple[Any, ...]] = [value if isinstance(value, tuple) else (value,) for value in index_values]
        index = pd.MultiIndex.from_tuples(tuples, names=names)
    else:
        index = pd.DatetimeIndex(index_values, name="date")
    return pd.DataFrame(rows, index=index, columns=columns, dtype=float)


def mean_information_coefficient(
    factor_data: pd.DataFrame,
    group_adjust: bool = False,
    by_group: bool = False,
    by_time: str | None = None,
) -> pd.Series | pd.DataFrame:
    """Aggregate IC by optional time and group dimensions."""

    information = factor_information_coefficient(
        factor_data,
        group_adjust=group_adjust,
        by_group=by_group,
    )
    if by_time is None and not by_group:
        return information.mean()
    if by_time is None:
        return information.groupby(level="group", observed=False, sort=True).mean()

    frequency = "ME" if by_time == "M" else by_time
    if by_group:
        grouped = information.reset_index().set_index("date")
        return grouped.groupby([pd.Grouper(freq=frequency), "group"], observed=False, sort=True).mean()
    return information.groupby(pd.Grouper(freq=frequency), observed=False, sort=True).mean()


def _to_weights(values: pd.Series, *, demeaned: bool, equal_weight: bool) -> pd.Series:
    """Compute one gross-normalized weight vector without changing ``values``."""

    weights = values.astype(float).copy(deep=True)
    if equal_weight:
        if demeaned:
            weights = weights - weights.median()
        negative = weights < 0
        positive = weights > 0
        weights.loc[negative] = -1.0
        weights.loc[positive] = 1.0
        if demeaned:
            if negative.any():
                weights.loc[negative] = weights.loc[negative] / negative.sum()
            if positive.any():
                weights.loc[positive] = weights.loc[positive] / positive.sum()
    elif demeaned:
        weights = weights - weights.mean()
    gross = weights.abs().sum(skipna=True)
    if gross == 0 or pd.isna(gross):
        return weights * np.nan
    return weights / gross


def _apply_groupwise_weights(
    values: pd.Series,
    keys: list[object],
    *,
    demeaned: bool,
    equal_weight: bool,
) -> pd.Series:
    """Apply the reviewed weight transformation once per date/group bucket."""

    result = pd.Series(np.nan, index=values.index, name="factor", dtype=float)
    for _, group in values.groupby(list(keys), observed=False, sort=True):
        result.loc[group.index] = _to_weights(group, demeaned=demeaned, equal_weight=equal_weight)
    return result


def factor_weights(
    factor_data: pd.DataFrame,
    demeaned: bool = True,
    group_adjust: bool = False,
    equal_weight: bool = False,
) -> pd.Series:
    """Build date-wise factor weights with optional group-neutral normalization."""

    copied = _copy_factor_data(factor_data)
    date_key = _date_key(copied)
    keys: list[object] = [date_key]
    if group_adjust:
        if "group" not in copied.columns:
            raise ValueError("factor_data must contain a 'group' column when group_adjust=True")
        keys.append(copied["group"])
    weights = _apply_groupwise_weights(
        copied["factor"],
        keys,
        demeaned=demeaned,
        equal_weight=equal_weight,
    )
    if group_adjust:
        weights = _apply_groupwise_weights(
            weights,
            [date_key],
            demeaned=False,
            equal_weight=False,
        )
    weights.name = "factor"
    return weights


def factor_returns(
    factor_data: pd.DataFrame,
    demeaned: bool = True,
    group_adjust: bool = False,
    equal_weight: bool = False,
    by_asset: bool = False,
) -> pd.DataFrame:
    """Compute weighted forward returns, optionally retaining individual assets.

    The aggregate path deliberately uses pandas' default sum projection: an
    all-missing weighted period becomes ``0.0`` for the date, matching the
    pinned strict surface and keeping the enhanced kernel profile-free.
    """

    copied = _copy_factor_data(factor_data)
    columns = _forward_columns(copied)
    weights = factor_weights(
        copied,
        demeaned=demeaned,
        group_adjust=group_adjust,
        equal_weight=equal_weight,
    )
    weighted = copied.loc[:, columns].multiply(weights, axis=0)
    if by_asset:
        return weighted
    return weighted.groupby(level="date", observed=False, sort=True).sum()


def factor_alpha_beta(
    factor_data: pd.DataFrame,
    returns: pd.DataFrame | pd.Series | None = None,
    demeaned: bool = True,
    group_adjust: bool = False,
    equal_weight: bool = False,
) -> pd.DataFrame:
    """Estimate annualized intercept and market beta from factor returns.

    The enhanced kernel uses a small NumPy least-squares projection.  The
    strict facade separately verifies the optional statsmodels boundary before
    delegating here, so importing this module remains optional-dependency free.
    """

    copied = _copy_factor_data(factor_data)
    columns = _forward_columns(copied)
    if returns is None:
        returns_frame = factor_returns(
            copied,
            demeaned=demeaned,
            group_adjust=group_adjust,
            equal_weight=equal_weight,
        )
    elif isinstance(returns, pd.Series):
        if len(columns) != 1:
            raise ValueError("a Series returns input requires exactly one forward-return column")
        returns_frame = returns.copy(deep=True).to_frame(columns[0])
    elif isinstance(returns, pd.DataFrame):
        returns_frame = returns.copy(deep=True)
    else:
        raise TypeError("returns must be a pandas Series, DataFrame, or None")

    universe = copied.groupby(level="date", observed=False, sort=True)[list(columns)].mean()
    universe = universe.reindex(returns_frame.index)
    result: dict[object, list[float]] = {}
    for period in returns_frame.columns:
        if period not in universe.columns:
            raise ValueError(f"returns column {period!r} is not a forward-return column")
        market = universe[period].to_numpy(dtype=float, copy=False)
        portfolio = returns_frame[period].to_numpy(dtype=float, copy=False)
        valid = np.isfinite(market) & np.isfinite(portfolio)
        if valid.sum() < 2:
            result[period] = [np.nan, np.nan]
            continue
        design = np.column_stack((np.ones(valid.sum()), market[valid]))
        intercept, beta = np.linalg.lstsq(design, portfolio[valid], rcond=None)[0]
        try:
            period_delta = pd.Timedelta(period)
            annualization = pd.Timedelta("252D") / period_delta
            annual_alpha = (1.0 + intercept) ** annualization - 1.0
        except (TypeError, ValueError, ZeroDivisionError):
            annual_alpha = np.nan
        result[period] = [float(annual_alpha), float(beta)]
    return pd.DataFrame(result, index=pd.Index(["Ann. alpha", "beta"]))


def quantile_turnover(quantile_factor: pd.Series, quantile: int, period: int = 1) -> pd.Series:
    """Return the fraction of names newly entering one quantile per date."""

    if not isinstance(quantile_factor, pd.Series):
        raise TypeError("quantile_factor must be a pandas Series")
    if not isinstance(quantile_factor.index, pd.MultiIndex) or quantile_factor.index.nlevels != 2:
        raise ValueError("quantile_factor must use a two-level (date, asset) MultiIndex")
    if not isinstance(period, int) or period <= 0:
        raise ValueError("period must be a positive integer")
    copied = quantile_factor.copy(deep=True)
    copied.index = copied.index.set_names(("date", "asset"))
    selected = copied[copied == quantile]
    if selected.empty:
        return pd.Series([], dtype=float, name=quantile)
    dates: list[object] = []
    names: list[set[object]] = []
    for date, group in selected.groupby(level="date", observed=False, sort=True):
        dates.append(date)
        names.append(set(group.index.get_level_values("asset")))
    selected_dates = pd.DatetimeIndex(dates, name="date")
    # ``groupby`` materializes scalar timestamps and loses the source level's
    # frequency.  When this quantile is represented on every source date, use
    # that public MultiIndex level directly, matching the pinned Series index
    # metadata for Day and BusinessDay calendars.
    multi_index = cast("pd.MultiIndex", copied.index)
    date_level_position = multi_index.names.index("date")
    source_dates = pd.DatetimeIndex(multi_index.levels[date_level_position], name="date")
    index = source_dates if selected_dates.equals(source_dates) else selected_dates
    turnover = pd.Series(np.nan, index=index, name=quantile, dtype=float)
    for position in range(period, len(names)):
        current = names[position]
        turnover.iloc[position] = len(current - names[position - period]) / len(current) if current else np.nan
    return turnover


def factor_rank_autocorrelation(factor_data: pd.DataFrame, period: int = 1) -> pd.Series:
    """Compute cross-sectional rank autocorrelation after a date-period shift."""

    if not isinstance(period, int) or period <= 0:
        raise ValueError("period must be a positive integer")
    copied = _copy_factor_data(factor_data)
    ranks = copied["factor"].groupby(level="date", observed=False, sort=True).rank()
    by_asset = ranks.unstack("asset")
    autocorrelation = by_asset.corrwith(by_asset.shift(period), axis=1)
    autocorrelation.name = period
    return autocorrelation


def cumulative_returns(returns: pd.Series | pd.DataFrame | np.ndarray) -> pd.Series | pd.DataFrame | np.ndarray:
    """Compound simple returns from one, treating missing observations as zero.

    This is the local, profile-free equivalent of the pinned Alphalens
    ``empyrical.cum_returns(..., starting_value=1)`` boundary.  It must not
    use fincore's validation-wrapped metric dispatcher because legacy factor
    return streams intentionally allow missing observations.
    """

    if isinstance(returns, pd.Series):
        return (returns.copy(deep=True).fillna(0.0) + 1.0).cumprod()
    if isinstance(returns, pd.DataFrame):
        return (returns.copy(deep=True).fillna(0.0) + 1.0).cumprod()
    source = np.array(returns, copy=True)
    source[np.isnan(source)] = 0.0
    return cast("np.ndarray", np.cumprod(source + 1.0, axis=0))


def mean_return_by_quantile(
    factor_data: pd.DataFrame,
    by_date: bool = False,
    by_group: bool = False,
    demeaned: bool = True,
    group_adjust: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return mean forward returns and standard errors by factor quantile."""

    copied = _copy_factor_data(factor_data)
    if "factor_quantile" not in copied.columns:
        raise ValueError("factor_data must contain a 'factor_quantile' column")
    if by_group and "group" not in copied.columns:
        raise ValueError("factor_data must contain a 'group' column when by_group=True")
    if group_adjust:
        copied = _demean_forward_returns(copied, by_group=True)
    elif demeaned:
        copied = _demean_forward_returns(copied, by_group=False)
    columns = _forward_columns(copied)
    groupers: list[object] = [copied["factor_quantile"], _date_key(copied)]
    if by_group:
        groupers.append(copied["group"])
    per_date = copied.groupby(groupers, observed=False, sort=True)[list(columns)].agg(["mean", "std", "count"])
    per_date_mean = per_date.xs("mean", axis=1, level=1)
    if by_date:
        mean_returns = cast("pd.DataFrame", per_date_mean)
        standard_error = cast(
            "pd.DataFrame",
            per_date.xs("std", axis=1, level=1) / np.sqrt(per_date.xs("count", axis=1, level=1)),
        )
        return mean_returns, standard_error

    aggregate_keys: list[object] = [per_date_mean.index.get_level_values("factor_quantile")]
    if by_group:
        aggregate_keys.append(per_date_mean.index.get_level_values("group"))
    aggregate = per_date_mean.groupby(aggregate_keys, observed=False, sort=True).agg(["mean", "std", "count"])
    mean_returns = cast("pd.DataFrame", aggregate.xs("mean", axis=1, level=1))
    standard_error = cast(
        "pd.DataFrame",
        aggregate.xs("std", axis=1, level=1) / np.sqrt(aggregate.xs("count", axis=1, level=1)),
    )
    return mean_returns, standard_error


def compute_mean_returns_spread(
    mean_returns: pd.DataFrame,
    upper_quant: int,
    lower_quant: int,
    std_err: pd.DataFrame | None = None,
) -> tuple[pd.Series | pd.DataFrame, pd.Series | pd.DataFrame | None]:
    """Compute the upper-minus-lower quantile mean-return spread and error."""

    if not isinstance(mean_returns, pd.DataFrame):
        raise TypeError("mean_returns must be a pandas DataFrame")

    def _quantile_slice(frame: pd.DataFrame, quantile: int) -> pd.Series | pd.DataFrame:
        if isinstance(frame.index, pd.MultiIndex):
            if "factor_quantile" not in frame.index.names:
                raise ValueError("a MultiIndex mean_returns input must contain a 'factor_quantile' level")
            return frame.xs(quantile, level="factor_quantile")
        if frame.index.name != "factor_quantile":
            raise ValueError("mean_returns must use a 'factor_quantile' index")
        return frame.loc[quantile]

    difference = _quantile_slice(mean_returns, upper_quant) - _quantile_slice(mean_returns, lower_quant)
    if std_err is None:
        return difference, None
    if not isinstance(std_err, pd.DataFrame):
        raise TypeError("std_err must be a pandas DataFrame or None")
    upper_error = _quantile_slice(std_err, upper_quant)
    lower_error = _quantile_slice(std_err, lower_quant)
    return difference, cast("pd.Series | pd.DataFrame", (upper_error**2 + lower_error**2) ** 0.5)


def _event_factor_copy(factor: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Validate/copy an event factor object carrying a two-level date/asset index."""

    if not isinstance(factor, (pd.Series, pd.DataFrame)):
        raise TypeError("factor must be a pandas Series or DataFrame")
    if not isinstance(factor.index, pd.MultiIndex) or factor.index.nlevels != 2:
        raise ValueError("factor must use a two-level (date, asset) MultiIndex")
    copied = factor.copy(deep=True)
    copied.index = copied.index.set_names(("date", "asset"))
    return copied


def _event_assets_at(factor: pd.Series | pd.DataFrame, timestamp: object) -> pd.Index:
    """Get the asset universe for one timestamp without ``.loc`` shape assumptions."""

    dates = factor.index.get_level_values("date")
    assets = factor.index.get_level_values("asset")
    return cast("pd.Index", pd.Index(assets[dates == timestamp]).unique())


def common_start_returns(
    factor: pd.Series | pd.DataFrame,
    returns: pd.DataFrame,
    before: int,
    after: int,
    cumulative: bool = False,
    mean_by_date: bool = False,
    demean_by: pd.Series | pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Align return windows around each factor date on a shared integer offset index."""

    if not isinstance(before, int) or not isinstance(after, int) or before < 0 or after < 0:
        raise ValueError("before and after must be non-negative integers")
    factor_copy = _event_factor_copy(factor)
    if not isinstance(returns, pd.DataFrame):
        raise TypeError("returns must be a pandas DataFrame")
    returns_copy = returns.copy(deep=True)
    returns_copy.index = pd.DatetimeIndex(returns_copy.index, name="date")
    if returns_copy.index.has_duplicates:
        raise ValueError("returns index must be unique")
    if not cumulative:
        returns_copy = pd.DataFrame(
            {column: cumulative_returns(returns_copy[column]) for column in returns_copy.columns},
            index=returns_copy.index,
        )
    demean_copy = _event_factor_copy(demean_by) if demean_by is not None else None
    windows: list[pd.Series | pd.DataFrame] = []
    for timestamp, group in factor_copy.groupby(level="date", observed=False, sort=True):
        try:
            day_zero = returns_copy.index.get_loc(timestamp)
        except KeyError:
            continue
        if not isinstance(day_zero, (int, np.integer)):
            raise ValueError("returns index must map each factor date to one row")
        start = max(int(day_zero) - before, 0)
        stop = min(int(day_zero) + after + 1, len(returns_copy.index))
        event_assets = pd.Index(group.index.get_level_values("asset")).unique()
        demean_assets = _event_assets_at(demean_copy, timestamp) if demean_copy is not None else pd.Index([])
        all_assets = event_assets.append(demean_assets.difference(event_assets))
        missing = all_assets.difference(returns_copy.columns)
        if not missing.empty:
            raise ValueError(f"returns do not contain factor assets: {list(missing)!r}")
        window = returns_copy.iloc[start:stop].loc[:, all_assets].copy()
        window.index = pd.RangeIndex(start - int(day_zero), stop - int(day_zero))
        if demean_copy is not None:
            if demean_assets.empty:
                raise ValueError("demean_by has no assets at a factor event date")
            mean = window.loc[:, demean_assets].mean(axis=1)
            window = window.loc[:, event_assets].sub(mean, axis=0)
        else:
            window = window.loc[:, event_assets]
        windows.append(window.mean(axis=1) if mean_by_date else window)
    if not windows:
        return pd.DataFrame(index=pd.RangeIndex(-before, after + 1))
    # Older pandas sorted the union of unequal event windows during concat.
    # Make that source-visible integer-offset order explicit under pandas 3.
    return pd.concat(windows, axis=1).sort_index()


def _average_event_window(
    event_factor: pd.Series,
    returns: pd.DataFrame,
    before: int,
    after: int,
    demean_by: pd.Series | None,
) -> pd.DataFrame:
    """Return the event-window mean/std pair for one quantile/group slice."""

    aligned = common_start_returns(
        event_factor,
        returns,
        before,
        after,
        cumulative=True,
        mean_by_date=True,
        demean_by=demean_by,
    )
    aligned = aligned.replace([np.inf, -np.inf], np.nan)
    return pd.DataFrame({"mean": aligned.mean(axis=1), "std": aligned.std(axis=1)}).T


def average_cumulative_return_by_quantile(
    factor_data: pd.DataFrame,
    returns: pd.DataFrame,
    periods_before: int = 10,
    periods_after: int = 15,
    demeaned: bool = True,
    group_adjust: bool = False,
    by_group: bool = False,
) -> pd.DataFrame:
    """Average cumulative event returns for each factor quantile (and group)."""

    data = _event_factor_copy(factor_data)
    if not isinstance(data, pd.DataFrame) or "factor_quantile" not in data.columns:
        raise ValueError("factor_data must contain a 'factor_quantile' column")
    if not isinstance(returns, pd.DataFrame):
        raise TypeError("returns must be a pandas DataFrame")
    if (group_adjust or by_group) and "group" not in data.columns:
        raise ValueError("factor_data must contain a 'group' column for group analytics")
    quantiles = sorted(data["factor_quantile"].dropna().unique())

    def _demean_source(group_data: pd.DataFrame) -> pd.Series | None:
        if group_adjust:
            return group_data["factor_quantile"]
        if demeaned:
            return data["factor_quantile"]
        return None

    if by_group:
        group_rows: list[pd.Series] = []
        group_row_index: list[tuple[object, str, object]] = []
        for group, group_data in data.groupby("group", observed=False, sort=True):
            for quantile in sorted(group_data["factor_quantile"].dropna().unique()):
                event = _average_event_window(
                    group_data.loc[group_data["factor_quantile"] == quantile, "factor_quantile"],
                    returns,
                    periods_before,
                    periods_after,
                    _demean_source(group_data),
                )
                for statistic in ("mean", "std"):
                    group_rows.append(cast("pd.Series", event.loc[statistic]))
                    group_row_index.append((cast("object", quantile), statistic, cast("object", group)))
        index = pd.MultiIndex.from_tuples(group_row_index, names=("factor_quantile", None, "group"))
        return pd.DataFrame(group_rows, index=index)

    rows: list[pd.Series] = []
    row_index: list[tuple[object, str]] = []
    for quantile in quantiles:
        event_factor = data.loc[data["factor_quantile"] == quantile, "factor_quantile"]
        if group_adjust:
            group_events: list[pd.DataFrame] = []
            for _, group_data in data.groupby("group", observed=False, sort=True):
                selected = group_data.loc[group_data["factor_quantile"] == quantile, "factor_quantile"]
                if not selected.empty:
                    group_events.append(
                        common_start_returns(
                            selected,
                            returns,
                            periods_before,
                            periods_after,
                            cumulative=True,
                            mean_by_date=True,
                            demean_by=group_data["factor_quantile"],
                        )
                    )
            aligned = pd.concat(group_events, axis=1) if group_events else pd.DataFrame()
            event = pd.DataFrame({"mean": aligned.mean(axis=1), "std": aligned.std(axis=1)}).T
        else:
            event = _average_event_window(
                event_factor,
                returns,
                periods_before,
                periods_after,
                data["factor_quantile"] if demeaned else None,
            )
        for statistic in ("mean", "std"):
            rows.append(cast("pd.Series", event.loc[statistic]))
            row_index.append((cast("object", quantile), statistic))
    index = pd.MultiIndex.from_tuples(row_index, names=("factor_quantile", None))
    return pd.DataFrame(rows, index=index)


__all__ = [
    "average_cumulative_return_by_quantile",
    "common_start_returns",
    "compute_mean_returns_spread",
    "cumulative_returns",
    "factor_alpha_beta",
    "factor_information_coefficient",
    "factor_rank_autocorrelation",
    "factor_returns",
    "factor_weights",
    "mean_information_coefficient",
    "mean_return_by_quantile",
    "quantile_turnover",
]
