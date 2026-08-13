"""Standalone factor-data preparation and quantization kernel.

This module deliberately has no compatibility-profile switch.  It normalizes
caller input once, computes forward returns, joins optional groups, and emits
structured loss diagnostics.  The Alphalens facade owns legacy stdout and
exception projection at its boundary.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd

from fincore.factor_analysis.calendar import (
    diff_custom_calendar_timedeltas,
    infer_trading_calendar,
    timedelta_to_string,
)
from fincore.factor_analysis.exceptions import FactorLossExceededError, NonMatchingTimezoneError


@dataclass(frozen=True)
class FactorLossReport:
    """A transparent breakdown of rows removed during factor preparation."""

    input_count: int
    finite_factor_count: int
    forward_returns_count: int
    binning_count: int
    factor_input_loss: float
    forward_returns_loss: float
    binning_loss: float
    total_loss: float

    @property
    def legacy_forward_returns_loss(self) -> float:
        """Loss wording used by the pinned strict facade's stdout projection."""

        if self.input_count == 0:
            return 0.0
        return (self.input_count - self.forward_returns_count) / self.input_count


@dataclass(frozen=True)
class PreparedFactorData:
    """Prepared factor table, loss accounting, and inferred trading calendar."""

    data: pd.DataFrame
    loss_report: FactorLossReport
    calendar: Any


def _require_factor_series(factor: pd.Series) -> pd.Series:
    """Copy and normalize a two-level ``(date, asset)`` factor series."""

    if not isinstance(factor, pd.Series):
        raise TypeError("factor must be a pandas Series indexed by date and asset")
    factor_copy = factor.copy(deep=True)
    if not isinstance(factor_copy.index, pd.MultiIndex) or factor_copy.index.nlevels != 2:
        raise ValueError("factor must use a two-level MultiIndex of date and asset")
    if factor_copy.index.has_duplicates:
        raise ValueError("factor index must be unique")
    dates = pd.DatetimeIndex(factor_copy.index.get_level_values(0))
    assets = pd.Index(factor_copy.index.get_level_values(1))
    factor_copy.index = pd.MultiIndex.from_arrays((dates, assets), names=("date", "asset"))
    try:
        factor_copy = pd.to_numeric(factor_copy, errors="raise").astype(float)
    except (TypeError, ValueError) as error:
        raise ValueError("factor values must be numeric") from error
    factor_copy.name = "factor"
    return factor_copy


def _require_prices(prices: pd.DataFrame, assets: pd.Index) -> pd.DataFrame:
    """Copy, validate, and sort a wide price table without mutating caller data."""

    if not isinstance(prices, pd.DataFrame):
        raise TypeError("prices must be a pandas DataFrame indexed by date")
    prices_copy = prices.copy(deep=True)
    if prices_copy.index.has_duplicates:
        raise ValueError("prices index must be unique")
    try:
        prices_copy.index = pd.DatetimeIndex(prices_copy.index, name="date")
    except (TypeError, ValueError) as error:
        raise ValueError("prices index must contain datetimes") from error
    if prices_copy.columns.has_duplicates:
        raise ValueError("prices columns must be unique")
    missing_assets = assets.unique().difference(prices_copy.columns)
    if not missing_assets.empty:
        raise ValueError(f"prices do not contain factor assets: {list(missing_assets)!r}")
    try:
        prices_copy = prices_copy.apply(pd.to_numeric, errors="raise").astype(float)
    except (TypeError, ValueError) as error:
        raise ValueError("prices values must be numeric") from error
    return cast("pd.DataFrame", prices_copy.sort_index())


def _factor_dates(factor: pd.Series) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(factor.index.get_level_values("date").unique()).sort_values()


def _period_label(prices_index: pd.DatetimeIndex, factor_dates: pd.DatetimeIndex, period: int, calendar: Any) -> str:
    """Find the first valid source-normalized interval for a return period."""

    for date in factor_dates:
        location = prices_index.get_indexer(pd.DatetimeIndex((date,)))[0]
        if location >= 0 and location + period < len(prices_index):
            return timedelta_to_string(
                diff_custom_calendar_timedeltas(prices_index[location], prices_index[location + period], calendar)
            )
    # A period without price buffer still has a deterministic fallback label.
    return f"{period}D"


def _calendar_date_index(dates: pd.DatetimeIndex, calendar: Any) -> pd.DatetimeIndex:
    """Attach an inferred offset where it is valid, without relying on mutation."""

    try:
        return pd.DatetimeIndex(dates, name="date", freq=calendar)
    except ValueError:
        return pd.DatetimeIndex(dates, name="date")


def compute_forward_returns(
    factor: pd.Series,
    prices: pd.DataFrame,
    periods: Sequence[int] = (1, 5, 10),
    filter_zscore: float | None = None,
    cumulative_returns: bool = True,
) -> pd.DataFrame:
    """Compute forward percentage returns aligned exactly to a factor MultiIndex."""

    factor_copy = _require_factor_series(factor)
    factor_dates = _factor_dates(factor_copy)
    prices_copy = _require_prices(prices, pd.Index(factor_copy.index.get_level_values("asset")))
    prices_index = pd.DatetimeIndex(prices_copy.index)
    if factor_dates.tz != prices_index.tz:
        raise NonMatchingTimezoneError(
            "The timezone of 'factor' is not the same as the timezone of 'prices'. "
            "See the pandas methods tz_localize and tz_convert."
        )
    if not periods or any(not isinstance(period, (int, np.integer)) or int(period) <= 0 for period in periods):
        raise ValueError("periods must contain positive integers")

    calendar = infer_trading_calendar(factor_dates, prices_index)
    shared_dates = pd.DatetimeIndex(factor_dates.intersection(prices_index).sort_values())
    if shared_dates.empty:
        raise ValueError(
            "Factor and prices indices don't match: make sure they have the same convention "
            "in terms of datetimes and symbol-names"
        )

    assets = pd.Index(factor_copy.index.get_level_values("asset").unique())
    selected_prices = cast("pd.DataFrame", prices_copy.reindex(columns=assets))
    labels: list[str] = []
    raw_values: dict[str, np.ndarray] = {}
    for raw_period in sorted({int(period) for period in periods}):
        base_returns = selected_prices.pct_change(raw_period if cumulative_returns else 1, fill_method=None)
        forward = base_returns.shift(-raw_period).reindex(shared_dates)
        if filter_zscore is not None:
            if not isinstance(filter_zscore, (int, float, np.number)) or not np.isfinite(filter_zscore):
                raise ValueError("filter_zscore must be a finite number or None")
            mean = forward.mean(axis=0)
            standard_deviation = forward.std(axis=0)
            forward = forward.mask((forward.sub(mean)).abs() > float(filter_zscore) * standard_deviation, np.nan)
        label = _period_label(prices_index, shared_dates, raw_period, calendar)
        if label in raw_values:
            raise ValueError(f"periods produce duplicate forward-return label {label!r}")
        labels.append(label)
        raw_values[label] = forward.to_numpy().reshape(-1)

    date_level = _calendar_date_index(shared_dates, calendar)
    product_index = pd.MultiIndex.from_product((date_level, assets), names=("date", "asset"))
    result = pd.DataFrame(raw_values, index=product_index).reindex(factor_copy.index)
    return cast("pd.DataFrame", result.loc[:, labels])


def _quantile_calculation(
    values: pd.Series,
    quantiles: int | Sequence[float] | None,
    bins: int | Sequence[float] | None,
    zero_aware: bool,
    no_raise: bool,
) -> pd.Series:
    try:
        if quantiles is not None and not zero_aware:
            return cast("pd.Series", pd.qcut(values, quantiles, labels=False) + 1)
        if bins is not None and not zero_aware:
            return cast("pd.Series", pd.cut(values, bins, labels=False) + 1)
        bucket_count = quantiles if quantiles is not None else bins
        assert isinstance(bucket_count, int)
        split = bucket_count // 2
        if quantiles is not None:
            positive = pd.qcut(values[values >= 0], split, labels=False) + split + 1
            negative = pd.qcut(values[values < 0], split, labels=False) + 1
        else:
            positive = pd.cut(values[values >= 0], split, labels=False) + split + 1
            negative = pd.cut(values[values < 0], split, labels=False) + 1
        return cast("pd.Series", pd.concat((positive, negative)).sort_index())
    except Exception:
        if no_raise:
            return pd.Series(np.nan, index=values.index, dtype=float)
        raise


def quantize_factor(
    factor_data: pd.DataFrame,
    quantiles: int | Sequence[float] | None = 5,
    bins: int | Sequence[float] | None = None,
    by_group: bool = False,
    no_raise: bool = False,
    zero_aware: bool = False,
) -> pd.Series:
    """Assign one period-wise quantile/bin label per finite factor observation."""

    if not isinstance(factor_data, pd.DataFrame):
        raise TypeError("factor_data must be a pandas DataFrame")
    if "factor" not in factor_data.columns:
        raise ValueError("factor_data must contain a 'factor' column")
    if not isinstance(factor_data.index, pd.MultiIndex) or factor_data.index.nlevels != 2:
        raise ValueError("factor_data must use a two-level MultiIndex")
    if not ((quantiles is not None and bins is None) or (quantiles is None and bins is not None)):
        raise ValueError("Either quantiles or bins should be provided")
    zero_aware_bucket = quantiles if quantiles is not None else bins
    if zero_aware:
        if not isinstance(zero_aware_bucket, int):
            raise ValueError("zero_aware should only be True when quantiles or bins is an integer")
        if zero_aware_bucket < 2:
            raise ValueError("zero_aware requires at least two quantiles or bins")
    if by_group and "group" not in factor_data.columns:
        raise ValueError("factor_data must contain a 'group' column when by_group=True")

    data = factor_data.copy(deep=True)
    data.index = data.index.set_names(("date", "asset"))
    groupers: list[Any] = [data.index.get_level_values("date")]
    if by_group:
        groupers.append(data["group"])
    result = data.groupby(groupers, observed=False, group_keys=False, sort=False)["factor"].apply(
        _quantile_calculation,
        quantiles=quantiles,
        bins=bins,
        zero_aware=zero_aware,
        no_raise=no_raise,
    )
    result.name = "factor_quantile"
    return result.dropna()


def _normalize_groupby(
    groupby: Mapping[Hashable, Hashable] | pd.Series | None,
    factor_index: pd.MultiIndex,
    groupby_labels: Mapping[Hashable, Hashable] | None,
) -> pd.Series | None:
    if groupby is None:
        return None
    assets = pd.Index(factor_index.get_level_values("asset"))
    if isinstance(groupby, Mapping):
        mapping = pd.Series(dict(groupby), dtype=object)
        missing = assets.unique().difference(mapping.index)
        if not missing.empty:
            raise KeyError(f"Assets {list(missing)!r} not in group mapping")
        values = mapping.reindex(assets).to_numpy()
        groups = pd.Series(values, index=factor_index, name="group")
    elif isinstance(groupby, pd.Series):
        group_copy = groupby.copy(deep=True)
        if isinstance(group_copy.index, pd.MultiIndex):
            groups = group_copy.reindex(factor_index)
            groups.index = factor_index
            groups.name = "group"
        else:
            missing = assets.unique().difference(group_copy.index)
            if not missing.empty:
                raise KeyError(f"Assets {list(missing)!r} not in group mapping")
            groups = pd.Series(group_copy.reindex(assets).to_numpy(), index=factor_index, name="group")
    else:
        raise TypeError("groupby must be a mapping, Series, or None")
    if groupby_labels is not None:
        labels = pd.Series(dict(groupby_labels), dtype=object)
        valid_groups = pd.Index(groups.dropna().unique())
        missing_labels = valid_groups.difference(labels.index)
        if not missing_labels.empty:
            raise KeyError(f"groups {list(missing_labels)!r} not in passed group names")
        groups = pd.Series(labels.reindex(groups).to_numpy(), index=factor_index, name="group")
    return groups.astype("category")


def _loss_report(input_count: int, finite_count: int, forward_count: int, binning_count: int) -> FactorLossReport:
    if input_count <= 0:
        raise ValueError("factor must contain at least one observation")
    factor_loss = (input_count - finite_count) / input_count
    forward_loss = (finite_count - forward_count) / input_count
    binning_loss = (forward_count - binning_count) / input_count
    return FactorLossReport(
        input_count=input_count,
        finite_factor_count=finite_count,
        forward_returns_count=forward_count,
        binning_count=binning_count,
        factor_input_loss=factor_loss,
        forward_returns_loss=forward_loss,
        binning_loss=binning_loss,
        total_loss=factor_loss + forward_loss + binning_loss,
    )


def _prepare_from_forward_returns(
    factor: pd.Series,
    forward_returns: pd.DataFrame,
    *,
    groupby: Mapping[Hashable, Hashable] | pd.Series | None = None,
    binning_by_group: bool = False,
    quantiles: int | Sequence[float] | None = 5,
    bins: int | Sequence[float] | None = None,
    groupby_labels: Mapping[Hashable, Hashable] | None = None,
    max_loss: float = 0.35,
    zero_aware: bool = False,
    calendar: Any = None,
) -> PreparedFactorData:
    """Internal shared cleaner for direct and price-derived public pathways."""

    if not isinstance(max_loss, (int, float)) or not 0 <= float(max_loss) <= 1:
        raise ValueError("max_loss must be a number between 0 and 1")
    factor_copy = _require_factor_series(factor)
    if not isinstance(forward_returns, pd.DataFrame) or not isinstance(forward_returns.index, pd.MultiIndex):
        raise TypeError("forward_returns must be a MultiIndex pandas DataFrame")
    forward_copy = forward_returns.copy(deep=True)
    forward_copy.index = forward_copy.index.set_names(("date", "asset"))
    input_count = len(factor_copy)
    finite_mask = np.isfinite(factor_copy.to_numpy(dtype=float, copy=False))
    finite_factor = factor_copy.loc[finite_mask]
    if finite_factor.empty:
        raise ValueError("factor must contain at least one finite observation")
    groups = _normalize_groupby(groupby, finite_factor.index, groupby_labels)

    merged = forward_copy.reindex(factor_copy.index).copy()
    merged["factor"] = finite_factor
    if groups is not None:
        merged["group"] = groups
    merged = merged.dropna()
    forward_count = len(merged)
    quantiles_result = quantize_factor(
        merged,
        quantiles=quantiles,
        bins=bins,
        by_group=binning_by_group,
        no_raise=float(max_loss) != 0,
        zero_aware=zero_aware,
    )
    merged["factor_quantile"] = quantiles_result
    merged = merged.dropna()
    merged.index = merged.index.set_names(("date", "asset"))
    report = _loss_report(input_count, len(finite_factor), forward_count, len(merged))
    if report.total_loss > float(max_loss):
        message = (
            f"max_loss ({float(max_loss) * 100:.1f}%) exceeded {report.total_loss * 100:.1f}%, consider increasing it."
        )
        raise FactorLossExceededError(message, report)
    return PreparedFactorData(
        data=merged.copy(deep=True),
        loss_report=report,
        calendar=calendar
        if calendar is not None
        else infer_trading_calendar(_factor_dates(factor_copy), _factor_dates(factor_copy)),
    )


def prepare_factor_data(
    factor: pd.Series,
    prices: pd.DataFrame,
    *,
    groupby: Mapping[Hashable, Hashable] | pd.Series | None = None,
    quantiles: int | Sequence[float] | None = 5,
    bins: int | Sequence[float] | None = None,
    periods: Sequence[int] = (1, 5, 10),
    max_loss: float = 0.35,
    binning_by_group: bool = False,
    filter_zscore: float | None = None,
    groupby_labels: Mapping[Hashable, Hashable] | None = None,
    zero_aware: bool = False,
    cumulative_returns: bool = True,
) -> PreparedFactorData:
    """Normalize factor/prices and return a cleaned factor table with loss detail."""

    factor_copy = _require_factor_series(factor)
    prices_copy = _require_prices(prices, pd.Index(factor_copy.index.get_level_values("asset")))
    forward_returns = compute_forward_returns(
        factor_copy,
        prices_copy,
        periods=periods,
        filter_zscore=filter_zscore,
        cumulative_returns=cumulative_returns,
    )
    calendar = infer_trading_calendar(_factor_dates(factor_copy), prices_copy.index)
    return _prepare_from_forward_returns(
        factor_copy,
        forward_returns,
        groupby=groupby,
        binning_by_group=binning_by_group,
        quantiles=quantiles,
        bins=bins,
        groupby_labels=groupby_labels,
        max_loss=max_loss,
        zero_aware=zero_aware,
        calendar=calendar,
    )


def prepare_factor_data_from_forward_returns(
    factor: pd.Series,
    forward_returns: pd.DataFrame,
    *,
    groupby: Mapping[Hashable, Hashable] | pd.Series | None = None,
    binning_by_group: bool = False,
    quantiles: int | Sequence[float] | None = 5,
    bins: int | Sequence[float] | None = None,
    groupby_labels: Mapping[Hashable, Hashable] | None = None,
    max_loss: float = 0.35,
    zero_aware: bool = False,
) -> PreparedFactorData:
    """Prepare a factor table when forward returns have already been computed."""

    return _prepare_from_forward_returns(
        factor,
        forward_returns,
        groupby=groupby,
        binning_by_group=binning_by_group,
        quantiles=quantiles,
        bins=bins,
        groupby_labels=groupby_labels,
        max_loss=max_loss,
        zero_aware=zero_aware,
    )


__all__ = [
    "FactorLossExceededError",
    "FactorLossReport",
    "PreparedFactorData",
    "compute_forward_returns",
    "prepare_factor_data",
    "prepare_factor_data_from_forward_returns",
    "quantize_factor",
]
