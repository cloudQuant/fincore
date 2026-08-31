"""Compute-once assembly of enhanced factor-analysis models.

This module consumes already-clean factor data.  It intentionally does not
re-enter the raw factor/price preparation path and does not import a plotting
library; later renderers receive only :class:`FactorAnalysisModel` data.
"""

from __future__ import annotations

import operator
from collections.abc import Hashable, Sequence
from typing import Any, cast

import numpy as np
import pandas as pd

import fincore.factor_analysis.performance as performance
import fincore.factor_analysis.portfolio as portfolio
from fincore.factor_analysis.calendar import get_forward_returns_columns
from fincore.factor_analysis.models import (
    EventAnalysisModel,
    FactorAnalysisConfig,
    FactorAnalysisModel,
    FactorGroupAnalysis,
    fingerprint_value,
    frozen_mapping,
    snapshot_pandas,
)


def _copy_clean_factor_data(factor_data: pd.DataFrame) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Validate and own a clean-factor snapshot without performing cleaning."""

    if not isinstance(factor_data, pd.DataFrame):
        raise TypeError("factor_data must be a pandas DataFrame")
    if not isinstance(factor_data.index, pd.MultiIndex) or factor_data.index.nlevels != 2:
        raise ValueError("factor_data must use a two-level (date, asset) MultiIndex")
    if "factor" not in factor_data.columns:
        raise ValueError("factor_data must contain a 'factor' column")
    if "factor_quantile" not in factor_data.columns:
        raise ValueError("factor_data must contain a 'factor_quantile' column")
    snapshot = cast("pd.DataFrame", snapshot_pandas(factor_data))
    forward_columns = get_forward_returns_columns(snapshot.columns)
    if not len(forward_columns):
        raise ValueError("factor_data must contain at least one forward-return column")
    if not all(isinstance(period, str) for period in forward_columns):
        raise ValueError("forward-return labels must use canonical timedelta strings")
    return snapshot, tuple(cast("str", period) for period in forward_columns)


def _normalize_periods(available: tuple[str, ...], periods: Sequence[str] | None) -> tuple[str, ...]:
    """Validate a deterministic nonempty forward-period selection."""

    if periods is not None and (isinstance(periods, (str, bytes, bytearray)) or not isinstance(periods, Sequence)):
        raise TypeError("periods must be a sequence")
    selected = available if periods is None else tuple(periods)
    if not selected:
        raise ValueError("at least one forward period must be selected")
    if len(set(selected)) != len(selected):
        raise ValueError("forward period selection must not contain duplicates")
    unknown = tuple(period for period in selected if period not in available)
    if unknown:
        raise ValueError(f"unknown forward periods: {list(unknown)!r}")
    return selected


def _normalize_positive_lags(
    turnover_periods: Sequence[int],
    *,
    allow_legacy_zero: bool = False,
) -> tuple[int, ...]:
    """Validate turnover/rank lags, with one private strict legacy escape hatch."""

    if isinstance(turnover_periods, (str, bytes, bytearray)) or not isinstance(turnover_periods, Sequence):
        raise TypeError("turnover_periods must be a sequence")
    normalized = tuple(turnover_periods)
    if not normalized:
        raise ValueError("turnover_periods must contain at least one positive lag")
    if any(
        not isinstance(period, int) or isinstance(period, bool) or (not allow_legacy_zero and period < 1)
        for period in normalized
    ):
        raise ValueError("turnover_periods must contain positive integers")
    if len(set(normalized)) != len(normalized):
        raise ValueError("turnover_periods must not contain duplicates")
    return normalized


def _normalize_time_aggregation(time_aggregation: Sequence[str]) -> tuple[str, ...]:
    """Validate named pandas aggregation frequencies for the model payload."""

    if isinstance(time_aggregation, (str, bytes, bytearray)) or not isinstance(time_aggregation, Sequence):
        raise TypeError("time_aggregation must be a sequence")
    normalized = tuple(time_aggregation)
    if any(not isinstance(frequency, str) or not frequency for frequency in normalized):
        raise ValueError("time_aggregation must contain nonempty frequency strings")
    if len(set(normalized)) != len(normalized):
        raise ValueError("time_aggregation must not contain duplicates")
    return normalized


def _with_selected_periods(snapshot: pd.DataFrame, selected: tuple[str, ...]) -> pd.DataFrame:
    """Build the internal computation table without changing the owned snapshot."""

    forward_columns = get_forward_returns_columns(snapshot.columns)
    return snapshot.drop(columns=[column for column in forward_columns if column not in selected]).copy(deep=True)


def _quantile_statistics(factor_data: pd.DataFrame) -> pd.DataFrame:
    """Return the renderer-ready quantile statistics table from one snapshot."""

    statistics = factor_data.groupby("factor_quantile", observed=False, sort=True)["factor"].agg(
        ["min", "max", "mean", "std", "count"]
    )
    total = statistics["count"].sum()
    statistics["count %"] = statistics["count"] / total * 100.0 if total else np.nan
    return statistics


def _as_spread_frame(value: pd.Series | pd.DataFrame, name: str) -> pd.DataFrame:
    """Normalize a scalar-period spread to the model's table-only contract."""

    return value.to_frame(name) if isinstance(value, pd.Series) else value.copy(deep=True)


def _quantile_cumulative_returns(mean_by_date: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Precompute quantile growth curves so renderers never re-enter a kernel."""

    if not len(mean_by_date.columns):
        return {}
    base_period = mean_by_date.columns[0]
    results: dict[str, pd.DataFrame] = {}
    for period in mean_by_date.columns:
        values = mean_by_date[period]
        converted = values.add(1.0).pow(pd.Timedelta(base_period) / pd.Timedelta(period)).sub(1.0)
        wide = converted.unstack("factor_quantile")
        results[cast("str", period)] = wide.fillna(0.0).add(1.0).cumprod().loc[:, ::-1]
    return results


def _mean_information(information: pd.DataFrame, *, by_group: bool) -> pd.Series | pd.DataFrame:
    """Aggregate an already-computed IC table without invoking the IC kernel again."""

    if by_group and isinstance(information.index, pd.MultiIndex) and "group" in information.index.names:
        return information.groupby(level="group", observed=False, sort=True).mean()
    return information.mean()


def _time_aggregated_information(
    information: pd.DataFrame,
    frequencies: tuple[str, ...],
    *,
    by_group: bool,
) -> dict[str, pd.Series | pd.DataFrame]:
    """Derive time summaries from the one IC snapshot, never by recomputation."""

    results: dict[str, pd.Series | pd.DataFrame] = {}
    if information.empty and by_group:
        # ``reset_index().set_index('date')`` loses the datetime dtype for an
        # empty MultiIndex under pandas 3.  Retain a typed empty result so an
        # otherwise valid all-missing group model can reach the renderer's
        # source-shaped empty/error policy.
        empty_index = pd.MultiIndex.from_arrays(
            [pd.DatetimeIndex([], name="date"), pd.Index([], name="group")],
            names=("date", "group"),
        )
        for frequency in frequencies:
            results[frequency] = pd.DataFrame(index=empty_index, columns=information.columns, dtype=float)
        return results
    for frequency in frequencies:
        normalized_frequency = "ME" if frequency == "M" else frequency
        if by_group and isinstance(information.index, pd.MultiIndex) and "group" in information.index.names:
            indexed = information.reset_index().set_index("date")
            results[frequency] = indexed.groupby(
                [pd.Grouper(freq=normalized_frequency), "group"], observed=False, sort=True
            ).mean()
        else:
            date_index = pd.DatetimeIndex(information.index, name="date")
            results[frequency] = (
                information.set_axis(date_index, axis="index")
                .groupby(pd.Grouper(freq=normalized_frequency), observed=False, sort=True)
                .mean()
            )
    return results


def _quantile_values(factor_data: pd.DataFrame) -> tuple[Hashable, ...]:
    """Return stable, nonmissing quantile labels for spread and turnover tables."""

    values = factor_data["factor_quantile"].dropna().unique().tolist()
    return tuple(cast("Hashable", value) for value in sorted(values))


def _legacy_unvalidated_quantile_turnover(
    factor_data: pd.DataFrame,
    quantile: int,
    period: int,
) -> pd.Series:
    """Replay pinned turnover's unvalidated zero and negative shift grammar."""

    quantile_factor = factor_data["factor_quantile"]
    quantile_names = quantile_factor[quantile_factor == quantile]
    sets = quantile_names.groupby(level="date", observed=False, sort=True).apply(
        lambda values: set(values.index.get_level_values("asset"))
    )
    shifted = sets.shift(period)
    new_names = (sets - shifted).dropna()
    turnover = new_names.apply(len) / sets.apply(len)
    turnover.name = quantile
    return turnover


def _legacy_unvalidated_rank_autocorrelation(factor_data: pd.DataFrame, period: int) -> pd.Series:
    """Replay source rank autocorrelation's unvalidated zero/negative shift."""

    ranks = factor_data.groupby(level="date", observed=False, sort=True)["factor"].rank()
    by_asset = ranks.reset_index().pivot(index="date", columns="asset", values="factor")
    result = by_asset.corrwith(by_asset.shift(period), axis=1)
    result.name = period
    return result


def _quantile_turnover_tables(
    factor_data: pd.DataFrame,
    lags: tuple[int, ...],
    *,
    allow_legacy_zero: bool = False,
    legacy_quantiles: tuple[int, ...] | None = None,
) -> dict[int, pd.DataFrame]:
    """Assemble typed quantile turnover tables from per-quantile kernel Series."""

    quantiles = _quantile_values(factor_data) if legacy_quantiles is None else legacy_quantiles
    results: dict[int, pd.DataFrame] = {}
    for lag in lags:
        columns = {
            quantile: (
                _legacy_unvalidated_quantile_turnover(factor_data, cast("int", quantile), lag)
                if lag <= 0 and allow_legacy_zero
                else performance.quantile_turnover(factor_data["factor_quantile"], cast("int", quantile), period=lag)
            )
            for quantile in quantiles
        }
        results[lag] = pd.DataFrame(columns)
    return results


def _rank_autocorrelation_table(
    factor_data: pd.DataFrame,
    lags: tuple[int, ...],
    *,
    allow_legacy_zero: bool = False,
) -> pd.DataFrame:
    """Collect all requested rank autocorrelation lags in one table."""

    return pd.DataFrame(
        {
            lag: (
                _legacy_unvalidated_rank_autocorrelation(factor_data, lag)
                if lag <= 0 and allow_legacy_zero
                else performance.factor_rank_autocorrelation(factor_data, period=lag)
            )
            for lag in lags
        }
    )


def _event_model(
    factor_data: pd.DataFrame,
    event_returns: pd.DataFrame | None,
    *,
    event_before: int | None,
    event_after: int | None,
    long_short: bool,
    group_neutral: bool,
    by_group: bool,
    allow_legacy_event_windows: bool = False,
) -> tuple[EventAnalysisModel | None, pd.DataFrame | None]:
    """Build event data only when a complete optional event input is present."""

    if event_returns is None:
        return None, None
    if not isinstance(event_returns, pd.DataFrame):
        raise TypeError("event_returns must be a pandas DataFrame")
    # Even an incomplete event request belongs in the model provenance: a
    # later caller must not receive the same result fingerprint after changing
    # supplied event data merely because no event section was requested yet.
    returns_snapshot = cast("pd.DataFrame", snapshot_pandas(event_returns))
    if event_before is None or event_after is None:
        return None, returns_snapshot
    if not allow_legacy_event_windows and (
        not isinstance(event_before, int) or not isinstance(event_after, int) or event_before < 0 or event_after < 0
    ):
        raise ValueError("event_before and event_after must be non-negative integers")
    demean_by = factor_data["factor_quantile"] if long_short else None
    common_start_returns = (
        performance._common_start_returns if allow_legacy_event_windows else performance.common_start_returns
    )
    average_cumulative_returns = (
        performance._average_cumulative_return_by_quantile
        if allow_legacy_event_windows
        else performance.average_cumulative_return_by_quantile
    )
    legacy_window_kwargs: dict[str, bool] = {"_allow_legacy_event_windows": True} if allow_legacy_event_windows else {}
    windows = common_start_returns(
        factor_data["factor_quantile"],
        returns_snapshot,
        event_before,
        event_after,
        cumulative=True,
        mean_by_date=True,
        demean_by=demean_by,
        **legacy_window_kwargs,
    )
    average = average_cumulative_returns(
        factor_data,
        returns_snapshot,
        periods_before=event_before,
        periods_after=event_after,
        demeaned=long_short,
        group_adjust=group_neutral,
        by_group=by_group,
        **legacy_window_kwargs,
    )
    aggregate_average = (
        average_cumulative_returns(
            factor_data,
            returns_snapshot,
            periods_before=event_before,
            periods_after=event_after,
            demeaned=long_short,
            group_adjust=group_neutral,
            by_group=False,
            **legacy_window_kwargs,
        )
        if by_group
        else average
    )
    distribution = windows.stack().dropna()
    if isinstance(distribution, pd.DataFrame):
        distribution = distribution.stack().dropna()
    if not isinstance(distribution, pd.Series):  # pragma: no cover - pandas contract guard
        raise TypeError("event return distribution must reduce to a pandas Series")
    return (
        EventAnalysisModel(
            event_windows=windows.copy(deep=True),
            mean_returns=windows.mean(axis=1),
            return_distribution=distribution,
            quantile_average_returns=average.copy(deep=True),
            aggregate_quantile_average_returns=aggregate_average.copy(deep=True),
        ),
        returns_snapshot,
    )


def _group_analysis(
    factor_data: pd.DataFrame,
    group: Hashable,
    *,
    long_short: bool,
    equal_weight: bool,
    lags: tuple[int, ...],
) -> FactorGroupAnalysis:
    """Compute the strongly typed, renderer-ready model for one named group."""

    group_data = factor_data.loc[factor_data["group"] == group].copy(deep=True)
    weights = performance.factor_weights(group_data, demeaned=long_short, equal_weight=equal_weight)
    returns = performance.factor_returns(group_data, demeaned=long_short, equal_weight=equal_weight)
    mean_returns, std_error = performance.mean_return_by_quantile(group_data, demeaned=long_short)
    information = performance.factor_information_coefficient(group_data)
    turnover = _quantile_turnover_tables(group_data, lags)
    return FactorGroupAnalysis(
        group=group,
        quantile_statistics=_quantile_statistics(group_data),
        factor_weights=weights.to_frame("factor"),
        factor_returns=returns,
        mean_returns_by_quantile=mean_returns,
        std_error_by_quantile=std_error,
        information_coefficient=information,
        mean_information_coefficient=information.mean(),
        quantile_turnover=frozen_mapping(turnover),
        rank_autocorrelation=_rank_autocorrelation_table(group_data, lags),
    )


def _analyze_factor(
    factor_data: pd.DataFrame,
    *,
    long_short: bool = True,
    group_neutral: bool = False,
    equal_weight: bool = False,
    by_group: bool = False,
    periods: Sequence[str] | None = None,
    turnover_periods: Sequence[int] = (1,),
    time_aggregation: Sequence[str] = ("M",),
    include_portfolio_inputs: bool = True,
    portfolio_capital: int | float | None = None,
    portfolio_benchmark_period: str = "1D",
    event_returns: pd.DataFrame | None = None,
    event_before: int | None = None,
    event_after: int | None = None,
    allow_legacy_zero_turnover: bool = False,
    legacy_turnover_quantiles: tuple[int, ...] | None = None,
    allow_legacy_event_windows: bool = False,
) -> FactorAnalysisModel:
    """Compute one immutable, renderer-ready model from already-clean data.

    ``factor_data`` must have crossed :func:`prepare_factor_data` (or the
    equivalent forward-return preparation boundary) already.  This function
    owns a deep input snapshot, invokes analytical kernels during assembly,
    and leaves later consumers to read fields rather than recalculate them.
    """

    snapshot, available_periods = _copy_clean_factor_data(factor_data)
    selected_periods = _normalize_periods(available_periods, periods)
    lags = _normalize_positive_lags(turnover_periods, allow_legacy_zero=allow_legacy_zero_turnover)
    aggregations = _normalize_time_aggregation(time_aggregation)
    if not isinstance(portfolio_benchmark_period, str):
        raise TypeError("portfolio_benchmark_period must be a string")
    if portfolio_capital is not None and not isinstance(portfolio_capital, (int, float)):
        raise TypeError("portfolio_capital must be a number or None")

    normalized_event_before = event_before
    normalized_event_after = event_after
    if allow_legacy_event_windows:
        try:
            normalized_event_before = None if event_before is None else operator.index(event_before)
            normalized_event_after = None if event_after is None else operator.index(event_after)
        except TypeError as error:
            raise ValueError("event_before and event_after must be non-negative integers") from error
    # The frozen enhanced config deliberately rejects signed event windows.
    # Strict tear sheets need the pinned source's permissive indexing grammar,
    # but never expose this private assembly model, so retain a valid config
    # snapshot while carrying the actual signed offsets into event assembly.
    config_event_before = normalized_event_before
    config_event_after = normalized_event_after
    if allow_legacy_event_windows:
        if config_event_before is not None and config_event_before < 0:
            config_event_before = 0
        if config_event_after is not None and config_event_after < 0:
            config_event_after = 0

    config = FactorAnalysisConfig(
        long_short=long_short,
        group_neutral=group_neutral,
        equal_weight=equal_weight,
        by_group=by_group,
        periods=selected_periods,
        event_before=config_event_before,
        event_after=config_event_after,
        turnover_periods=lags,
        time_aggregation=aggregations,
        include_portfolio_inputs=include_portfolio_inputs,
        portfolio_capital=portfolio_capital,
        portfolio_benchmark_period=portfolio_benchmark_period,
    )
    analysis_data = _with_selected_periods(snapshot, selected_periods)
    has_group = "group" in analysis_data.columns
    effective_by_group = by_group and has_group

    weights = performance.factor_weights(
        analysis_data,
        demeaned=long_short,
        group_adjust=group_neutral,
        equal_weight=equal_weight,
    )
    factor_returns = performance.factor_returns(
        analysis_data,
        demeaned=long_short,
        group_adjust=group_neutral,
        equal_weight=equal_weight,
    )
    alpha_beta = performance.factor_alpha_beta(
        analysis_data,
        returns=factor_returns,
        demeaned=long_short,
        group_adjust=group_neutral,
        equal_weight=equal_weight,
    )
    mean_returns, std_error = performance.mean_return_by_quantile(
        analysis_data,
        by_group=effective_by_group,
        demeaned=long_short,
        group_adjust=group_neutral,
    )
    mean_by_date, std_error_by_date = performance.mean_return_by_quantile(
        analysis_data,
        by_date=True,
        by_group=effective_by_group,
        demeaned=long_short,
        group_adjust=group_neutral,
    )
    # A by-group model also owns the source workflow's aggregate snapshots.
    # They cannot be recovered by averaging group means when group sizes differ,
    # and retaining both lets renderers stay compute-free after assembly.
    if effective_by_group:
        aggregate_mean_returns, aggregate_std_error = performance.mean_return_by_quantile(
            analysis_data,
            by_group=False,
            demeaned=long_short,
            group_adjust=group_neutral,
        )
        aggregate_mean_by_date, aggregate_std_error_by_date = performance.mean_return_by_quantile(
            analysis_data,
            by_date=True,
            by_group=False,
            demeaned=long_short,
            group_adjust=group_neutral,
        )
    else:
        aggregate_mean_returns = mean_returns
        aggregate_std_error = std_error
        aggregate_mean_by_date = mean_by_date
        aggregate_std_error_by_date = std_error_by_date
    quantiles = _quantile_values(analysis_data)
    # A requested by-group view can legitimately have no observed groups
    # (for example an all-missing group column).  Keep its empty analytical
    # snapshots constructible; strict source renderers then retain their own
    # chart/error projection rather than failing model assembly on a missing
    # quantile slice.
    if len(quantiles) >= 2 and not mean_returns.empty:
        spread, spread_std = performance.compute_mean_returns_spread(
            mean_returns,
            cast("int", quantiles[-1]),
            cast("int", quantiles[0]),
            std_error,
        )
        mean_spread = _as_spread_frame(spread, "spread")
        mean_spread_std = None if spread_std is None else _as_spread_frame(spread_std, "spread")
    else:
        mean_spread = pd.DataFrame(columns=selected_periods, dtype=float)
        mean_spread_std = None

    if len(quantiles) >= 2:
        aggregate_spread, aggregate_spread_std = performance.compute_mean_returns_spread(
            aggregate_mean_returns,
            cast("int", quantiles[-1]),
            cast("int", quantiles[0]),
            aggregate_std_error,
        )
        aggregate_mean_spread = _as_spread_frame(aggregate_spread, "spread")
        aggregate_mean_spread_std = (
            None if aggregate_spread_std is None else _as_spread_frame(aggregate_spread_std, "spread")
        )
    else:
        aggregate_mean_spread = pd.DataFrame(columns=selected_periods, dtype=float)
        aggregate_mean_spread_std = None

    # Keep the requested group view plus one aggregate snapshot when needed.
    # All later tables/renderers consume these stored values rather than
    # re-entering the analytical kernels.
    information = performance.factor_information_coefficient(
        analysis_data,
        group_adjust=group_neutral,
        by_group=effective_by_group,
    )
    aggregate_information = (
        performance.factor_information_coefficient(
            analysis_data,
            group_adjust=group_neutral,
            by_group=False,
        )
        if effective_by_group
        else information
    )
    mean_information = _mean_information(information, by_group=effective_by_group)
    time_aggregated = _time_aggregated_information(
        information,
        aggregations,
        by_group=effective_by_group,
    )
    aggregate_mean_information = _mean_information(aggregate_information, by_group=False)
    aggregate_time_aggregated = _time_aggregated_information(
        aggregate_information,
        aggregations,
        by_group=False,
    )
    # The legacy summary table intentionally ignores ``group_neutral`` for
    # IC.  Retain that source-specific aggregate once during assembly rather
    # than making the strict renderer re-enter a numerical kernel.
    summary_information = (
        performance.factor_information_coefficient(analysis_data, group_adjust=False, by_group=False)
        if group_neutral
        else aggregate_information
    )
    turnover = _quantile_turnover_tables(
        analysis_data,
        lags,
        allow_legacy_zero=allow_legacy_zero_turnover,
        legacy_quantiles=legacy_turnover_quantiles,
    )
    rank_autocorrelation = _rank_autocorrelation_table(
        analysis_data,
        lags,
        allow_legacy_zero=allow_legacy_zero_turnover,
    )

    cumulative = {
        period: portfolio.factor_cumulative_returns(
            analysis_data,
            period,
            long_short=long_short,
            group_neutral=group_neutral,
            equal_weight=equal_weight,
        )
        for period in selected_periods
    }
    quantile_cumulative = _quantile_cumulative_returns(aggregate_mean_by_date)
    positions = {
        period: portfolio.factor_positions(
            analysis_data,
            period,
            long_short=long_short,
            group_neutral=group_neutral,
            equal_weight=equal_weight,
        )
        for period in selected_periods
    }
    portfolio_inputs = (
        portfolio.build_factor_portfolio_inputs(
            analysis_data,
            selected_periods[0],
            capital=config.portfolio_capital,
            long_short=long_short,
            group_neutral=group_neutral,
            equal_weight=equal_weight,
            benchmark_period=config.portfolio_benchmark_period,
        )
        if include_portfolio_inputs
        else None
    )

    grouped: dict[Hashable, FactorGroupAnalysis] = {}
    if effective_by_group:
        for group in sorted(analysis_data["group"].dropna().unique().tolist(), key=repr):
            grouped[cast("Hashable", group)] = _group_analysis(
                analysis_data,
                cast("Hashable", group),
                long_short=long_short,
                equal_weight=equal_weight,
                lags=lags,
            )

    event_model, event_input_snapshot = _event_model(
        analysis_data,
        event_returns,
        event_before=normalized_event_before,
        event_after=normalized_event_after,
        long_short=long_short,
        group_neutral=group_neutral,
        by_group=effective_by_group,
        allow_legacy_event_windows=allow_legacy_event_windows,
    )
    quantile_statistics = _quantile_statistics(analysis_data)
    result_payload = {
        "config": config,
        "input_snapshot": snapshot,
        "event_input_snapshot": event_input_snapshot,
        "quantile_statistics": quantile_statistics,
        "factor_weights": weights.to_frame("factor"),
        "factor_returns": factor_returns,
        "factor_cumulative_returns": cumulative,
        "quantile_cumulative_returns": quantile_cumulative,
        "factor_positions": positions,
        "alpha_beta": alpha_beta,
        "mean_returns_by_quantile": mean_returns,
        "std_error_by_quantile": std_error,
        "mean_returns_by_date": mean_by_date,
        "std_error_by_date": std_error_by_date,
        "aggregate_mean_returns_by_quantile": aggregate_mean_returns,
        "aggregate_std_error_by_quantile": aggregate_std_error,
        "aggregate_mean_returns_by_date": aggregate_mean_by_date,
        "aggregate_std_error_by_date": aggregate_std_error_by_date,
        "aggregate_mean_return_spread": aggregate_mean_spread,
        "aggregate_mean_return_spread_std": aggregate_mean_spread_std,
        "mean_return_spread": mean_spread,
        "mean_return_spread_std": mean_spread_std,
        "information_coefficient": information,
        "mean_information_coefficient": mean_information,
        "aggregate_information_coefficient": aggregate_information,
        "aggregate_mean_information_coefficient": aggregate_mean_information,
        "summary_information_coefficient": summary_information,
        "quantile_turnover": turnover,
        "rank_autocorrelation": rank_autocorrelation,
        "grouped_results": grouped,
        "time_aggregated_results": time_aggregated,
        "aggregate_time_aggregated_results": aggregate_time_aggregated,
        "portfolio_inputs": portfolio_inputs,
        "event_returns": event_model,
    }

    return FactorAnalysisModel(
        config=config,
        factor_data=snapshot,
        forward_periods=selected_periods,
        quantile_statistics=quantile_statistics,
        factor_weights=weights.to_frame("factor"),
        factor_returns=factor_returns,
        factor_cumulative_returns=frozen_mapping(cumulative),
        quantile_cumulative_returns=frozen_mapping(quantile_cumulative),
        factor_positions=frozen_mapping(positions),
        alpha_beta=alpha_beta,
        mean_returns_by_quantile=mean_returns,
        std_error_by_quantile=std_error,
        mean_returns_by_date=mean_by_date,
        std_error_by_date=std_error_by_date,
        aggregate_mean_returns_by_quantile=aggregate_mean_returns,
        aggregate_std_error_by_quantile=aggregate_std_error,
        aggregate_mean_returns_by_date=aggregate_mean_by_date,
        aggregate_std_error_by_date=aggregate_std_error_by_date,
        aggregate_mean_return_spread=aggregate_mean_spread,
        aggregate_mean_return_spread_std=aggregate_mean_spread_std,
        mean_return_spread=mean_spread,
        mean_return_spread_std=mean_spread_std,
        information_coefficient=information,
        mean_information_coefficient=mean_information,
        aggregate_information_coefficient=aggregate_information,
        aggregate_mean_information_coefficient=aggregate_mean_information,
        summary_information_coefficient=summary_information,
        quantile_turnover=frozen_mapping(turnover),
        rank_autocorrelation=rank_autocorrelation,
        grouped_results=frozen_mapping(grouped),
        time_aggregated_results=frozen_mapping(time_aggregated),
        aggregate_time_aggregated_results=frozen_mapping(aggregate_time_aggregated),
        portfolio_inputs=portfolio_inputs,
        event_input_snapshot=event_input_snapshot,
        event_returns=event_model,
        result_fingerprint=fingerprint_value(result_payload),
    )


def analyze_factor(
    factor_data: pd.DataFrame,
    *,
    long_short: bool = True,
    group_neutral: bool = False,
    equal_weight: bool = False,
    by_group: bool = False,
    periods: Sequence[str] | None = None,
    turnover_periods: Sequence[int] = (1,),
    time_aggregation: Sequence[str] = ("M",),
    include_portfolio_inputs: bool = True,
    portfolio_capital: int | float | None = None,
    portfolio_benchmark_period: str = "1D",
    event_returns: pd.DataFrame | None = None,
    event_before: int | None = None,
    event_after: int | None = None,
) -> FactorAnalysisModel:
    """Compute one immutable, renderer-ready model from already-clean data.

    The enhanced public contract always requires positive, unique turnover
    lags.  Source-compatible tear-sheet oddities are intentionally kept in a
    private assembly bridge so they cannot weaken this API's validation.
    """

    return _analyze_factor(
        factor_data,
        long_short=long_short,
        group_neutral=group_neutral,
        equal_weight=equal_weight,
        by_group=by_group,
        periods=periods,
        turnover_periods=turnover_periods,
        time_aggregation=time_aggregation,
        include_portfolio_inputs=include_portfolio_inputs,
        portfolio_capital=portfolio_capital,
        portfolio_benchmark_period=portfolio_benchmark_period,
        event_returns=event_returns,
        event_before=event_before,
        event_after=event_after,
    )


def _analyze_factor_for_strict_turnover(
    factor_data: pd.DataFrame,
    *,
    allow_legacy_zero_turnover: bool = False,
    legacy_turnover_quantiles: tuple[int, ...] | None = None,
    allow_legacy_event_windows: bool = False,
    **kwargs: object,
) -> FactorAnalysisModel:
    """Private strict-tear-sheet bridge for pinned turnover table quirks."""

    return _analyze_factor(
        factor_data,
        **cast("Any", kwargs),
        allow_legacy_zero_turnover=allow_legacy_zero_turnover,
        legacy_turnover_quantiles=legacy_turnover_quantiles,
        allow_legacy_event_windows=allow_legacy_event_windows,
    )


__all__ = ["analyze_factor"]
