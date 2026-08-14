"""Compute-once assembly of enhanced factor-analysis models.

This module consumes already-clean factor data.  It intentionally does not
re-enter the raw factor/price preparation path and does not import a plotting
library; later renderers receive only :class:`FactorAnalysisModel` data.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence  # noqa: TC003
from typing import cast

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
    snapshot = factor_data.copy(deep=True)
    forward_columns = get_forward_returns_columns(snapshot.columns)
    if not len(forward_columns):
        raise ValueError("factor_data must contain at least one forward-return column")
    if not all(isinstance(period, str) for period in forward_columns):
        raise ValueError("forward-return labels must use canonical timedelta strings")
    return snapshot, tuple(cast("str", period) for period in forward_columns)


def _normalize_periods(available: tuple[str, ...], periods: Sequence[str] | None) -> tuple[str, ...]:
    """Validate a deterministic nonempty forward-period selection."""

    selected = available if periods is None else tuple(periods)
    if not selected:
        raise ValueError("at least one forward period must be selected")
    if len(set(selected)) != len(selected):
        raise ValueError("forward period selection must not contain duplicates")
    unknown = tuple(period for period in selected if period not in available)
    if unknown:
        raise ValueError(f"unknown forward periods: {list(unknown)!r}")
    return selected


def _normalize_positive_lags(turnover_periods: Sequence[int]) -> tuple[int, ...]:
    """Validate turnover/rank lags once for both renderer-facing fields."""

    normalized = tuple(turnover_periods)
    if not normalized:
        raise ValueError("turnover_periods must contain at least one positive lag")
    if any(not isinstance(period, int) or period <= 0 for period in normalized):
        raise ValueError("turnover_periods must contain positive integers")
    if len(set(normalized)) != len(normalized):
        raise ValueError("turnover_periods must not contain duplicates")
    return normalized


def _normalize_time_aggregation(time_aggregation: Sequence[str]) -> tuple[str, ...]:
    """Validate named pandas aggregation frequencies for the model payload."""

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


def _quantile_turnover_tables(factor_data: pd.DataFrame, lags: tuple[int, ...]) -> dict[int, pd.DataFrame]:
    """Assemble typed quantile turnover tables from per-quantile kernel Series."""

    quantiles = _quantile_values(factor_data)
    results: dict[int, pd.DataFrame] = {}
    for lag in lags:
        columns = {
            quantile: performance.quantile_turnover(factor_data["factor_quantile"], cast("int", quantile), period=lag)
            for quantile in quantiles
        }
        results[lag] = pd.DataFrame(columns)
    return results


def _rank_autocorrelation_table(factor_data: pd.DataFrame, lags: tuple[int, ...]) -> pd.DataFrame:
    """Collect all requested rank autocorrelation lags in one table."""

    return pd.DataFrame({lag: performance.factor_rank_autocorrelation(factor_data, period=lag) for lag in lags})


def _event_model(
    factor_data: pd.DataFrame,
    event_returns: pd.DataFrame | None,
    *,
    event_before: int | None,
    event_after: int | None,
    long_short: bool,
    group_neutral: bool,
    by_group: bool,
) -> tuple[EventAnalysisModel | None, pd.DataFrame | None]:
    """Build event data only when a complete optional event input is present."""

    if event_returns is None:
        return None, None
    if not isinstance(event_returns, pd.DataFrame):
        raise TypeError("event_returns must be a pandas DataFrame")
    # Even an incomplete event request belongs in the model provenance: a
    # later caller must not receive the same result fingerprint after changing
    # supplied event data merely because no event section was requested yet.
    returns_snapshot = event_returns.copy(deep=True)
    if event_before is None or event_after is None:
        return None, returns_snapshot
    if not isinstance(event_before, int) or not isinstance(event_after, int) or event_before < 0 or event_after < 0:
        raise ValueError("event_before and event_after must be non-negative integers")
    demean_by = factor_data["factor_quantile"] if long_short else None
    windows = performance.common_start_returns(
        factor_data["factor_quantile"],
        returns_snapshot,
        event_before,
        event_after,
        cumulative=True,
        mean_by_date=True,
        demean_by=demean_by,
    )
    average = performance.average_cumulative_return_by_quantile(
        factor_data,
        returns_snapshot,
        periods_before=event_before,
        periods_after=event_after,
        demeaned=long_short,
        group_adjust=group_neutral,
        by_group=by_group,
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
    include_pyfolio: bool = True,
    pyfolio_capital: int | float | None = None,
    pyfolio_benchmark_period: str = "1D",
    event_returns: pd.DataFrame | None = None,
    event_before: int | None = None,
    event_after: int | None = None,
) -> FactorAnalysisModel:
    """Compute one immutable, renderer-ready model from already-clean data.

    ``factor_data`` must have crossed :func:`prepare_factor_data` (or the
    equivalent forward-return preparation boundary) already.  This function
    owns a deep input snapshot, invokes analytical kernels during assembly,
    and leaves later consumers to read fields rather than recalculate them.
    """

    snapshot, available_periods = _copy_clean_factor_data(factor_data)
    selected_periods = _normalize_periods(available_periods, periods)
    lags = _normalize_positive_lags(turnover_periods)
    aggregations = _normalize_time_aggregation(time_aggregation)
    if not isinstance(pyfolio_benchmark_period, str):
        raise TypeError("pyfolio_benchmark_period must be a string")
    if pyfolio_capital is not None and not isinstance(pyfolio_capital, (int, float)):
        raise TypeError("pyfolio_capital must be a number or None")

    config = FactorAnalysisConfig(
        long_short=long_short,
        group_neutral=group_neutral,
        equal_weight=equal_weight,
        by_group=by_group,
        periods=selected_periods,
        event_before=event_before,
        event_after=event_after,
        turnover_periods=lags,
        time_aggregation=aggregations,
        include_pyfolio=include_pyfolio,
        pyfolio_capital=pyfolio_capital,
        pyfolio_benchmark_period=pyfolio_benchmark_period,
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
    mean_by_date, _ = performance.mean_return_by_quantile(
        analysis_data,
        by_date=True,
        by_group=effective_by_group,
        demeaned=long_short,
        group_adjust=group_neutral,
    )
    quantiles = _quantile_values(analysis_data)
    if len(quantiles) >= 2:
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

    # Compute the IC exactly once.  All aggregate tables below consume this
    # snapshot instead of calling mean_information_coefficient again.
    information = performance.factor_information_coefficient(
        analysis_data,
        group_adjust=group_neutral,
        by_group=effective_by_group,
    )
    mean_information = _mean_information(information, by_group=effective_by_group)
    time_aggregated = _time_aggregated_information(
        information,
        aggregations,
        by_group=effective_by_group,
    )
    turnover = _quantile_turnover_tables(analysis_data, lags)
    rank_autocorrelation = _rank_autocorrelation_table(analysis_data, lags)

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
    pyfolio_inputs = (
        portfolio.create_pyfolio_input(
            analysis_data,
            selected_periods[0],
            capital=config.pyfolio_capital,
            long_short=long_short,
            group_neutral=group_neutral,
            equal_weight=equal_weight,
            benchmark_period=config.pyfolio_benchmark_period,
        )
        if include_pyfolio
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
        event_before=event_before,
        event_after=event_after,
        long_short=long_short,
        group_neutral=group_neutral,
        by_group=effective_by_group,
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
        "factor_positions": positions,
        "alpha_beta": alpha_beta,
        "mean_returns_by_quantile": mean_returns,
        "std_error_by_quantile": std_error,
        "mean_returns_by_date": mean_by_date,
        "mean_return_spread": mean_spread,
        "mean_return_spread_std": mean_spread_std,
        "information_coefficient": information,
        "mean_information_coefficient": mean_information,
        "quantile_turnover": turnover,
        "rank_autocorrelation": rank_autocorrelation,
        "grouped_results": grouped,
        "time_aggregated_results": time_aggregated,
        "pyfolio_inputs": pyfolio_inputs,
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
        factor_positions=frozen_mapping(positions),
        alpha_beta=alpha_beta,
        mean_returns_by_quantile=mean_returns,
        std_error_by_quantile=std_error,
        mean_returns_by_date=mean_by_date,
        mean_return_spread=mean_spread,
        mean_return_spread_std=mean_spread_std,
        information_coefficient=information,
        mean_information_coefficient=mean_information,
        quantile_turnover=frozen_mapping(turnover),
        rank_autocorrelation=rank_autocorrelation,
        grouped_results=frozen_mapping(grouped),
        time_aggregated_results=frozen_mapping(time_aggregated),
        pyfolio_inputs=pyfolio_inputs,
        event_returns=event_model,
        result_fingerprint=fingerprint_value(result_payload),
    )


__all__ = ["analyze_factor"]
