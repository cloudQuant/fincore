"""Project a precomputed factor-analysis model into the common report model."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd

from fincore.factor_analysis.models import FactorAnalysisModel
from fincore.factor_analysis.render_matplotlib import (
    build_information_table,
    build_quantile_statistics_table,
    build_returns_table,
    build_turnover_tables,
)
from fincore.report.models import ReportDocument, ReportSection

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = ["build_factor_report"]


def _scalar_metrics(value: pd.Series | pd.DataFrame, *, prefix: str) -> dict[str, float]:
    if isinstance(value, pd.Series):
        return {f"{prefix}.{key}": float(item) for key, item in value.items()}
    if value.empty:
        return {}
    first = value.iloc[0]
    return {f"{prefix}.{key}": float(item) for key, item in first.items()}


def _turnover_by_quantile(quantile_turnover: Mapping[int, pd.DataFrame]) -> pd.DataFrame:
    """Mean turnover per quantile, using already computed turnover tables."""

    table = pd.DataFrame(
        {f"{lag}D": values.mean() for lag, values in sorted(quantile_turnover.items())}
    )
    table.index.name = "factor_quantile"
    return table


def _coverage_metrics(
    factor_data: pd.DataFrame, forward_periods: tuple[str, ...]
) -> dict[str, Any]:
    observed = len(factor_data)
    finite = int(factor_data["factor"].notna().sum())
    metrics: dict[str, Any] = {
        "observation_count": observed,
        "finite_factor_count": finite,
        "coverage_ratio": finite / observed if observed else float("nan"),
        "quantile_count": int(factor_data["factor_quantile"].nunique()),
    }
    for period in forward_periods:
        if period in factor_data.columns:
            metrics[f"forward_return_count.{period}"] = int(factor_data[period].notna().sum())
    return metrics


def _dataset_metadata(factor_data: pd.DataFrame) -> dict[str, Any]:
    dates = pd.DatetimeIndex(factor_data.index.get_level_values("date"))
    assets = factor_data.index.get_level_values("asset").unique()
    if not len(dates):
        return {"asset_count": 0, "date_count": 0, "start": "", "end": ""}
    return {
        "asset_count": len(assets),
        "date_count": int(dates.nunique()),
        "start": dates.min().isoformat(),
        "end": dates.max().isoformat(),
    }


def build_factor_report(model: FactorAnalysisModel, *, title: str = "Factor Analysis Report") -> ReportDocument:
    """Build a report by projecting an existing factor model, never recomputing it."""

    if not isinstance(model, FactorAnalysisModel):
        raise TypeError("model must be a FactorAnalysisModel")

    forward_periods = model.forward_periods
    periods = tuple(str(period) for period in forward_periods)
    factor_data = model.factor_data
    factor_cumulative_returns = model.factor_cumulative_returns
    information_coefficient = model.information_coefficient
    alpha_beta = model.alpha_beta
    mean_returns_by_quantile = model.mean_returns_by_quantile
    mean_return_spread = model.mean_return_spread
    rank_autocorrelation = model.rank_autocorrelation
    quantile_turnover = model.quantile_turnover
    mean_information_coefficient = model.mean_information_coefficient
    result_fingerprint = model.result_fingerprint
    cumulative_series = {
        f"factor_cumulative_returns.{period}": values
        for period, values in factor_cumulative_returns.items()
    }
    cumulative_units = dict.fromkeys(cumulative_series, "growth_multiple")
    cumulative_legends = {
        key: f"Factor {period}" for key, period in ((key, key.rsplit(".", maxsplit=1)[-1]) for key in cumulative_series)
    }
    ic_series = {f"ic.{period}": information_coefficient[period] for period in periods}
    ic_units = dict.fromkeys(ic_series, "correlation")
    ic_legends = {key: f"{period} IC" for key, period in zip(ic_series, periods, strict=True)}

    information_summary = build_information_table(information_coefficient)
    quantile_statistics = build_quantile_statistics_table(factor_data)
    returns_summary = build_returns_table(alpha_beta, mean_returns_by_quantile, mean_return_spread)
    autocorrelation: dict[object, pd.Series] = {
        int(lag): rank_autocorrelation[lag] for lag in rank_autocorrelation.columns
    }
    turnover_input: dict[object, pd.DataFrame] = {
        int(lag): values for lag, values in quantile_turnover.items()
    }
    turnover_summary, autocorrelation_summary = build_turnover_tables(autocorrelation, turnover_input)
    turnover_by_quantile = _turnover_by_quantile(quantile_turnover)

    summary_metrics: dict[str, Any] = {
        "forward_period_count": len(periods),
        "result_fingerprint": result_fingerprint,
        **_scalar_metrics(mean_information_coefficient, prefix="mean_information_coefficient"),
        **_coverage_metrics(factor_data, forward_periods),
    }
    summary_tables = {
        "quantile_statistics": quantile_statistics,
        "alpha_beta": alpha_beta,
        "mean_returns_by_quantile": mean_returns_by_quantile,
        "information_coefficient": information_coefficient,
    }
    return ReportDocument(
        domain="factor",
        title=title,
        sections=(
            ReportSection(
                key="factor_summary",
                title="Factor summary",
                metrics=summary_metrics,
                tables=summary_tables,
                series=cumulative_series,
                units=cumulative_units,
                legends=cumulative_legends,
            ),
            ReportSection(
                key="factor_information",
                title="Information coefficient",
                tables={"information_summary": information_summary},
                series=ic_series,
                units=ic_units,
                legends=ic_legends,
            ),
            ReportSection(
                key="factor_returns",
                title="Factor returns",
                tables={
                    "returns_summary": returns_summary,
                    "alpha_beta": alpha_beta,
                    "mean_returns_by_quantile": mean_returns_by_quantile,
                },
            ),
            ReportSection(
                key="factor_turnover",
                title="Turnover and autocorrelation",
                tables={
                    "mean_turnover_by_quantile": turnover_by_quantile,
                    "rank_autocorrelation_summary": autocorrelation_summary,
                    "turnover_summary": turnover_summary,
                },
            ),
        ),
        metadata={
            "forward_periods": periods,
            "factor_result_fingerprint": result_fingerprint,
            **_dataset_metadata(factor_data),
        },
    )
