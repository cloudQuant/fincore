"""Project a precomputed factor-analysis model into the common report model."""

from __future__ import annotations

from typing import Any

import pandas as pd

from fincore.factor_analysis.models import FactorAnalysisModel
from fincore.factor_analysis.render_matplotlib import (
    build_information_table,
    build_quantile_statistics_table,
    build_returns_table,
    build_turnover_tables,
)
from fincore.report.models import ReportDocument, ReportSection

__all__ = ["build_factor_report"]


def _scalar_metrics(value: pd.Series | pd.DataFrame, *, prefix: str) -> dict[str, float]:
    if isinstance(value, pd.Series):
        return {f"{prefix}.{key}": float(item) for key, item in value.items()}
    if value.empty:
        return {}
    first = value.iloc[0]
    return {f"{prefix}.{key}": float(item) for key, item in first.items()}


def _turnover_by_quantile(model: FactorAnalysisModel) -> pd.DataFrame:
    """Mean turnover per quantile, using already computed turnover tables."""

    table = pd.DataFrame({f"{lag}D": values.mean() for lag, values in sorted(model.quantile_turnover.items())})
    table.index.name = "factor_quantile"
    return table


def _coverage_metrics(model: FactorAnalysisModel) -> dict[str, Any]:
    observed = len(model.factor_data)
    finite = int(model.factor_data["factor"].notna().sum())
    metrics: dict[str, Any] = {
        "observation_count": observed,
        "finite_factor_count": finite,
        "coverage_ratio": finite / observed if observed else float("nan"),
        "quantile_count": int(model.factor_data["factor_quantile"].nunique()),
    }
    for period in model.forward_periods:
        if period in model.factor_data.columns:
            metrics[f"forward_return_count.{period}"] = int(model.factor_data[period].notna().sum())
    return metrics


def _dataset_metadata(model: FactorAnalysisModel) -> dict[str, Any]:
    dates = pd.DatetimeIndex(model.factor_data.index.get_level_values("date"))
    assets = model.factor_data.index.get_level_values("asset").unique()
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

    periods = tuple(str(period) for period in model.forward_periods)
    cumulative_series = {
        f"factor_cumulative_returns.{period}": values for period, values in model.factor_cumulative_returns.items()
    }
    cumulative_units = dict.fromkeys(cumulative_series, "growth_multiple")
    cumulative_legends = {
        key: f"Factor {period}" for key, period in ((key, key.rsplit(".", maxsplit=1)[-1]) for key in cumulative_series)
    }
    ic_series = {f"ic.{period}": model.information_coefficient[period] for period in periods}
    ic_units = dict.fromkeys(ic_series, "correlation")
    ic_legends = {key: f"{period} IC" for key, period in zip(ic_series, periods, strict=True)}

    information_summary = build_information_table(model.information_coefficient)
    quantile_statistics = build_quantile_statistics_table(model.factor_data)
    returns_summary = build_returns_table(model.alpha_beta, model.mean_returns_by_quantile, model.mean_return_spread)
    autocorrelation = {int(lag): model.rank_autocorrelation[lag] for lag in model.rank_autocorrelation.columns}
    turnover_summary, autocorrelation_summary = build_turnover_tables(autocorrelation, model.quantile_turnover)
    turnover_by_quantile = _turnover_by_quantile(model)

    summary_metrics: dict[str, Any] = {
        "forward_period_count": len(periods),
        "result_fingerprint": model.result_fingerprint,
        **_scalar_metrics(model.mean_information_coefficient, prefix="mean_information_coefficient"),
        **_coverage_metrics(model),
    }
    summary_tables = {
        "quantile_statistics": quantile_statistics,
        "alpha_beta": model.alpha_beta,
        "mean_returns_by_quantile": model.mean_returns_by_quantile,
        "information_coefficient": model.information_coefficient,
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
                    "alpha_beta": model.alpha_beta,
                    "mean_returns_by_quantile": model.mean_returns_by_quantile,
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
            "factor_result_fingerprint": model.result_fingerprint,
            **_dataset_metadata(model),
        },
    )
