"""Project a precomputed factor-analysis model into the common report model."""

from __future__ import annotations

from typing import Any

import pandas as pd

from fincore.factor_analysis.models import FactorAnalysisModel
from fincore.report.models import ReportDocument, ReportSection

__all__ = ["build_factor_report"]


def _scalar_metrics(value: pd.Series | pd.DataFrame, *, prefix: str) -> dict[str, float]:
    if isinstance(value, pd.Series):
        return {f"{prefix}.{key}": float(item) for key, item in value.items()}
    if value.empty:
        return {}
    first = value.iloc[0]
    return {f"{prefix}.{key}": float(item) for key, item in first.items()}


def build_factor_report(model: FactorAnalysisModel, *, title: str = "Factor Analysis Report") -> ReportDocument:
    """Build a report by projecting an existing factor model, never recomputing it."""

    if not isinstance(model, FactorAnalysisModel):
        raise TypeError("model must be a FactorAnalysisModel")
    cumulative_series = {
        f"factor_cumulative_returns.{period}": values for period, values in model.factor_cumulative_returns.items()
    }
    units = dict.fromkeys(cumulative_series, "growth_multiple")
    legends = {
        key: f"Factor {period}" for key, period in ((key, key.rsplit(".", maxsplit=1)[-1]) for key in cumulative_series)
    }
    summary_metrics: dict[str, Any] = {
        "forward_period_count": len(model.forward_periods),
        "result_fingerprint": model.result_fingerprint,
        **_scalar_metrics(model.mean_information_coefficient, prefix="mean_information_coefficient"),
    }
    return ReportDocument(
        domain="factor",
        title=title,
        sections=(
            ReportSection(
                key="factor_summary",
                title="Factor summary",
                metrics=summary_metrics,
                tables={
                    "quantile_statistics": model.quantile_statistics,
                    "alpha_beta": model.alpha_beta,
                    "mean_returns_by_quantile": model.mean_returns_by_quantile,
                    "information_coefficient": model.information_coefficient,
                },
                series=cumulative_series,
                units=units,
                legends=legends,
            ),
        ),
        metadata={"forward_periods": model.forward_periods, "factor_result_fingerprint": model.result_fingerprint},
    )
