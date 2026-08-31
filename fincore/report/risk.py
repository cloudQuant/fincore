"""Project canonical risk-validation records into the shared report model."""

from __future__ import annotations

from typing import Any

import pandas as pd

from fincore.report.models import ReportDocument, ReportSection
from fincore.risk.report import RiskValidationReport

__all__ = ["build_risk_report"]


def _scalar_diagnostics(values: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in values.items() if value is None or isinstance(value, (str, bool, int, float))}


def build_risk_report(report: RiskValidationReport, *, title: str = "Risk Validation Report") -> ReportDocument:
    """Create a report model by consuming an already validated risk result."""

    if not isinstance(report, RiskValidationReport):
        raise TypeError("report must be a RiskValidationReport")
    payload = report.to_dict()
    backtest = payload["backtest"]
    metrics: dict[str, Any] = {
        "status": report.status,
        "forecast_event_count": len(report.forecast_events),
        "refit_count": len(report.refits),
        **_scalar_diagnostics(dict(report.diagnostics)),
    }
    if isinstance(backtest, dict):
        metrics.update({f"backtest.{key}": value for key, value in _scalar_diagnostics(backtest).items()})
    return ReportDocument(
        domain="risk",
        title=title,
        sections=(
            ReportSection(
                key="risk_validation",
                title="Risk validation",
                metrics=metrics,
                tables={
                    "forecast_events": pd.DataFrame(payload["forecast_events"]),
                    "refits": pd.DataFrame(payload["refits"]),
                },
                notes=(report.disclosure,),
            ),
        ),
        metadata={"inputs_digest": report.inputs_digest, "schema_version": report.schema_version},
    )
