"""Declarative, renderer-independent workflow plans for factor analysis."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["FACTOR_WORKFLOWS", "FactorWorkflow"]


@dataclass(frozen=True, slots=True)
class FactorWorkflow:
    """One named factor workflow sharing the canonical analysis model builder."""

    workflow_id: str
    model_operation_id: str
    sections: tuple[str, ...]
    renderer: None = None


_ANALYZE_OPERATION = "factor_analysis.analysis.analyze_factor"

FACTOR_WORKFLOWS: tuple[FactorWorkflow, ...] = (
    FactorWorkflow("summary", _ANALYZE_OPERATION, ("returns", "information", "turnover")),
    FactorWorkflow("returns", _ANALYZE_OPERATION, ("returns",)),
    FactorWorkflow("information", _ANALYZE_OPERATION, ("information",)),
    FactorWorkflow("turnover", _ANALYZE_OPERATION, ("turnover",)),
    FactorWorkflow("full", _ANALYZE_OPERATION, ("returns", "information", "turnover")),
    FactorWorkflow("event_returns", _ANALYZE_OPERATION, ("event_returns",)),
    FactorWorkflow("event_study", _ANALYZE_OPERATION, ("event_distribution", "event_returns", "returns")),
)

if len({workflow.workflow_id for workflow in FACTOR_WORKFLOWS}) != len(FACTOR_WORKFLOWS):  # pragma: no cover
    raise RuntimeError("factor workflow identifiers must be unique")
