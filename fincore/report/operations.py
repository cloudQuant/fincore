"""Explicit report compute operations for the canonical runtime catalog."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fincore.runtime.specs import OperationSpec, make_operations_provider

from .factor.compute import build_factor_report
from .portfolio.compute import build_portfolio_report
from .risk import build_risk_report

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = ["operations"]

_BINDINGS: tuple[tuple[str, Callable[..., Any]], ...] = (
    ("report.factor.build_factor_report", build_factor_report),
    ("report.portfolio.build_portfolio_report", build_portfolio_report),
    ("report.risk.build_risk_report", build_risk_report),
)
_OPERATIONS = tuple(
    OperationSpec(
        operation_id=operation_id,
        capability_id=operation_id,
        domain="report",
        callable=callable_,
        provenance={"owner": "report", "kernel_module": callable_.__module__},
    )
    for operation_id, callable_ in _BINDINGS
)


operations = make_operations_provider(_OPERATIONS)
