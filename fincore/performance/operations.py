"""Explicit canonical operation declarations for performance analytics."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fincore.runtime import OperationSpec

from .cashflows import cashflow_adjusted_returns, cashflow_adjusted_twr
from .disclosures import render_disclosure
from .inference import sharpe_confidence_interval, sharpe_standard_error, standard_error_of_mean
from .returns import mwr, twr, xirr

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = ["operations"]

_BINDINGS: tuple[tuple[str, Callable[..., Any]], ...] = (
    ("performance.cashflows.cashflow_adjusted_returns", cashflow_adjusted_returns),
    ("performance.cashflows.cashflow_adjusted_twr", cashflow_adjusted_twr),
    ("performance.disclosures.render_disclosure", render_disclosure),
    ("performance.inference.sharpe_confidence_interval", sharpe_confidence_interval),
    ("performance.inference.sharpe_standard_error", sharpe_standard_error),
    ("performance.inference.standard_error_of_mean", standard_error_of_mean),
    ("performance.returns.mwr", mwr),
    ("performance.returns.twr", twr),
    ("performance.returns.xirr", xirr),
)
_OPERATIONS = tuple(
    OperationSpec(
        operation_id=operation_id,
        capability_id=operation_id,
        domain="performance",
        callable=callable_,
        provenance={"owner": "performance", "kernel_module": callable_.__module__},
    )
    for operation_id, callable_ in _BINDINGS
)


def operations() -> tuple[OperationSpec, ...]:
    """Return immutable direct-call operation metadata for performance analytics."""
    return _OPERATIONS
