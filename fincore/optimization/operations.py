"""Explicit canonical operation declarations for optimization."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fincore.runtime.specs import OperationSpec, make_operations_provider

from .frontier import efficient_frontier
from .objectives import optimize
from .risk_parity import risk_parity

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = ["operations"]

_BINDINGS: tuple[tuple[str, Callable[..., Any]], ...] = (
    ("optimization.frontier.efficient_frontier", efficient_frontier),
    ("optimization.objectives.optimize", optimize),
    ("optimization.risk_parity.risk_parity", risk_parity),
)

_OPERATIONS = tuple(
    OperationSpec(
        operation_id=operation_id,
        capability_id=operation_id,
        domain="optimization",
        callable=callable_,
        provenance={"owner": "optimization", "kernel_module": callable_.__module__},
    )
    for operation_id, callable_ in _BINDINGS
)


operations = make_operations_provider(_OPERATIONS)
