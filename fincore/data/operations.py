"""Explicit canonical operation declarations for data acquisition and snapshots."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fincore.runtime.specs import OperationSpec, make_operations_provider

from .providers import fetch_multiple_prices, fetch_price_data, get_provider
from .snapshots import DataSnapshot

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = ["operations"]

_BINDINGS: tuple[tuple[str, Callable[..., Any]], ...] = (
    ("data.providers.fetch_multiple_prices", fetch_multiple_prices),
    ("data.providers.fetch_price_data", fetch_price_data),
    ("data.providers.get_provider", get_provider),
    ("data.snapshots.DataSnapshot.from_frame", DataSnapshot.from_frame),
)

_OPERATIONS = tuple(
    OperationSpec(
        operation_id=operation_id,
        capability_id=operation_id,
        domain="data",
        callable=callable_,
        provenance={"owner": "data", "kernel_module": callable_.__module__},
    )
    for operation_id, callable_ in _BINDINGS
)


operations = make_operations_provider(_OPERATIONS)
