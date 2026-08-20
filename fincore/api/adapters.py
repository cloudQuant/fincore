"""Profile adapters that isolate strict from enhanced execution paths.

A strict profile (``strict_*``) calls its frozen kernel directly and never
enters the enhanced validation pipeline or constructs an enhanced stateful
class.  An enhanced profile routes through :func:`fincore.api.invoke.invoke`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fincore.api.catalog import OperationCatalog

from fincore.api.invoke import invoke, resolve_kernel
from fincore.contracts.profiles import STRICT_PROFILES
from fincore.results import AnalysisResult

__all__ = ["is_strict_profile", "route"]


def is_strict_profile(profile: str) -> bool:
    return profile in STRICT_PROFILES


def route(
    catalog: OperationCatalog,
    operation_id: str,
    profile: str,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Execute an operation under its profile, isolating strict from enhanced.

    Strict profiles return the raw kernel value directly (frozen behavior);
    enhanced profiles return an :class:`AnalysisResult`.
    """
    if is_strict_profile(profile):
        definition = catalog.resolve_definition(operation_id, profile)
        return resolve_kernel(definition.kernel_ref)(*args, **kwargs)
    return invoke(catalog, operation_id, profile, *args, **kwargs)
