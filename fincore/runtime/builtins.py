"""The one explicit, lazy composition root for builtin domain operations."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

from .catalog import OperationCatalog

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from .specs import OperationSpec

_BUILTIN_PROVIDER_PATHS: tuple[str, ...] = (
    "fincore.metrics.operations:operations",
    "fincore.performance.operations:operations",
    "fincore.portfolio.operations:operations",
    "fincore.factor_analysis.operations:operations",
    "fincore.simulation.operations:operations",
    "fincore.optimization.operations:operations",
    "fincore.data.operations:operations",
    "fincore.extensions.operations:operations",
    "fincore.risk.operations:operations",
    "fincore.attribution.operations:operations",
    "fincore.report.operations:operations",
)


def compose_catalog(providers: Iterable[Callable[[], Iterable[OperationSpec]]]) -> OperationCatalog:
    """Build a catalog from only the explicit provider sequence supplied."""
    operations: list[OperationSpec] = []
    for provider in providers:
        if not callable(provider):
            raise TypeError("operation provider must be callable")
        operations.extend(provider())
    return OperationCatalog(tuple(operations))


def _load_provider(provider_path: str) -> Callable[[], Iterable[OperationSpec]]:
    module_name, separator, attribute = provider_path.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError(f"invalid builtin operation provider path: {provider_path!r}")
    provider = getattr(import_module(module_name), attribute)
    if not callable(provider):
        raise TypeError(f"builtin operation provider is not callable: {provider_path!r}")
    return provider


def builtin_catalog() -> OperationCatalog:
    """Construct the builtin catalog from the audited fixed provider path list."""
    return compose_catalog(_load_provider(provider_path) for provider_path in _BUILTIN_PROVIDER_PATHS)
