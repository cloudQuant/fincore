"""Execute direct domain callables registered in an :class:`OperationCatalog`."""

from __future__ import annotations

from collections.abc import Mapping
from time import perf_counter_ns
from typing import TYPE_CHECKING, Any

from .data import AnalysisSnapshot
from .results import Result

if TYPE_CHECKING:
    from .catalog import OperationCatalog


def _mapping(value: Mapping[str, Any], label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    if any(not isinstance(key, str) for key in value):
        raise TypeError(f"{label} keys must be strings")
    return dict(value)


def run(
    operation_id: str,
    inputs: Mapping[str, Any],
    config: Mapping[str, Any] | None = None,
    *,
    catalog: OperationCatalog,
) -> Result:
    """Run one operation by directly invoking its registered domain callable.

    ``config`` is orchestration-only provenance for now. It is deliberately not
    forwarded to the domain function, which prevents runtime configuration from
    becoming a second domain API or altering the canonical call path.
    """
    snapshot = AnalysisSnapshot.from_inputs(_mapping(inputs, "inputs"))
    run_config = _mapping(config, "config") if config is not None else {}
    config_digest = AnalysisSnapshot.from_inputs({"config": run_config}).digest
    return _run_snapshot(
        catalog.resolve(operation_id),
        snapshot,
        run_config,
        catalog_digest=catalog.digest,
        cache="disabled",
        config_digest=config_digest,
    )


def _run_snapshot(
    spec: Any,
    snapshot: AnalysisSnapshot,
    config: Mapping[str, Any],
    *,
    catalog_digest: str,
    cache: str,
    config_digest: str,
) -> Result:
    """Run a resolved operation against an already-isolated input snapshot."""
    started_ns = perf_counter_ns()
    value = spec.callable(**snapshot.materialize())
    elapsed_ns = perf_counter_ns() - started_ns
    return Result(
        value=value,
        metadata={
            "operation_id": spec.operation_id,
            "capability_id": spec.capability_id,
            "domain": spec.domain,
            "implementation_fingerprint": spec.implementation_fingerprint,
            "input_digest": snapshot.digest,
            "config_digest": config_digest,
            "catalog_digest": catalog_digest,
            "duration_ns": elapsed_ns,
            "config": dict(config),
            "cache": cache,
        },
    )
