"""Execute direct domain callables registered in an :class:`OperationCatalog`."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from time import perf_counter_ns
from types import MappingProxyType
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


@dataclass(frozen=True, slots=True, init=False)
class OperationRequest:
    """One isolated operation request that can safely enter a batch plan."""

    operation_id: str
    snapshot: AnalysisSnapshot
    _config_snapshot: AnalysisSnapshot = field(repr=False)

    def __init__(
        self,
        operation_id: str,
        inputs: Mapping[str, Any],
        config: Mapping[str, Any] | None = None,
    ) -> None:
        if not isinstance(operation_id, str) or not operation_id.strip():
            raise ValueError("operation_id must be a non-empty string")
        input_snapshot = AnalysisSnapshot.from_inputs(_mapping(inputs, "inputs"))
        config_snapshot = AnalysisSnapshot.from_inputs(
            {"config": _mapping(config, "config") if config is not None else {}}
        )
        object.__setattr__(self, "operation_id", operation_id)
        object.__setattr__(self, "snapshot", input_snapshot)
        object.__setattr__(self, "_config_snapshot", config_snapshot)

    @property
    def config(self) -> Mapping[str, Any]:
        """Return an independent copy of the request's isolated config."""
        config = self._config_snapshot.materialize()["config"]
        if not isinstance(config, Mapping):  # pragma: no cover - established by __init__.
            raise TypeError("operation request config must be a mapping")
        return config

    @property
    def config_digest(self) -> str:
        """Return the deterministic digest of the isolated configuration."""
        return self._config_snapshot.digest


@dataclass(frozen=True, slots=True)
class ExecutionPlan:
    """A catalog-bound, ordered batch of already-isolated requests."""

    catalog_digest: str
    requests: tuple[OperationRequest, ...]


def plan(requests: Iterable[OperationRequest], *, catalog: OperationCatalog) -> ExecutionPlan:
    """Validate requests against one immutable catalog snapshot without running them."""
    planned_requests = tuple(requests)
    for request in planned_requests:
        if not isinstance(request, OperationRequest):
            raise TypeError("requests must contain OperationRequest instances")
        catalog.resolve(request.operation_id)
    return ExecutionPlan(catalog_digest=catalog.digest, requests=planned_requests)


def batch(
    requests: Iterable[OperationRequest] | ExecutionPlan,
    *,
    catalog: OperationCatalog,
) -> tuple[Result, ...]:
    """Execute an ordered batch against the exact catalog it was planned for."""
    execution_plan = requests if isinstance(requests, ExecutionPlan) else plan(requests, catalog=catalog)
    if execution_plan.catalog_digest != catalog.digest:
        raise ValueError("execution plan was created for another catalog snapshot")
    return tuple(
        run(
            request.operation_id,
            request.snapshot.materialize(),
            request.config,
            catalog=catalog,
        )
        for request in execution_plan.requests
    )


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
