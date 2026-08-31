"""Pre-indexed immutable catalog of canonical domain operations."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

from .specs import OperationSpec


@dataclass(frozen=True, slots=True)
class OperationCatalog:
    """Resolve each operation ID to one direct domain callable in constant time."""

    operations: tuple[OperationSpec, ...]
    _by_operation_id: Mapping[str, OperationSpec] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        by_operation_id: dict[str, OperationSpec] = {}
        by_capability_id: dict[str, OperationSpec] = {}
        ordered_operations = tuple(sorted(self.operations, key=lambda item: item.operation_id))
        for spec in ordered_operations:
            if not isinstance(spec, OperationSpec):
                raise TypeError("operations must contain OperationSpec instances")
            if spec.operation_id in by_operation_id:
                raise ValueError(f"duplicate operation_id: {spec.operation_id}")
            previous = by_capability_id.get(spec.capability_id)
            if previous is not None and previous.implementation_fingerprint != spec.implementation_fingerprint:
                raise ValueError(
                    "leaf capability must have one canonical implementation: "
                    f"{spec.capability_id} maps to both "
                    f"{previous.implementation_fingerprint} and {spec.implementation_fingerprint}"
                )
            if previous is not None:
                if previous.semantic_mode is None or spec.semantic_mode is None:
                    raise ValueError(
                        "multiple operation IDs for one leaf capability require distinct approved semantic modes: "
                        f"{spec.capability_id}"
                    )
                if previous.semantic_mode == spec.semantic_mode:
                    raise ValueError(
                        "multiple operation IDs for one leaf capability cannot reuse a semantic mode: "
                        f"{spec.capability_id}.{spec.semantic_mode}"
                    )
            by_operation_id[spec.operation_id] = spec
            by_capability_id.setdefault(spec.capability_id, spec)
        object.__setattr__(self, "operations", ordered_operations)
        object.__setattr__(self, "_by_operation_id", MappingProxyType(dict(by_operation_id)))

    @property
    def operation_ids(self) -> tuple[str, ...]:
        """Return the catalog's deterministic operation-ID order."""
        return tuple(self._by_operation_id)

    @property
    def digest(self) -> str:
        """Return the stable identity of this immutable operation snapshot."""
        payload = [
            {
                "operation_id": spec.operation_id,
                "capability_id": spec.capability_id,
                "domain": spec.domain,
                "implementation_fingerprint": spec.implementation_fingerprint,
                "optional_extra": spec.optional_extra,
                "deterministic": spec.deterministic,
                "rng_policy": spec.rng_policy,
                "semantic_mode": spec.semantic_mode,
                "mode_approval": spec.mode_approval,
            }
            for spec in self.operations
        ]
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def resolve(self, operation_id: str) -> OperationSpec:
        """Return the exact registered spec for one operation ID."""
        try:
            return self._by_operation_id[operation_id]
        except KeyError as exc:
            raise KeyError(f"unknown operation_id: {operation_id}") from exc
