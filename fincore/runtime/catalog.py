"""Pre-indexed immutable catalog of canonical domain operations."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Mapping

from fincore.exceptions import OperationResolutionError

from .specs import OperationSpec

if TYPE_CHECKING:
    from fincore.extensions.snapshot import ExtensionSnapshot


@dataclass(frozen=True, slots=True)
class OperationCatalog:
    """Resolve each operation ID to one direct domain callable in constant time."""

    operations: tuple[OperationSpec, ...]
    extension_digest: str | None = None
    _by_operation_id: Mapping[str, OperationSpec] = field(init=False, repr=False, compare=False)
    _extension_snapshot: object | None = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.extension_digest is not None and (
            not isinstance(self.extension_digest, str) or not self.extension_digest
        ):
            raise ValueError("extension_digest must be a non-empty string or None")
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
        object.__setattr__(self, "_extension_snapshot", None)

    @property
    def operation_ids(self) -> tuple[str, ...]:
        """Return the catalog's deterministic operation-ID order."""
        return tuple(self._by_operation_id)

    @property
    def extension_snapshot(self) -> object | None:
        """Return the exact immutable extension snapshot pinned by this catalog."""

        return self._extension_snapshot

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
        payload.append({"extension_digest": self.extension_digest})
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def resolve(self, operation_id: str) -> OperationSpec:
        """Return the exact registered spec for one operation ID."""
        try:
            return self._by_operation_id[operation_id]
        except KeyError as exc:
            raise OperationResolutionError(operation_id) from exc

    def with_extensions(self, extension_snapshot: ExtensionSnapshot) -> OperationCatalog:
        """Return a new catalog pinned to one immutable extension snapshot.

        Runtime deliberately uses only the snapshot protocol here instead of
        importing the extensions domain. This keeps the runtime independent of
        optional extension implementations while rejecting malformed objects.
        """

        if self.extension_digest is not None:
            raise ValueError("catalog is already bound to an extension snapshot")
        operations: Any = getattr(extension_snapshot, "operations", None)
        digest: Any = getattr(extension_snapshot, "digest", None)
        if not isinstance(operations, tuple) or not all(
            isinstance(operation, OperationSpec) for operation in operations
        ):
            raise TypeError("extension_snapshot must expose a tuple of OperationSpec operations")
        if not isinstance(digest, str) or not digest:
            raise TypeError("extension_snapshot must expose a non-empty digest")
        catalog = OperationCatalog((*self.operations, *operations), extension_digest=digest)
        object.__setattr__(catalog, "_extension_snapshot", extension_snapshot)
        return catalog
