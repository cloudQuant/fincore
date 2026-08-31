"""Immutable operation declarations for the canonical runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Callable, Mapping


def _freeze_metadata(value: Any) -> Any:
    """Recursively detach schema/provenance metadata from its declaration caller."""
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze_metadata(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_metadata(item) for item in value)
    if isinstance(value, set):
        return frozenset(_freeze_metadata(item) for item in value)
    return value


def _required_identifier(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


@dataclass(frozen=True, slots=True)
class OperationSpec:
    """One domain-owned operation and its direct canonical implementation.

    The callable is the domain function itself. There are intentionally no
    public-path aliases, legacy profiles, string signatures, adapters, or
    result projections in this declaration.
    """

    operation_id: str
    capability_id: str
    domain: str
    callable: Callable[..., Any]
    input_schema: Mapping[str, Any] = field(default_factory=dict)
    output_schema: Any = None
    optional_extra: str | None = None
    deterministic: bool = True
    rng_policy: str = "none"
    semantic_mode: str | None = None
    mode_approval: str | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in ("operation_id", "capability_id", "domain", "rng_policy"):
            _required_identifier(getattr(self, field_name), field_name)
        if not callable(self.callable):
            raise TypeError("callable must be callable")
        if self.optional_extra is not None:
            _required_identifier(self.optional_extra, "optional_extra")
        if (self.semantic_mode is None) != (self.mode_approval is None):
            raise ValueError("semantic_mode and mode_approval must be provided together")
        if self.semantic_mode is not None:
            _required_identifier(self.semantic_mode, "semantic_mode")
            _required_identifier(self.mode_approval, "mode_approval")
        object.__setattr__(self, "input_schema", _freeze_metadata(self.input_schema))
        object.__setattr__(self, "output_schema", _freeze_metadata(self.output_schema))
        object.__setattr__(self, "provenance", _freeze_metadata(self.provenance))

    @property
    def implementation_fingerprint(self) -> str:
        """A stable identity for detecting duplicate domain implementations."""
        module = getattr(self.callable, "__module__", None)
        qualname = getattr(self.callable, "__qualname__", None)
        if not isinstance(module, str) or not isinstance(qualname, str):
            raise TypeError("operation callable must expose __module__ and __qualname__")
        return f"{module}:{qualname}"
