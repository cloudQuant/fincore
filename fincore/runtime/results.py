"""Result values emitted by canonical runtime execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class Result:
    """The natural domain value plus immutable orchestration provenance."""

    value: Any
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def with_metadata(self, **updates: Any) -> Result:
        """Return the same natural value with explicitly updated provenance."""
        return Result(value=self.value, metadata={**self.metadata, **updates})
