"""Immutable OperationCatalog with unique keys and lazy reference resolution.

The catalog is the single semantic authority for the enhanced layer.  It is
built once from the frozen registries and then treated as read-only.  No
optional heavy dependency is imported at construction time.

A definition is uniquely identified by ``(operation_id, semantic_profile)``;
a binding is uniquely identified by ``public_path``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping

if TYPE_CHECKING:
    from fincore.api.specs import OperationDefinition, PublicBinding

__all__ = ["OperationCatalog"]


def _definition_key(operation_id: str, semantic_profile: str) -> tuple[str, str]:
    return (operation_id, semantic_profile)


@dataclass(frozen=True)
class OperationCatalog:
    """An immutable collection of operation definitions and public bindings."""

    definitions: tuple[OperationDefinition, ...]
    bindings: tuple[PublicBinding, ...]

    def __post_init__(self) -> None:
        self._validate_uniqueness()

    def _validate_uniqueness(self) -> None:
        def_keys: set[tuple[str, str]] = set()
        for definition in self.definitions:
            key = _definition_key(definition.operation_id, definition.semantic_profile)
            if key in def_keys:
                raise ValueError(f"duplicate operation_id+profile: {key}")
            def_keys.add(key)

        public_paths: set[str] = set()
        for binding in self.bindings:
            if binding.public_path in public_paths:
                raise ValueError(f"duplicate public_path: {binding.public_path}")
            public_paths.add(binding.public_path)
            if _definition_key(binding.operation_id, binding.semantic_profile) not in def_keys:
                raise ValueError(
                    f"binding {binding.binding_id} references unknown operation "
                    f"{binding.operation_id}@{binding.semantic_profile}"
                )

    @property
    def definition_map(self) -> Mapping[tuple[str, str], OperationDefinition]:
        return {
            _definition_key(d.operation_id, d.semantic_profile): d for d in self.definitions
        }

    @property
    def public_path_map(self) -> Mapping[str, PublicBinding]:
        return {b.public_path: b for b in self.bindings}

    def resolve_definition(self, operation_id: str, semantic_profile: str) -> OperationDefinition:
        try:
            return self.definition_map[_definition_key(operation_id, semantic_profile)]
        except KeyError as exc:
            raise KeyError(f"unknown operation {operation_id}@{semantic_profile}") from exc

    def resolve_binding(self, public_path: str) -> PublicBinding:
        try:
            return self.public_path_map[public_path]
        except KeyError as exc:
            raise KeyError(f"unknown public_path: {public_path}") from exc
