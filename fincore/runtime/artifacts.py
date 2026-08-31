"""Explicit ownership and lifecycle management for rendered artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Mapping

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(frozen=True, slots=True)
class _ArtifactEntry:
    name: str | None
    value: Any
    owned: bool
    closer: Callable[[], None] | None


class ArtifactBundle:
    """Collect render outputs and close only resources created by Fincore.

    The caller declares ownership per resource when registering it.  A caller
    supplied axes object can therefore travel with a report without becoming a
    runtime-owned resource, while a Fincore-created figure is released exactly
    once when the bundle closes.
    """

    def __init__(self, *, metadata: Mapping[str, Any] | None = None) -> None:
        if metadata is not None and not isinstance(metadata, Mapping):
            raise TypeError("metadata must be a mapping or None")
        if metadata is not None and any(not isinstance(key, str) or not key for key in metadata):
            raise TypeError("metadata keys must be non-empty strings")
        self._entries: list[_ArtifactEntry] = []
        self._named_artifacts: dict[str, Any] = {}
        self._metadata = MappingProxyType(dict(metadata or {}))
        self._closed = False

    @property
    def artifacts(self) -> tuple[Any, ...]:
        """Return registered outputs in their production order."""
        return tuple(entry.value for entry in self._entries)

    @property
    def closed(self) -> bool:
        """Whether this bundle has completed its one lifecycle close attempt."""
        return self._closed

    @property
    def metadata(self) -> Mapping[str, Any]:
        """Return immutable renderer/runtime metadata for this artifact bundle."""

        return self._metadata

    @property
    def named_artifacts(self) -> Mapping[str, Any]:
        """Return named artifacts without exposing lifecycle internals for mutation."""

        return MappingProxyType(dict(self._named_artifacts))

    def add(
        self,
        artifact: Any,
        *,
        owned: bool,
        closer: Callable[[], None] | None = None,
        name: str | None = None,
    ) -> Any:
        """Register one output with explicit resource ownership.

        When ``owned`` is true and no closer is provided, a callable ``close``
        attribute on the artifact is used.  Outputs without a close method are
        valid inert artifacts such as tables, HTML strings, or file paths.
        """
        if self._closed:
            raise RuntimeError("artifact bundle is closed")
        if not isinstance(owned, bool):
            raise TypeError("owned must be a bool")
        if closer is not None and not callable(closer):
            raise TypeError("closer must be callable")
        if name is not None and (not isinstance(name, str) or not name):
            raise TypeError("name must be a non-empty string or None")
        if name is not None and name in self._named_artifacts:
            raise ValueError(f"duplicate artifact name: {name}")
        resolved_closer = closer
        if owned and resolved_closer is None:
            candidate = getattr(artifact, "close", None)
            if callable(candidate):
                resolved_closer = candidate
        self._entries.append(_ArtifactEntry(name=name, value=artifact, owned=owned, closer=resolved_closer))
        if name is not None:
            self._named_artifacts[name] = artifact
        return artifact

    def close(self) -> None:
        """Close each unique owned resource once, then permanently seal the bundle."""
        if self._closed:
            return
        self._closed = True
        closed_resource_ids: set[int] = set()
        first_error: Exception | None = None
        for entry in self._entries:
            if not entry.owned or entry.closer is None:
                continue
            resource_id = id(entry.value)
            if resource_id in closed_resource_ids:
                continue
            closed_resource_ids.add(resource_id)
            try:
                entry.closer()
            except Exception as error:  # noqa: BLE001 - renderer implementations control their exception types.
                if first_error is None:
                    first_error = error
        if first_error is not None:
            raise first_error

    def __enter__(self) -> ArtifactBundle:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()
