"""ArtifactBundle protocol with idempotent close and context-manager support.

All renderer artifacts implement a close that is idempotent (calling it twice
is a no-op) and exception-safe, and support ``with ... as artifacts:`` so
resources are released deterministically.
"""

from __future__ import annotations

from typing import Any, Protocol

__all__ = ["ArtifactBundle", "IdempotentCloseMixin"]


class ArtifactBundle(Protocol):
    """A resource bundle that owns files/figures and supports idempotent close."""

    def close(self) -> None:
        """Release owned resources; idempotent."""

    def __enter__(self) -> ArtifactBundle:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()


class IdempotentCloseMixin:
    """Mixin providing idempotent, exception-safe close semantics."""

    _closed: bool = False

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._close_resources()
        finally:
            self._closed = True

    def _close_resources(self) -> None:  # pragma: no cover - override hook
        raise NotImplementedError

    @property
    def closed(self) -> bool:
        return self._closed

    def __enter__(self) -> IdempotentCloseMixin:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()
