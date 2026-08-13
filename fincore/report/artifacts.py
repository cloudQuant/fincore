"""Backend-neutral handles returned by report and context rendering."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from pathlib import Path

    from fincore.report.model import ReportModel

__all__ = ["ReportArtifacts"]


@dataclass
class ReportArtifacts:
    """Rendered in-memory objects and explicitly exported files.

    Task 7 establishes the common lifecycle boundary.  Task 8 extends the
    same object for the full report model/render pipeline: ``model`` carries
    the computed-once :class:`~fincore.report.model.ReportModel` when a
    renderer consumed a precomputed model instead of computing its own.
    """

    backend: str
    figures: list[Any] = field(default_factory=list)
    files: list[Path] = field(default_factory=list)
    html: str | None = None
    model: ReportModel | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    _closed_resource_ids: set[int] = field(default_factory=set, init=False, repr=False)

    @property
    def closed(self) -> bool:
        """Whether every currently owned figure has been released."""

        return all(id(getattr(item, "figure", item)) in self._closed_resource_ids for item in self.figures)

    def __enter__(self) -> ReportArtifacts:
        return self

    def __exit__(self, exc_type: Any, exc: BaseException | None, traceback: Any) -> Literal[False]:
        try:
            self.close()
        except Exception as close_error:
            if exc is None:
                raise
            if hasattr(exc, "add_note"):
                exc.add_note(f"ReportArtifacts.close() also failed: {close_error!r}")
        return False

    def close(self) -> None:
        """Release figures owned by this result where the backend supports it."""

        seen: set[int] = set()
        first_error: Exception | None = None
        for item in self.figures:
            figure = getattr(item, "figure", item)
            identifier = id(figure)
            if identifier in seen or identifier in self._closed_resource_ids:
                continue
            seen.add(identifier)
            try:
                import matplotlib.figure
                import matplotlib.pyplot as plt
            except ImportError:
                matplotlib = None  # type: ignore[assignment]
            try:
                if matplotlib is not None and isinstance(figure, matplotlib.figure.Figure):
                    plt.close(figure)
                else:
                    close = getattr(figure, "close", None)
                    if callable(close):
                        close()
                self._closed_resource_ids.add(identifier)
            except Exception as error:  # noqa: BLE001 -- third-party resources define arbitrary close failures
                if first_error is None:
                    first_error = error
        if first_error is not None:
            raise first_error
