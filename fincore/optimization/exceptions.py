"""Optimization-domain errors exposed from their owning module."""

from __future__ import annotations

__all__ = ["OptimizationError"]


class OptimizationError(Exception):
    """Raised when a solver fails or produces an invalid allocation."""

    def __init__(
        self,
        message: str,
        status: int | None = None,
        solver_message: str | None = None,
    ) -> None:
        self.status = status
        self.solver_message = solver_message
        super().__init__(message)
