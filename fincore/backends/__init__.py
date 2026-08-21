"""Optional numerical execution backends.

The pandas/NumPy reference backend is the default.  Optional backends (e.g.
Array API) cover only dense kernels that do not depend on labels, timezones or
calendars; they are opt-in, fall back to the reference, and record their
backend/version in results.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import numpy as np

from fincore.backends.numpy_backend import NumPyBackend

__all__ = ["Backend", "NumPyBackend", "get_backend", "reference_backend"]


class Backend(Protocol):
    """A numerical backend exposing dense kernels."""

    name: str
    version: str

    def cum_returns(self, returns: np.ndarray) -> np.ndarray: ...
    def max_drawdown(self, returns: np.ndarray) -> float | np.ndarray: ...
    def sharpe_ratio(self, returns: np.ndarray, periods_per_year: int = 252) -> float: ...


reference_backend: Backend = NumPyBackend()

_BACKENDS: dict[str, Backend] = {"numpy": reference_backend}


def get_backend(name: str = "numpy") -> Backend:
    """Return a backend by name, falling back to the reference."""
    return _BACKENDS.get(name, reference_backend)
