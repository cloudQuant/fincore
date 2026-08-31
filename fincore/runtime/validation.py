"""Generic, domain-neutral validation primitives for canonical runtime code."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np

from fincore.exceptions import DependencyError

if TYPE_CHECKING:
    from types import ModuleType


def load_optional_module(
    module_name: str,
    *,
    dependency: str,
    extra: str | None = None,
    message: str | None = None,
) -> ModuleType:
    """Import one optional module only at its explicit capability boundary."""

    try:
        return importlib.import_module(module_name)
    except Exception as error:
        raise DependencyError(
            message or f"optional_dependency_missing: {dependency} is required",
            dependency=dependency,
            extra=extra,
        ) from error


def validate_mapping(value: Mapping[str, Any], *, name: str) -> dict[str, Any]:
    """Return a shallow mapping copy after enforcing stable string input names."""
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    if any(not isinstance(key, str) or not key for key in value):
        raise ValueError(f"{name} keys must be non-empty strings")
    return dict(value)


def validate_finite_array(
    value: Any,
    *,
    name: str,
    min_size: int = 1,
    ndim: int | None = None,
) -> np.ndarray:
    """Copy a dense numeric input and enforce generic shape/finite invariants."""
    if not isinstance(min_size, int) or min_size < 0:
        raise ValueError("min_size must be a non-negative integer")
    if ndim is not None and (not isinstance(ndim, int) or ndim < 0):
        raise ValueError("ndim must be a non-negative integer or None")
    try:
        array = np.array(value, dtype=float, copy=True)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be convertible to a numeric array") from error
    if array.ndim == 0:
        raise ValueError(f"{name} must be an array, not a scalar")
    if ndim is not None and array.ndim != ndim:
        dimensions = "one-dimensional" if ndim == 1 else f"{ndim}-dimensional"
        raise ValueError(f"{name} must be {dimensions}")
    if array.size < min_size:
        raise ValueError(f"{name} must contain at least {min_size} values")
    if not bool(np.isfinite(array).all()):
        raise ValueError(f"{name} must contain only finite values")
    return array
