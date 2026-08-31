"""Generic dense numerical primitives used by domain-owned calculations."""

from __future__ import annotations

from typing import Any

import numpy as np

from .validation import validate_finite_array


class NumPyBackend:
    """Small explicit NumPy backend with no financial-domain formulas."""

    name = "numpy"
    version = np.__version__

    def as_float_array(self, value: Any, *, name: str = "value", ndim: int | None = None) -> np.ndarray:
        """Copy and validate a dense floating-point input."""
        return validate_finite_array(value, name=name, ndim=ndim)

    def cumulative_product(self, value: Any) -> np.ndarray:
        """Return the generic cumulative product of a one-dimensional array."""
        return np.cumprod(self.as_float_array(value, ndim=1))

    def sample_standard_deviation(self, value: Any) -> float:
        """Return the unbiased sample standard deviation of a dense vector."""
        array = self.as_float_array(value, ndim=1)
        if array.size < 2:
            raise ValueError("value must contain at least 2 values for sample standard deviation")
        return float(np.std(array, ddof=1))
