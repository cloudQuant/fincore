"""Small, explicit rolling-array primitives shared by metric kernels."""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def rolling_window(value: np.ndarray, window: int) -> np.ndarray:
    """Return a zero-copy 1-D rolling window view with validated bounds."""
    array = np.asarray(value)
    if array.ndim != 1:
        raise ValueError("rolling_window requires a one-dimensional array")
    if not isinstance(window, int) or window <= 0:
        raise ValueError("window must be a positive integer")
    if window > array.size:
        raise ValueError("window cannot exceed the input length")
    return sliding_window_view(array, window_shape=window)
