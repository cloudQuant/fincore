"""Independent Extreme-Value-Theory reference formulas.

The helpers in this module intentionally use only NumPy.  They never import
``fincore`` so numerical tests can distinguish a correct implementation from
a production/oracle common-mode error.
"""

from __future__ import annotations

import numpy as np

__all__ = ["hill_threshold_reference"]


def hill_threshold_reference(data: np.ndarray, *, threshold: float, tail: str) -> tuple[float, np.ndarray]:
    """Return the threshold Hill estimate and the selected tail observations.

    For positive tail magnitudes ``x_i > u``, the threshold form of the Hill
    estimator is ``mean(log(x_i / u))``.  A lower-return tail is reflected into
    positive loss magnitudes before applying the same formula.
    """
    values = np.asarray(data, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("threshold must be finite and positive")

    if tail == "upper":
        magnitudes = values[values > 0.0]
    elif tail == "lower":
        magnitudes = -values[values < 0.0]
    else:
        raise ValueError("tail must be 'upper' or 'lower'")

    observations = magnitudes[magnitudes > threshold]
    if len(observations) < 10:
        raise ValueError("Not enough exceedances for Hill estimation (need >= 10)")
    return float(np.mean(np.log(observations / threshold))), observations
