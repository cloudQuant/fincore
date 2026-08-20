"""Independent Kupiec proportion-of-failures (POF) likelihood-ratio oracle.

Source
------
Kupiec, P. (1995). "Techniques for Verifying the Accuracy of Risk
Measurement Models." *Journal of Derivatives* 3(2), 73--84.

Formula
-------
For ``n`` observations, ``x`` exceptions and coverage tail probability
``p = 1 - confidence_level``::

    LR_POF = -2 * ln[ (1-p)^(n-x) * p^x / ((1 - x/n)^(n-x) * (x/n)^x) ]
           =  2 * [ x*ln(x/(n*p)) + (n-x)*ln((n-x)/(n*(1-p))) ]

The statistic is asymptotically chi-squared(1) and, crucially, is
**non-negative by construction**.  The ``x=0`` and ``x=n`` cases are handled
by the continuous limit using ``scipy.special.xlogy`` (``0 * log(0) = 0``).

Units
-----
Pure statistic (dimensionless).  ``observations`` and ``exceptions`` are
integer counts, ``confidence_level`` is in ``(0, 1)``.

Boundary limits
---------------
* ``x = 0``  -> ``LR = -2 * n * ln(1 - p)`` (finite, non-negative)
* ``x = n``  -> ``LR =  2 * n * ln(1 / p)`` (finite, non-negative)

This oracle never imports ``fincore``.
"""

from __future__ import annotations

import numpy as np
from scipy import special

__all__ = ["kupiec_lr_reference"]


def kupiec_lr_reference(observations: int, exceptions: int, confidence_level: float) -> float:
    """Return the reference Kupiec POF LR statistic.

    Parameters
    ----------
    observations : int
        Number of observations ``n``.  Must be ``>= 1``.
    exceptions : int
        Number of VaR exceptions ``x``.  Must satisfy ``0 <= x <= n``.
    confidence_level : float
        Coverage level in ``(0, 1)``.

    Returns
    -------
    float
        ``LR_POF >= 0``.  Returns ``0.0`` when ``observations <= 0`` (mirroring
        the public API's no-data convention).
    """
    if observations <= 0:
        return 0.0
    n = int(observations)
    x = int(exceptions)
    p = 1.0 - float(confidence_level)
    # Guard against caller misuse (mirroring the public API's tolerance).
    if not 0.0 < p < 1.0:
        return float("nan")

    term1 = special.xlogy(x, x / (n * p))
    term2 = special.xlogy(n - x, (n - x) / (n * (1.0 - p)))
    return float(2.0 * (term1 + term2))


def kupiec_lr_brute_reference(observations: int, exceptions: int, confidence_level: float) -> float:
    """Brute-force log-likelihood reference (independent of the xlogy form).

    Uses ``math.log`` on the two likelihood terms directly, but only when the
    arguments are strictly positive; returns ``inf``/finite values identical
    to the xlogy form at the boundaries via explicit limits.  Exists so that
    the xlogy implementation is cross-checked against a *different* code path.
    """
    import math

    if observations <= 0:
        return 0.0
    n = int(observations)
    x = int(exceptions)
    p = 1.0 - float(confidence_level)
    if not 0.0 < p < 1.0:
        return float("nan")

    if x == 0:
        return float(-2.0 * n * math.log1p(-p))
    if x == n:
        return float(2.0 * n * math.log(1.0 / p))

    observed = x / n
    term1 = x * math.log(observed / p)
    term2 = (n - x) * math.log((1.0 - observed) / (1.0 - p))
    return float(2.0 * (term1 + term2))
