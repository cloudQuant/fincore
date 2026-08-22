"""Independent Extreme-Value-Theory reference formulas.

The helpers in this module intentionally do not import ``fincore`` so numerical
tests can distinguish a correct implementation from a production/oracle
common-mode error.  The GEV expected-shortfall reference uses direct numerical
quadrature over SciPy's documented distribution, rather than the closed form
implemented by the production kernel.
"""

from __future__ import annotations

import numpy as np
from scipy import integrate, stats

__all__ = [
    "gev_upper_tail_cvar_quadrature_reference",
    "gpd_pwm_reference",
    "hill_threshold_reference",
]


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


def gpd_pwm_reference(excesses: np.ndarray) -> tuple[float, float]:
    """Estimate GPD shape and scale from independent sample L-moments.

    For sorted excesses ``x_(i)``, this uses the empirical probability-weighted
    moment ``b1 = mean((i - 1)/(n - 1) * x_(i))`` and the first two L-moments
    ``l1 = b0`` and ``l2 = 2*b1 - b0``.  The GPD PWM estimates are
    ``xi = 2 - l1/l2`` and ``beta = l1*(1-xi)``.
    """
    values = np.asarray(excesses, dtype=float).reshape(-1)
    if len(values) < 2 or not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("PWM reference requires at least two finite non-negative excesses")

    ordered = np.sort(values)
    n = len(ordered)
    b0 = float(np.mean(ordered))
    b1 = float(np.mean((np.arange(n, dtype=float) / (n - 1.0)) * ordered))
    l2 = 2.0 * b1 - b0
    if not np.isfinite(l2) or l2 <= 0.0:
        raise ValueError("PWM reference requires a positive second L-moment")

    xi = 2.0 - b0 / l2
    beta = b0 * (1.0 - xi)
    if not np.isfinite(xi) or not np.isfinite(beta) or beta <= 0.0:
        raise ValueError("PWM reference produced an invalid GPD scale")
    return float(xi), float(beta)


def gev_upper_tail_cvar_quadrature_reference(*, xi: float, mu: float, sigma: float, alpha: float) -> float:
    """Return upper-tail GEV ES by integrating its PDF above its quantile.

    SciPy's ``genextreme`` has shape ``c=-xi``.  This is deliberately a PDF
    quadrature oracle instead of reusing a closed-form GEV ES expression.
    """
    if not np.isfinite((xi, mu, sigma, alpha)).all() or sigma <= 0.0 or not 0.0 < alpha < 1.0:
        raise ValueError("GEV reference requires finite parameters, sigma > 0, and alpha in (0, 1)")
    if xi >= 1.0:
        raise ValueError("GEV expected shortfall is infinite for xi >= 1")

    distribution = stats.genextreme(-xi, loc=mu, scale=sigma)
    var = float(distribution.ppf(1.0 - alpha))
    upper = np.inf if xi >= 0.0 else mu - sigma / xi
    numerator, _ = integrate.quad(
        lambda value: float(value * distribution.pdf(value)),
        var,
        upper,
        epsabs=1e-11,
        epsrel=1e-11,
        limit=200,
    )
    return float(numerator / alpha)
