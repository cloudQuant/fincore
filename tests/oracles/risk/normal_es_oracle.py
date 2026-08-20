"""Independent analytic VaR / Expected-Shortfall oracles.

These are closed-form references for zero-mean normal and Student-t
distributions under the ``losses_negative`` convention (negative values are
losses, so ES <= VaR <= 0 for a coverage tail ``alpha = 1 - confidence``).

Normal
------
With ``z = Phi^{-1}(alpha)`` and ``phi`` the standard-normal density::

    VaR_alpha   = sigma * z
    ES_alpha    = -sigma * phi(z) / alpha

Student-t
---------
With ``t = F_nu^{-1}(alpha)`` (``scipy.stats.t.ppf``) and ``f`` the t density::

    VaR_alpha   = -t
    ES_alpha    = -f(t) / alpha * (nu + t**2) / (nu - 1)

All references use only NumPy/SciPy and never import ``fincore``.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

__all__ = ["normal_es_reference", "normal_var_reference", "student_t_es_reference", "student_t_var_reference"]


def normal_var_reference(sigma: float, confidence_level: float) -> float:
    """Analytic zero-mean normal VaR (negative under losses_negative)."""
    alpha = 1.0 - confidence_level
    z = float(stats.norm.ppf(alpha))
    return float(sigma * z)


def normal_es_reference(sigma: float, confidence_level: float) -> float:
    """Analytic zero-mean normal Expected Shortfall (negative)."""
    alpha = 1.0 - confidence_level
    z = float(stats.norm.ppf(alpha))
    return float(-sigma * stats.norm.pdf(z) / alpha)


def student_t_var_reference(sigma: float, confidence_level: float, nu: float) -> float:
    """Analytic scaled Student-t VaR (negative). ``sigma`` is the scale."""
    alpha = 1.0 - confidence_level
    t = float(stats.t.ppf(alpha, nu))
    return float(sigma * t)


def student_t_es_reference(sigma: float, confidence_level: float, nu: float) -> float:
    """Analytic scaled Student-t Expected Shortfall (negative)."""
    alpha = 1.0 - confidence_level
    t = float(stats.t.ppf(alpha, nu))
    density = float(stats.t.pdf(t, nu))
    return float(-sigma * density / alpha * (nu + t**2) / (nu - 1.0))


def normal_es_at_var(sigma: float, confidence_level: float) -> tuple[float, float]:
    """Return ``(var, es)`` analytic pair for zero-mean normal."""
    return normal_var_reference(sigma, confidence_level), normal_es_reference(sigma, confidence_level)
