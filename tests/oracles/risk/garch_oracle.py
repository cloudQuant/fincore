"""Independent GARCH-family recursion oracle.

Validates the *variance recursion* and *forecast recursion* of GARCH-family
models without calling ``fincore``.  The MLE fit itself is validated through
fixed, seeded fixtures (documented provenance in
``docs/quality/numerical-oracle-register.md``) rather than a second optimizer,
since the ``arch`` reference package is not importable in this environment.

Recursions verified here (all with fixed, user-supplied parameters):

* GARCH(1,1): ``s2[t] = omega + alpha*eps[t-1]^2 + beta*s2[t-1]``
* GJR(1,1):   ``s2[t] = omega + alpha*eps[t-1]^2 + gamma*I(eps<0)*eps[t-1]^2 + beta*s2[t-1]``
* EGARCH(1,1): ``log s2[t] = omega + alpha*|z[t-1]| + gamma*z[t-1] + beta*log s2[t-1]``
  where ``z[t-1] = eps[t-1] / sqrt(s2[t-1])`` is the previous **conditional**
  standardized innovation.

Forecast one-step and multi-step recurrences are also implemented so that the
EGARCH/GJR forecast paths can be checked independently of the GARCH path.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "egarch_conditional_var_reference",
    "egarch_forecast_reference",
    "garch_conditional_var_reference",
    "garch_forecast_reference",
    "gjr_conditional_var_reference",
    "gjr_forecast_reference",
]


def garch_conditional_var_reference(eps: np.ndarray, omega: float, alpha: float, beta: float) -> np.ndarray:
    """GARCH(1,1) conditional-variance recursion (independent reference)."""
    eps = np.asarray(eps, dtype=float)
    t = len(eps)
    s2 = np.empty(t, dtype=float)
    s2[0] = float(np.var(eps))
    for i in range(1, t):
        s2[i] = omega + alpha * eps[i - 1] ** 2 + beta * s2[i - 1]
    return s2


def gjr_conditional_var_reference(eps: np.ndarray, omega: float, alpha: float, gamma: float, beta: float) -> np.ndarray:
    """GJR-GARCH(1,1) conditional-variance recursion (independent reference)."""
    eps = np.asarray(eps, dtype=float)
    t = len(eps)
    s2 = np.empty(t, dtype=float)
    s2[0] = float(np.var(eps))
    for i in range(1, t):
        indicator = 1.0 if eps[i - 1] < 0 else 0.0
        s2[i] = omega + (alpha + gamma * indicator) * eps[i - 1] ** 2 + beta * s2[i - 1]
    return s2


def egarch_conditional_var_reference(
    eps: np.ndarray, omega: float, alpha: float, gamma: float, beta: float
) -> np.ndarray:
    """EGARCH(1,1) recursion with conditional standardized innovations."""
    eps = np.asarray(eps, dtype=float)
    t = len(eps)
    if t == 0:
        return np.empty(0, dtype=float)

    initial_s2 = float(np.var(eps))
    if initial_s2 <= 0.0:
        raise ValueError("EGARCH reference requires non-zero sample variance")

    log_s2 = np.empty(t, dtype=float)
    s2 = np.empty(t, dtype=float)
    s2[0] = initial_s2
    log_s2[0] = np.log(initial_s2)
    for i in range(1, t):
        z_prev = eps[i - 1] / np.sqrt(s2[i - 1])
        log_s2[i] = omega + alpha * np.abs(z_prev) + gamma * z_prev + beta * log_s2[i - 1]
        s2[i] = np.exp(log_s2[i])
    return s2


def garch_forecast_reference(
    omega: float, alpha: float, beta: float, last_s2: float, last_eps_sq: float, horizon: int
) -> np.ndarray:
    """GARCH(1,1) variance forecast (one-step then persistence recursion)."""
    out = np.empty(horizon, dtype=float)
    persistence = alpha + beta
    out[0] = omega + alpha * last_eps_sq + beta * last_s2
    for h in range(1, horizon):
        out[h] = omega + persistence * out[h - 1]
    return out


def gjr_forecast_reference(
    omega: float,
    alpha: float,
    gamma: float,
    beta: float,
    last_s2: float,
    last_eps: float,
    horizon: int,
) -> np.ndarray:
    """GJR-GARCH(1,1) variance forecast.

    Under normality, E[I(eps<0) * eps^2] = 0.5 * s2, giving the persistence
    ``alpha + 0.5*gamma + beta``.
    """
    out = np.empty(horizon, dtype=float)
    indicator = 1.0 if last_eps < 0 else 0.0
    out[0] = omega + (alpha + gamma * indicator) * last_eps**2 + beta * last_s2
    persistence = alpha + 0.5 * gamma + beta
    for h in range(1, horizon):
        out[h] = omega + persistence * out[h - 1]
    return out


def egarch_forecast_reference(
    omega: float,
    alpha: float,
    gamma: float,
    beta: float,
    last_log_s2: float,
    last_z: float,
    horizon: int,
) -> np.ndarray:
    """EGARCH(1,1) log-variance forecast.

    ``E[alpha*|z| + gamma*z] = alpha*sqrt(2/pi)`` under normality, so the
    forecast decays toward ``omega / (1 - beta)``.
    """
    out = np.empty(horizon, dtype=float)
    out[0] = omega + alpha * np.abs(last_z) + gamma * last_z + beta * last_log_s2
    drift = alpha * np.sqrt(2.0 / np.pi)
    for h in range(1, horizon):
        out[h] = omega + drift + beta * out[h - 1]
    return np.exp(out)
