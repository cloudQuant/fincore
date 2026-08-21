"""Independent multi-period Brinson linking oracle (Carino method).

The per-period Brinson effects are arithmetic and reconcile within a single
period, but multi-period attribution must compound geometrically.  The Carino
linking constant maps arithmetic period effects onto the geometric active
return.

Source
------
Carino, D. R. (1999). "Combining Attribution Effects Over Time." *Journal of
Performance Measurement* 3(4), 5--14.

Formula
-------
For period ``t`` with portfolio return ``r^p_t`` and benchmark return
``r^b_t``::

    k_t = [ln(1+r^p_t) - ln(1+r^b_t)] / (r^p_t - r^b_t)   if r^p_t != r^b_t
    k_t = 1 / (1 + r^p_t)                                  otherwise

    A = product_t (1 + r^p_t) / product_t (1 + r^b_t) - 1
    K = [ln(1 + A) - ln(1 + 0)] / (A - 0)

    E_cum = sum_t (k_t / K) * E_t      (E in {allocation, selection, interaction})

The cumulative effects then reconcile directly to the relative geometric
active return::

    (1+R_p)/(1+R_b) - 1 = allocation_cum + selection_cum + interaction_cum

This oracle never imports ``fincore``.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = ["carino_linking_reference"]


def _carino_k(rp: float, rb: float) -> float:
    if abs(rp - rb) < 1e-15:
        return 1.0 / (1.0 + rp)
    return (math.log1p(rp) - math.log1p(rb)) / (rp - rb)


def carino_linking_reference(
    period_effects: dict[str, np.ndarray],
    portfolio_period_returns: np.ndarray,
    benchmark_period_returns: np.ndarray,
) -> dict[str, float]:
    """Return cumulative linked Brinson effects.

    Parameters
    ----------
    period_effects : dict[str, np.ndarray]
        Per-period effects keyed by ``allocation``/``selection``/``interaction``.
    portfolio_period_returns : np.ndarray
        Per-period portfolio returns ``r^p_t``.
    benchmark_period_returns : np.ndarray
        Per-period benchmark returns ``r^b_t``.

    Returns
    -------
    dict[str, float]
        Cumulative linked effects plus the geometric active return under the
        key ``active_return``.
    """
    rp = np.asarray(portfolio_period_returns, dtype=float)
    rb = np.asarray(benchmark_period_returns, dtype=float)
    rp_cum = float(np.prod(1.0 + rp) - 1.0)
    rb_cum = float(np.prod(1.0 + rb) - 1.0)
    active_return = float((1.0 + rp_cum) / (1.0 + rb_cum) - 1.0)
    period_k = np.array([_carino_k(a, b) for a, b in zip(rp, rb, strict=True)])
    global_k = _carino_k(active_return, 0.0)

    out: dict[str, float] = {}
    for name, effects in period_effects.items():
        e = np.asarray(effects, dtype=float)
        out[name] = float(np.sum((period_k / global_k) * e))

    out["active_return"] = active_return
    return out
