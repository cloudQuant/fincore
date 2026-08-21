"""Independent BHB multi-period Brinson oracle with Carino linking.

This module is a standalone NumPy/math reference: it never imports
``fincore`` or calls production attribution functions. It computes the BHB
allocation, selection, and interaction effects directly before applying the
Carino (1999) linking transformation.

Provenance
----------
Brinson, Hood & Beebower (1986), *Financial Analysts Journal* 42(5), 39--44;
Carino, D. R. (1999), "Combining Attribution Effects Over Time," *Journal of
Performance Measurement* 3(4), 5--14.

Formula
-------
For period ``t`` with portfolio and benchmark returns ``r^p_t`` and
``r^b_t``::

    allocation_t  = sum_i (w^p_i - w^b_i) r^b_i
    selection_t   = sum_i w^b_i (r^p_i - r^b_i)
    interaction_t = sum_i (w^p_i - w^b_i) (r^p_i - r^b_i)

    k_t = [ln(1+r^p_t) - ln(1+r^b_t)] / (r^p_t-r^b_t)
    K   = [ln(1+R_p) - ln(1+R_b)] / (R_p-R_b)
    E   = sum_t (k_t / K) E_t

Here ``R_p`` and ``R_b`` are compounded portfolio and benchmark returns. The
linked effects reconcile to the BHB cumulative active return ``R_p - R_b``.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = ["brinson_carino_reference"]


def _carino_k(rp: float, rb: float) -> float:
    """Return the Carino coefficient, including its equal-return limit."""
    if abs(rp - rb) < 1e-15:
        return 1.0 / (1.0 + rp)
    return (math.log1p(rp) - math.log1p(rb)) / (rp - rb)


def _validate_carino_returns(values: np.ndarray, *, label: str) -> None:
    if not np.all(np.isfinite(values)) or np.any(values <= -1.0):
        raise ValueError(f"{label} must be finite and greater than -1 for Carino linking.")


def brinson_carino_reference(
    portfolio_returns: np.ndarray,
    benchmark_returns: np.ndarray,
    portfolio_weights: np.ndarray,
    benchmark_weights: np.ndarray,
) -> dict[str, float]:
    """Return independent BHB effects linked to cumulative active return."""
    rp = np.asarray(portfolio_returns, dtype=float)
    rb = np.asarray(benchmark_returns, dtype=float)
    wp = np.asarray(portfolio_weights, dtype=float)
    wb = np.asarray(benchmark_weights, dtype=float)

    if rp.ndim == 1:
        rp = rp[None, :]
        rb = rb[None, :]
    if wp.ndim == 1:
        wp = np.tile(wp, (rp.shape[0], 1))
    if wb.ndim == 1:
        wb = np.tile(wb, (rp.shape[0], 1))
    if not (rp.shape == rb.shape == wp.shape == wb.shape):
        raise ValueError("portfolio returns, benchmark returns, and weights must have consistent shapes.")

    _validate_carino_returns(rp, label="portfolio returns")
    _validate_carino_returns(rb, label="benchmark returns")
    if not np.all(np.isfinite(wp)) or not np.all(np.isfinite(wb)):
        raise ValueError("portfolio and benchmark weights must be finite.")

    portfolio_period = np.sum(wp * rp, axis=1)
    benchmark_period = np.sum(wb * rb, axis=1)
    _validate_carino_returns(portfolio_period, label="portfolio period returns")
    _validate_carino_returns(benchmark_period, label="benchmark period returns")

    allocation = np.sum((wp - wb) * rb, axis=1)
    selection = np.sum(wb * (rp - rb), axis=1)
    interaction = np.sum((wp - wb) * (rp - rb), axis=1)

    portfolio_cumulative = float(np.prod(1.0 + portfolio_period) - 1.0)
    benchmark_cumulative = float(np.prod(1.0 + benchmark_period) - 1.0)
    global_k = _carino_k(portfolio_cumulative, benchmark_cumulative)
    period_k = np.array(
        [
            _carino_k(portfolio_return, benchmark_return)
            for portfolio_return, benchmark_return in zip(portfolio_period, benchmark_period, strict=True)
        ]
    )
    scale = period_k / global_k

    allocation_cumulative = float(np.sum(scale * allocation))
    selection_cumulative = float(np.sum(scale * selection))
    interaction_cumulative = float(np.sum(scale * interaction))
    active_return = portfolio_cumulative - benchmark_cumulative
    return {
        "allocation": allocation_cumulative,
        "selection": selection_cumulative,
        "interaction": interaction_cumulative,
        "total": allocation_cumulative + selection_cumulative + interaction_cumulative,
        "portfolio_cumulative": portfolio_cumulative,
        "benchmark_cumulative": benchmark_cumulative,
        "active_return": active_return,
    }
