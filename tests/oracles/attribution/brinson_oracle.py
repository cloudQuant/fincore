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
from decimal import Decimal, localcontext

import numpy as np

__all__ = [
    "brinson_bhb_reference",
    "brinson_carino_decimal_reference",
    "brinson_carino_reference",
    "carino_k_decimal_reference",
]


def _carino_k(rp: float, rb: float) -> float:
    """Return the Carino coefficient, including its equal-return limit."""
    if rp == rb:
        return 1.0 / (1.0 + rp)
    return math.log1p((rp - rb) / (1.0 + rb)) / (rp - rb)


def carino_k_decimal_reference(rp: float, rb: float) -> float:
    """Return a high-precision Carino coefficient for boundary fixtures.

    ``Decimal.from_float`` intentionally retains the exact binary64 values
    received by the public API, so this oracle detects cancellation in a
    float implementation instead of comparing two differently rounded input
    values.
    """
    with localcontext() as context:
        context.prec = 80
        portfolio_return = Decimal.from_float(float(rp))
        benchmark_return = Decimal.from_float(float(rb))
        return float(_decimal_carino_k(portfolio_return, benchmark_return))


def _decimal_carino_k(portfolio_return: Decimal, benchmark_return: Decimal) -> Decimal:
    """Decimal implementation of Carino's coefficient and equality limit."""
    if portfolio_return == benchmark_return:
        return Decimal(1) / (Decimal(1) + portfolio_return)
    return ((Decimal(1) + portfolio_return).ln() - (Decimal(1) + benchmark_return).ln()) / (
        portfolio_return - benchmark_return
    )


def _validate_finite(values: np.ndarray, *, label: str) -> None:
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{label} must be finite.")


def _validate_carino_period_returns(values: np.ndarray, *, label: str) -> None:
    if not np.all(np.isfinite(values)) or np.any(values <= -1.0):
        raise ValueError(f"{label} must be finite and greater than -1 for Carino linking.")


def _validate_decimal_carino_period_returns(values: list[Decimal], *, label: str) -> None:
    if any(not value.is_finite() or value <= Decimal(-1) for value in values):
        raise ValueError(f"{label} must be finite and greater than -1 for Carino linking.")


def brinson_bhb_reference(
    portfolio_returns: np.ndarray,
    benchmark_returns: np.ndarray,
    portfolio_weights: np.ndarray,
    benchmark_weights: np.ndarray,
) -> dict[str, float]:
    """Return an independent one-period Brinson--Hood--Beebower decomposition.

    This intentionally uses a standalone NumPy calculation rather than the
    production helper.  The three effects are the BHB arithmetic components;
    their sum must equal the portfolio-minus-benchmark return for the period.
    """
    rp = np.asarray(portfolio_returns, dtype=float).reshape(-1)
    rb = np.asarray(benchmark_returns, dtype=float).reshape(-1)
    wp = np.asarray(portfolio_weights, dtype=float).reshape(-1)
    wb = np.asarray(benchmark_weights, dtype=float).reshape(-1)
    if not (rp.shape == rb.shape == wp.shape == wb.shape):
        raise ValueError("portfolio returns, benchmark returns, and weights must have the same shape.")
    _validate_finite(rp, label="portfolio returns")
    _validate_finite(rb, label="benchmark returns")
    _validate_finite(wp, label="portfolio weights")
    _validate_finite(wb, label="benchmark weights")

    allocation = float(np.sum((wp - wb) * rb))
    selection = float(np.sum(wb * (rp - rb)))
    interaction = float(np.sum((wp - wb) * (rp - rb)))
    portfolio_return = float(np.sum(wp * rp))
    benchmark_return = float(np.sum(wb * rb))
    return {
        "allocation": allocation,
        "selection": selection,
        "interaction": interaction,
        "total": allocation + selection + interaction,
        "portfolio_return": portfolio_return,
        "benchmark_return": benchmark_return,
    }


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

    _validate_finite(rp, label="portfolio returns")
    _validate_finite(rb, label="benchmark returns")
    _validate_finite(wp, label="portfolio weights")
    _validate_finite(wb, label="benchmark weights")

    portfolio_period = np.sum(wp * rp, axis=1)
    benchmark_period = np.sum(wb * rb, axis=1)
    _validate_carino_period_returns(portfolio_period, label="portfolio period returns")
    _validate_carino_period_returns(benchmark_period, label="benchmark period returns")

    allocation = np.sum((wp - wb) * rb, axis=1)
    selection = np.sum(wb * (rp - rb), axis=1)
    interaction = np.sum((wp - wb) * (rp - rb), axis=1)

    portfolio_cumulative = float(np.prod(1.0 + portfolio_period) - 1.0)
    benchmark_cumulative = float(np.prod(1.0 + benchmark_period) - 1.0)
    _validate_carino_period_returns(
        np.array([portfolio_cumulative, benchmark_cumulative]), label="cumulative portfolio and benchmark returns"
    )
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


def brinson_carino_decimal_reference(
    portfolio_returns: np.ndarray,
    benchmark_returns: np.ndarray,
    portfolio_weights: np.ndarray,
    benchmark_weights: np.ndarray,
) -> dict[str, float]:
    """Return a high-precision BHB/Carino reference for loss-boundary cases.

    The production API accepts binary64 inputs.  This implementation keeps
    those exact values, but evaluates aggregation, compounding, and Carino
    coefficients with 80-digit Decimal arithmetic.  It is deliberately kept
    separate from the NumPy oracle above so boundary tests have an independent
    numerical path.
    """
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

    _validate_finite(rp, label="portfolio returns")
    _validate_finite(rb, label="benchmark returns")
    _validate_finite(wp, label="portfolio weights")
    _validate_finite(wb, label="benchmark weights")

    with localcontext() as context:
        context.prec = 80
        decimal = Decimal.from_float
        portfolio_period = [
            sum(
                (decimal(float(weight)) * decimal(float(ret)) for weight, ret in zip(weights, returns, strict=True)),
                Decimal(0),
            )
            for weights, returns in zip(wp, rp, strict=True)
        ]
        benchmark_period = [
            sum(
                (decimal(float(weight)) * decimal(float(ret)) for weight, ret in zip(weights, returns, strict=True)),
                Decimal(0),
            )
            for weights, returns in zip(wb, rb, strict=True)
        ]
        _validate_decimal_carino_period_returns(portfolio_period, label="portfolio period returns")
        _validate_decimal_carino_period_returns(benchmark_period, label="benchmark period returns")

        allocation = [
            sum(
                (
                    (decimal(float(portfolio_weight)) - decimal(float(benchmark_weight)))
                    * decimal(float(benchmark_return))
                    for portfolio_weight, benchmark_weight, benchmark_return in zip(
                        portfolio_weights_row, benchmark_weights_row, benchmark_returns_row, strict=True
                    )
                ),
                Decimal(0),
            )
            for portfolio_weights_row, benchmark_weights_row, benchmark_returns_row in zip(wp, wb, rb, strict=True)
        ]
        selection = [
            sum(
                (
                    decimal(float(benchmark_weight))
                    * (decimal(float(portfolio_return)) - decimal(float(benchmark_return)))
                    for benchmark_weight, portfolio_return, benchmark_return in zip(
                        benchmark_weights_row, portfolio_returns_row, benchmark_returns_row, strict=True
                    )
                ),
                Decimal(0),
            )
            for benchmark_weights_row, portfolio_returns_row, benchmark_returns_row in zip(wb, rp, rb, strict=True)
        ]
        interaction = [
            sum(
                (
                    (decimal(float(portfolio_weight)) - decimal(float(benchmark_weight)))
                    * (decimal(float(portfolio_return)) - decimal(float(benchmark_return)))
                    for portfolio_weight, benchmark_weight, portfolio_return, benchmark_return in zip(
                        portfolio_weights_row,
                        benchmark_weights_row,
                        portfolio_returns_row,
                        benchmark_returns_row,
                        strict=True,
                    )
                ),
                Decimal(0),
            )
            for portfolio_weights_row, benchmark_weights_row, portfolio_returns_row, benchmark_returns_row in zip(
                wp, wb, rp, rb, strict=True
            )
        ]

        portfolio_cumulative = math.prod(Decimal(1) + value for value in portfolio_period) - Decimal(1)
        benchmark_cumulative = math.prod(Decimal(1) + value for value in benchmark_period) - Decimal(1)
        _validate_decimal_carino_period_returns(
            [portfolio_cumulative, benchmark_cumulative], label="cumulative portfolio and benchmark returns"
        )
        global_k = _decimal_carino_k(portfolio_cumulative, benchmark_cumulative)
        scale = [
            _decimal_carino_k(portfolio_return, benchmark_return) / global_k
            for portfolio_return, benchmark_return in zip(portfolio_period, benchmark_period, strict=True)
        ]
        allocation_cumulative = sum(
            (factor * effect for factor, effect in zip(scale, allocation, strict=True)), Decimal(0)
        )
        selection_cumulative = sum(
            (factor * effect for factor, effect in zip(scale, selection, strict=True)), Decimal(0)
        )
        interaction_cumulative = sum(
            (factor * effect for factor, effect in zip(scale, interaction, strict=True)), Decimal(0)
        )
        active_return = portfolio_cumulative - benchmark_cumulative

        return {
            "allocation": float(allocation_cumulative),
            "selection": float(selection_cumulative),
            "interaction": float(interaction_cumulative),
            "total": float(allocation_cumulative + selection_cumulative + interaction_cumulative),
            "portfolio_cumulative": float(portfolio_cumulative),
            "benchmark_cumulative": float(benchmark_cumulative),
            "active_return": float(active_return),
        }
