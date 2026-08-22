"""Independent arithmetic oracle for the enhanced factor cost ledger.

The oracle intentionally uses only NumPy arrays and never imports ``fincore``.
It encodes the disclosed simple-return accounting convention: holdings begin
at zero, turnover is one half of one-way traded gross weight, impact is paid on
each traded weight, and capacity is the tightest participation inequality.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["FactorCostLedgerReference", "factor_cost_ledger_reference"]


@dataclass(frozen=True)
class FactorCostLedgerReference:
    """Plain-array result for one independently calculated factor ledger."""

    trade_weights: np.ndarray
    participation: np.ndarray
    turnover: np.ndarray
    spread_cost: np.ndarray
    impact_cost: np.ndarray
    borrow_cost: np.ndarray
    total_cost: np.ndarray
    net_returns: np.ndarray
    capacity_by_date: np.ndarray
    maximum_capacity: float


def factor_cost_ledger_reference(
    gross_returns: np.ndarray,
    weights: np.ndarray,
    dollar_volume: np.ndarray,
    *,
    portfolio_value: float,
    half_spread_bps: float,
    impact_coefficient: float,
    impact_exponent: float,
    max_participation: float,
    borrow_rates: np.ndarray,
) -> FactorCostLedgerReference:
    """Calculate the documented cost/capacity equations with NumPy only."""

    gross = np.asarray(gross_returns, dtype=float)
    positions = np.asarray(weights, dtype=float)
    volume = np.asarray(dollar_volume, dtype=float)
    borrow = np.asarray(borrow_rates, dtype=float)
    prior = np.vstack((np.zeros((1, positions.shape[1]), dtype=float), positions[:-1]))
    trades = np.abs(positions - prior)
    participation = trades * float(portfolio_value) / volume
    turnover = 0.5 * np.sum(trades, axis=1)
    spread = np.sum(trades, axis=1) * float(half_spread_bps) / 10_000.0
    impact = np.sum(
        trades * float(impact_coefficient) * np.power(participation, float(impact_exponent)),
        axis=1,
    )
    borrow_cost = np.sum(np.maximum(-positions, 0.0) * borrow, axis=1)
    total = spread + impact + borrow_cost
    nonzero_trade = trades > 0.0
    limits = np.where(nonzero_trade, float(max_participation) * volume / trades, np.inf)
    capacity_by_date = np.min(limits, axis=1)
    return FactorCostLedgerReference(
        trade_weights=trades,
        participation=participation,
        turnover=turnover,
        spread_cost=spread,
        impact_cost=impact,
        borrow_cost=borrow_cost,
        total_cost=total,
        net_returns=gross - total,
        capacity_by_date=capacity_by_date,
        maximum_capacity=float(np.min(capacity_by_date)),
    )
