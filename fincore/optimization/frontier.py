"""Efficient frontier computation.

Computes the mean-variance efficient frontier for a set of assets
using quadratic optimization (scipy.optimize).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import optimize as sp_opt

from fincore.optimization._utils import OptimizationError, check_feasibility, validate_result

__all__ = ["efficient_frontier"]


def efficient_frontier(
    returns: pd.DataFrame,
    n_points: int = 50,
    risk_free_rate: float = 0.0,
    short_allowed: bool = False,
    max_weight: float = 1.0,
    periods_per_year: float = 252.0,
) -> dict[str, Any]:
    """Compute the mean-variance efficient frontier.

    Parameters
    ----------
    returns : pd.DataFrame
        Asset returns (T x N). Columns = asset names.
    n_points : int, default 50
        Number of points on the frontier.
    risk_free_rate : float, default 0.0
        Annual risk-free rate (used for Sharpe calculation).
    short_allowed : bool, default False
        Whether short selling is allowed.
    max_weight : float, default 1.0
        Maximum weight per asset.
    periods_per_year : float, default 252
        Explicit annualization factor for the input return frequency.  It is
        recorded in the result rather than silently assuming daily returns.

    Returns
    -------
    dict
        - 'frontier_returns': array of annualised portfolio returns
        - 'frontier_volatilities': array of annualised portfolio volatilities
        - 'frontier_sharpe': array of Sharpe ratios
        - 'frontier_weights': (n_points x N) weight matrix
        - 'min_variance': dict with keys 'weights', 'return', 'volatility'
        - 'max_sharpe': dict with keys 'weights', 'return', 'volatility', 'sharpe'
        - 'asset_names': list of asset names
    """
    if not isinstance(returns, pd.DataFrame) or returns.empty:
        raise ValueError("returns must be a non-empty DataFrame.")

    if returns.shape[0] < 2:
        raise ValueError("At least 2 observations are required for frontier computation.")

    if returns.shape[1] < 2:
        raise ValueError("At least 2 assets required for frontier computation.")

    if n_points < 2:
        raise ValueError("n_points must be >= 2.")

    if max_weight <= 0:
        raise ValueError("max_weight must be > 0.")

    if not np.isfinite(max_weight):
        raise ValueError("max_weight must be finite.")

    if not np.isfinite(periods_per_year) or periods_per_year <= 0:
        raise ValueError("periods_per_year must be a positive finite value.")

    if not np.isfinite(risk_free_rate):
        raise ValueError("risk_free_rate must be finite.")

    if not np.isfinite(returns.values).all():
        raise ValueError("returns contains NaN or infinite values.")

    mu = returns.mean().to_numpy(dtype=float) * periods_per_year
    cov = returns.cov().to_numpy(dtype=float) * periods_per_year
    check_feasibility(cov)
    n = len(mu)
    asset_names = list(returns.columns)

    # --- weight bounds ---
    lb = -max_weight if short_allowed else 0.0
    bounds = [(lb, max_weight)] * n
    if n * max_weight < 1.0 - 1e-12:
        raise OptimizationError(
            "weight bounds are infeasible: the sum of maximum asset weights is below 1",
        )

    # --- constraints: weights sum to 1 ---
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    # --- helper: portfolio stats ---
    def _port_vol(w: np.ndarray) -> float:
        return float(np.sqrt(w @ cov @ w))

    def _port_ret(w: np.ndarray) -> float:
        return float(w @ mu)

    def _validate_weight_contract(w: np.ndarray, *, context: str, target: float | None = None) -> dict[str, float]:
        """Validate post-solver feasibility rather than trusting ``success`` alone."""
        sum_residual = abs(float(np.sum(w)) - 1.0)
        lower_residual = max(float(lb - np.min(w)), 0.0)
        upper_residual = max(float(np.max(w) - max_weight), 0.0)
        target_residual = 0.0 if target is None else abs(_port_ret(w) - target)
        tolerance = 1e-8
        if sum_residual > tolerance or lower_residual > tolerance or upper_residual > tolerance:
            raise OptimizationError(
                f"Optimization returned infeasible weights for {context}: "
                f"sum={sum_residual:.3e}, lower={lower_residual:.3e}, upper={upper_residual:.3e}"
            )
        if target is not None and target_residual > tolerance * max(1.0, abs(target)):
            raise OptimizationError(
                f"Optimization target-return residual exceeds tolerance for {context}: {target_residual:.3e}"
            )
        return {
            "sum_residual": sum_residual,
            "lower_bound_residual": lower_residual,
            "upper_bound_residual": upper_residual,
            "target_return_residual": target_residual,
        }

    def _max_return_weights() -> np.ndarray:
        """Solve the bounded linear max-return endpoint deterministically."""
        weights = np.full(n, lb, dtype=float)
        remaining = 1.0 - float(np.sum(weights))
        capacity = max_weight - lb
        for asset in np.argsort(-mu, kind="stable"):
            allocation = min(capacity, remaining)
            weights[asset] += allocation
            remaining -= allocation
            if remaining <= 1e-14:
                break
        if remaining > 1e-10:  # defensive: the algebraic feasibility test above should prevent this.
            raise OptimizationError("weight bounds cannot produce a fully invested portfolio")
        return weights

    # --- minimum-variance portfolio ---
    w0 = np.ones(n) / n
    res_mv = sp_opt.minimize(
        _port_vol,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    mv_w = validate_result(res_mv, context="min_variance")
    mv_diagnostics = _validate_weight_contract(mv_w, context="min_variance")
    mv_ret = _port_ret(mv_w)
    mv_vol = _port_vol(mv_w)

    # --- max-Sharpe portfolio ---
    def _neg_sharpe(w: np.ndarray) -> float:
        vol = _port_vol(w)
        if vol < 1e-12:
            return 1e6  # pragma: no cover -- Edge case for optimization
        return -((_port_ret(w) - risk_free_rate) / vol)

    res_ms = sp_opt.minimize(
        _neg_sharpe,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    ms_w = validate_result(res_ms, context="max_sharpe")
    ms_diagnostics = _validate_weight_contract(ms_w, context="max_sharpe")
    ms_ret = _port_ret(ms_w)
    ms_vol = _port_vol(ms_w)
    ms_sharpe = (ms_ret - risk_free_rate) / ms_vol if ms_vol > 1e-12 else 0.0

    # --- frontier points ---
    ret_min = mv_ret
    max_return_weights = _max_return_weights()
    max_return_diagnostics = _validate_weight_contract(max_return_weights, context="max_return_endpoint")
    ret_max = _port_ret(max_return_weights)
    target_rets = np.linspace(ret_min, ret_max, n_points)

    frontier_vols = np.empty(n_points)
    frontier_rets = np.empty(n_points)
    frontier_weights = np.empty((n_points, n))

    frontier_diagnostics: list[dict[str, Any]] = []
    previous_weights = mv_w
    for i, target in enumerate(target_rets):
        if np.isclose(ret_max, ret_min, rtol=0.0, atol=1e-12):
            w_valid = mv_w.copy()
            diagnostics = {**mv_diagnostics, "solver": "min_variance_degenerate"}
        elif i == 0:
            w_valid = mv_w.copy()
            diagnostics = {**mv_diagnostics, "solver": "min_variance"}
        elif i == n_points - 1:
            w_valid = max_return_weights.copy()
            diagnostics = {**max_return_diagnostics, "solver": "bounded_linear_max_return"}
        else:
            cons_i = [*constraints, {"type": "eq", "fun": lambda w, t=target: _port_ret(w) - t}]
            res = sp_opt.minimize(
                _port_vol,
                previous_weights,
                method="SLSQP",
                bounds=bounds,
                constraints=cons_i,
                options={"ftol": 1e-12, "maxiter": 1000},
            )
            w_valid = validate_result(res, context=f"frontier_point_{i}", allow_nan=False)
            diagnostics = {
                **_validate_weight_contract(w_valid, context=f"frontier_point_{i}", target=target),
                "solver": "SLSQP",
                "solver_status": int(res.status),
                "solver_message": str(res.message),
            }

        _validate_weight_contract(w_valid, context=f"frontier_point_{i}", target=target)
        frontier_weights[i] = w_valid
        frontier_rets[i] = _port_ret(w_valid)
        frontier_vols[i] = _port_vol(w_valid)
        frontier_diagnostics.append(diagnostics)
        previous_weights = w_valid

    frontier_sharpe = np.where(
        frontier_vols > 1e-12,
        (frontier_rets - risk_free_rate) / frontier_vols,
        0.0,
    )

    return {
        "frontier_returns": frontier_rets,
        "frontier_volatilities": frontier_vols,
        "frontier_sharpe": frontier_sharpe,
        "frontier_weights": frontier_weights,
        "min_variance": {
            "weights": mv_w,
            "return": mv_ret,
            "volatility": mv_vol,
        },
        "max_sharpe": {
            "weights": ms_w,
            "return": ms_ret,
            "volatility": ms_vol,
            "sharpe": ms_sharpe,
        },
        "asset_names": asset_names,
        "periods_per_year": float(periods_per_year),
        "solver_diagnostics": {
            "min_variance": mv_diagnostics,
            "max_sharpe": ms_diagnostics,
            "frontier": frontier_diagnostics,
        },
    }
