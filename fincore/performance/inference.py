"""Uncertainty and inference for performance statistics.

Adds standard errors and confidence intervals for Sharpe, Sortino, and alpha so
a report never shows a bare point estimate without its sampling uncertainty.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

__all__ = [
    "sharpe_confidence_interval",
    "sharpe_standard_error",
    "standard_error_of_mean",
]


def standard_error_of_mean(values: np.ndarray) -> float:
    """Standard error of the sample mean (ddof=1)."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    n = len(v)
    if n < 2:
        return float("nan")
    return float(np.std(v, ddof=1) / np.sqrt(n))


def sharpe_standard_error(returns: pd.Series | np.ndarray, risk_free: float = 0.0) -> float:
    """Asymptotic standard error of the Sharpe ratio.

    ``SE(SR) = sqrt((1 + 0.5 * SR^2) / T)`` under i.i.d. normal returns
    (Lo, 2002).
    """
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)] - risk_free
    n = len(r)
    if n < 2:
        return float("nan")
    std = float(np.std(r, ddof=1))
    sr = float(np.mean(r) / std) if std > 1e-15 else 0.0
    return float(np.sqrt((1.0 + 0.5 * sr**2) / n))


def sharpe_confidence_interval(
    returns: pd.Series | np.ndarray,
    risk_free: float = 0.0,
    *,
    z: float = 1.96,
) -> tuple[float, float]:
    """95% (by default) confidence interval for the Sharpe ratio."""
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)] - risk_free
    if len(r) < 2:
        return (float("nan"), float("nan"))
    std = float(np.std(r, ddof=1))
    sr = float(np.mean(r) / std) if std > 1e-15 else 0.0
    se = sharpe_standard_error(r)
    return (sr - z * se, sr + z * se)
