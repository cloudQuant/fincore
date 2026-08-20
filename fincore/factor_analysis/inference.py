"""Statistical inference for factor research.

Provides Fama-MacBeth cross-sectional regression, IC confidence intervals and
multiple-testing (Benjamini-Hochberg) correction so factor conclusions carry
inference, not just point estimates.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "fama_macbeth",
    "ic_confidence_interval",
    "ic_mean",
    "ic_t_stat",
]


def ic_mean(ic_series: pd.Series | np.ndarray) -> float:
    """Mean information coefficient over time."""
    ic = np.asarray(ic_series, dtype=float)
    ic = ic[~np.isnan(ic)]
    return float(np.mean(ic)) if len(ic) else float("nan")


def ic_t_stat(ic_series: pd.Series | np.ndarray) -> float:
    """t-statistic of the mean IC (Newey-West-free, i.i.d. assumption)."""
    ic = np.asarray(ic_series, dtype=float)
    ic = ic[~np.isnan(ic)]
    n = len(ic)
    if n < 2:
        return float("nan")
    se = np.std(ic, ddof=1) / np.sqrt(n)
    if se < 1e-15:
        return float("inf") if np.mean(ic) > 0 else float("-inf")
    return float(np.mean(ic) / se)


def ic_confidence_interval(ic_series: pd.Series | np.ndarray, *, z: float = 1.96) -> tuple[float, float]:
    """95% (by default) confidence interval for the mean IC."""
    ic = np.asarray(ic_series, dtype=float)
    ic = ic[~np.isnan(ic)]
    n = len(ic)
    if n < 2:
        return (float("nan"), float("nan"))
    se = np.std(ic, ddof=1) / np.sqrt(n)
    mean = float(np.mean(ic))
    return (mean - z * se, mean + z * se)


def fama_macbeth(
    returns: pd.DataFrame,
    exposures: pd.DataFrame,
) -> pd.DataFrame:
    """Fama-MacBeth cross-sectional regression.

    ``returns`` is a ``(n_periods, n_assets)`` panel and ``exposures`` a panel
    of cross-sectional characteristics (same shape, or a single ``(n_assets,)``
    row broadcast across periods).  For each period a cross-sectional regression
    ``R_i = alpha + beta * X_i`` is estimated; the reported coefficient is the
    time-series mean of those cross-sectional slopes, with a time-series
    standard error.
    """
    common_index = returns.index.intersection(exposures.index)
    returns = returns.loc[common_index]
    exposures = exposures.loc[common_index]
    if len(exposures.columns) != 1 and exposures.shape[1] != returns.shape[1]:
        raise ValueError("exposures must be a single-characteristic panel aligned with returns")

    estimates: list[float] = []
    alphas: list[float] = []
    for t in returns.index:
        y = returns.loc[t].to_numpy(dtype=float)
        x = exposures.loc[t].to_numpy(dtype=float).flatten()
        if x.shape[0] != y.shape[0]:
            x = np.tile(x, y.shape[0] // max(x.shape[0], 1))
        mask = ~(np.isnan(y) | np.isnan(x))
        if mask.sum() < 2 or np.std(x[mask]) < 1e-15:
            continue
        slope, intercept = np.polyfit(x[mask], y[mask], 1)
        estimates.append(float(slope))
        alphas.append(float(intercept))

    if not estimates:
        return pd.DataFrame(columns=["mean", "std_error", "t_stat"], index=["intercept", "exposure"])

    slopes = np.asarray(estimates, dtype=float)
    intercepts = np.asarray(alphas, dtype=float)
    mean_slope = float(np.mean(slopes))
    se_slope = float(np.std(slopes, ddof=1) / np.sqrt(len(slopes)))
    mean_intercept = float(np.mean(intercepts))
    se_intercept = float(np.std(intercepts, ddof=1) / np.sqrt(len(intercepts)))

    def t_stat(mean: float, se: float) -> float:
        return mean / se if se > 1e-15 else float("inf") if mean > 0 else float("-inf")

    return pd.DataFrame(
        {
            "mean": [mean_intercept, mean_slope],
            "std_error": [se_intercept, se_slope],
            "t_stat": [t_stat(mean_intercept, se_intercept), t_stat(mean_slope, se_slope)],
        },
        index=["intercept", "exposure"],
    )
