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
    of cross-sectional characteristics with asset labels matching
    ``returns.columns``. A single ``(n_assets,)`` exposure row is treated as a
    static cross-section and broadcast across all return dates. For a panel,
    unavailable exposure dates become missing rows and are skipped; values are
    always reindexed by *asset label*, never by input column position.

    For each usable period a cross-sectional regression ``R_i = alpha + beta
    * X_i`` is estimated. The reported coefficient is the time-series mean of
    those slopes with an i.i.d. time-series standard error. This routine does
    not claim a HAC or clustered standard error.
    """
    if not isinstance(returns, pd.DataFrame) or not isinstance(exposures, pd.DataFrame):
        raise TypeError("returns and exposures must be pandas DataFrames")
    if returns.empty or returns.shape[1] < 2:
        raise ValueError("returns must contain at least two asset columns")
    if returns.index.has_duplicates or exposures.index.has_duplicates:
        raise ValueError("returns and exposures indices must not contain duplicates")
    if returns.columns.has_duplicates or exposures.columns.has_duplicates:
        raise ValueError("returns and exposures columns must not contain duplicates")

    missing_assets = returns.columns.difference(exposures.columns)
    if len(missing_assets):
        raise ValueError(f"exposures are missing return asset columns: {list(missing_assets)!r}")

    if len(exposures) == 1:
        static = exposures.iloc[0].reindex(returns.columns)
        aligned_exposures = pd.DataFrame(
            np.tile(static.to_numpy(dtype=float), (len(returns), 1)),
            index=returns.index,
            columns=returns.columns,
        )
    else:
        aligned_exposures = exposures.reindex(index=returns.index, columns=returns.columns)

    estimates: list[float] = []
    alphas: list[float] = []
    for t in returns.index:
        y = returns.loc[t].to_numpy(dtype=float)
        x = aligned_exposures.loc[t].to_numpy(dtype=float)
        mask = np.isfinite(y) & np.isfinite(x)
        if mask.sum() < 2 or np.std(x[mask]) < 1e-15:
            continue
        design = np.column_stack((np.ones(mask.sum()), x[mask]))
        coefficients, _, rank, _ = np.linalg.lstsq(design, y[mask], rcond=None)
        if rank < 2:  # pragma: no cover - protected by the variance guard
            continue
        alphas.append(float(coefficients[0]))
        estimates.append(float(coefficients[1]))

    if not estimates:
        return pd.DataFrame(columns=["mean", "std_error", "t_stat"], index=["intercept", "exposure"])

    slopes = np.asarray(estimates, dtype=float)
    intercepts = np.asarray(alphas, dtype=float)
    mean_slope = float(np.mean(slopes))
    mean_intercept = float(np.mean(intercepts))
    se_slope = float(np.std(slopes, ddof=1) / np.sqrt(len(slopes))) if len(slopes) > 1 else float("nan")
    se_intercept = float(np.std(intercepts, ddof=1) / np.sqrt(len(intercepts))) if len(intercepts) > 1 else float("nan")

    def t_stat(mean: float, se: float) -> float:
        if not np.isfinite(se):
            return float("nan")
        if se > 1e-15:
            return mean / se
        if mean > 0.0:
            return float("inf")
        if mean < 0.0:
            return float("-inf")
        return 0.0

    return pd.DataFrame(
        {
            "mean": [mean_intercept, mean_slope],
            "std_error": [se_intercept, se_slope],
            "t_stat": [t_stat(mean_intercept, se_intercept), t_stat(mean_slope, se_slope)],
        },
        index=["intercept", "exposure"],
    )
