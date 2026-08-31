"""Shared rolling moments plus vectorised rolling alpha/beta kernels.

The moments module computes first/second moments ONCE per window and
reuses them across metrics, so ``RollingEngine.compute([...])`` costs a
single pass per moment instead of one pass per metric.  It also provides
the vectorised rolling alpha/beta kernel (no per-window Python loop) and
a bounded-memory chunked rolling max drawdown kernel with numeric parity
to the legacy implementations.

Only numpy/pandas math lives here — no ``fincore`` imports beyond the
alignment contract — so the module cannot participate in import cycles
with ``fincore.metrics``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd

from fincore.runtime.time_series import align_binary_metric_inputs

__all__ = [
    "MOMENT_NEEDS",
    "RollingMoments",
    "beta_from_moments",
    "mean_return_from_moments",
    "roll_alpha_beta_vectorized",
    "roll_max_drawdown_chunked",
    "sharpe_from_moments",
    "sortino_from_moments",
    "volatility_from_moments",
]

# Which moments each engine metric consumes.  ``RollingEngine.compute``
# unions these across the requested metric set so a single-metric call
# pays for exactly the moments it used to compute inline and a
# multi-metric call shares them.
MOMENT_NEEDS: dict[str, frozenset[str]] = {
    "sharpe": frozenset({"mean", "std"}),
    "volatility": frozenset({"std"}),
    "sortino": frozenset({"mean", "downside_rms"}),
    "mean_return": frozenset({"mean"}),
    "beta": frozenset({"cov", "var"}),
}


@dataclass(frozen=True)
class RollingMoments:
    """First/second moments of one rolling window configuration.

    Computed once per engine ``compute`` call and shared by every
    moment-based metric.  ``needs`` selects which moments are computed;
    metrics only read the moments declared in :data:`MOMENT_NEEDS`.

    Attributes
    ----------
    window : int
        Rolling window size.
    mean : pd.Series | None
        Rolling mean of ``returns`` (``min_periods=window``).
    std : pd.Series | None
        Rolling sample standard deviation, ``ddof=1``.
    downside_rms : pd.Series | None
        Root of the rolling mean of squared downside deviations
        ``clip(returns, -inf, 0)`` — the unannualised denominator of the
        canonical ``downside_risk``.
    cov : pd.Series | None
        Rolling covariance of inner-aligned (returns, factor_returns).
    var : pd.Series | None
        Rolling variance of inner-aligned factor returns.
    """

    window: int
    mean: pd.Series | None = None
    std: pd.Series | None = None
    downside_rms: pd.Series | None = None
    cov: pd.Series | None = None
    var: pd.Series | None = None

    @classmethod
    def build(
        cls,
        returns: pd.Series,
        factor_returns: pd.Series | None = None,
        *,
        window: int,
        needs: frozenset[str] | None = None,
    ) -> RollingMoments:
        """Compute the requested moments once for the given window."""
        requested = needs or frozenset({"mean", "std", "downside_rms", "cov", "var"})
        mean: pd.Series | None = None
        std: pd.Series | None = None
        downside_rms: pd.Series | None = None
        cov: pd.Series | None = None
        var: pd.Series | None = None

        if "mean" in requested:
            mean = returns.rolling(window, min_periods=window).mean()
        if "std" in requested:
            std = returns.rolling(window, min_periods=window).std(ddof=1)
        if "downside_rms" in requested:
            downside = np.clip(returns.to_numpy(dtype=np.float64), -np.inf, 0.0)
            downside_rms = np.sqrt(
                pd.Series(downside, index=returns.index).pow(2.0).rolling(window, min_periods=window).mean()
            )
        if ("cov" in requested or "var" in requested) and factor_returns is not None:
            # Series inputs with inner alignment always produce Series.
            aligned_returns, aligned_factor = align_binary_metric_inputs(returns, factor_returns, alignment="inner")
            aligned_returns = cast("pd.Series", aligned_returns)
            aligned_factor = cast("pd.Series", aligned_factor)
            if "cov" in requested:
                cov = cast("pd.Series", aligned_returns.rolling(window, min_periods=window).cov(aligned_factor))
            if "var" in requested:
                var = cast("pd.Series", aligned_factor.rolling(window, min_periods=window).var())

        return cls(window=window, mean=mean, std=std, downside_rms=downside_rms, cov=cov, var=var)


def _require(series: pd.Series | None, name: str) -> pd.Series:
    if series is None:
        raise ValueError(f"moment {name!r} was not built; add it to the moment needs set")
    return series


def sharpe_from_moments(moments: RollingMoments, ann: float, sqrt_ann: float) -> pd.Series:
    """Rolling Sharpe ratio from shared moments (full windows, NaN-dropped)."""
    del ann  # sharpe only needs the square root of the annualisation factor
    rolling_mean = _require(moments.mean, "mean")
    rolling_std = _require(moments.std, "std")
    with np.errstate(divide="ignore", invalid="ignore"):
        return (rolling_mean / rolling_std * sqrt_ann).dropna()


def volatility_from_moments(moments: RollingMoments, ann: float, sqrt_ann: float) -> pd.Series:
    """Annualised rolling volatility from shared moments."""
    del ann
    rolling_std = _require(moments.std, "std")
    return (rolling_std * sqrt_ann).dropna()


def sortino_from_moments(moments: RollingMoments, ann: float, sqrt_ann: float) -> pd.Series:
    """Rolling Sortino ratio matching the canonical ``roll_sortino_ratio``.

    The canonical metric divides the annualised mean by the annualised
    downside risk, where downside risk is the ROOT-MEAN-SQUARE of returns
    clipped at zero (not the sample standard deviation of clipped
    returns); ``(mean * ann) / (rms * sqrt(ann)) == (mean / rms) * sqrt(ann)``.
    """
    del ann
    rolling_mean = _require(moments.mean, "mean")
    rolling_downside = _require(moments.downside_rms, "downside_rms")
    with np.errstate(divide="ignore", invalid="ignore"):
        return (rolling_mean / rolling_downside * sqrt_ann).dropna()


def mean_return_from_moments(moments: RollingMoments, ann: float, sqrt_ann: float) -> pd.Series:
    """Annualised rolling mean return from shared moments."""
    del sqrt_ann
    rolling_mean = _require(moments.mean, "mean")
    return (rolling_mean * ann).dropna()


def beta_from_moments(moments: RollingMoments, ann: float, sqrt_ann: float) -> pd.Series:
    """Rolling beta versus the benchmark from shared moments.

    Matches ``roll_beta``: pandas rolling covariance divided by rolling
    variance (ddof=1 both) over inner-aligned inputs, sliced to full
    windows.
    """
    del ann, sqrt_ann
    rolling_cov = _require(moments.cov, "cov")
    rolling_var = _require(moments.var, "var")
    with np.errstate(divide="ignore", invalid="ignore"):
        result = rolling_cov / rolling_var
    return result.iloc[moments.window - 1 :]


def roll_alpha_beta_vectorized(
    returns: pd.Series,
    factor_returns: pd.Series,
    window: int,
    risk_free: float = 0.0,
    ann_factor: float = 252.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised rolling alpha/beta replacing the per-window Python loop.

    Mirrors the per-window ``alpha_beta_aligned`` nanmean-based
    estimator:

    * ``beta = sum((f - fbar)(r - rbar)) / sum((f - fbar)^2)`` over the
      value pairs valid in both series (NaN pairs skipped exactly like
      the legacy kernel), with variances below ``1e-30`` treated as NaN;
    * ``alpha = (1 + mean(r_adj - beta * f_adj)) ** ann - 1``.

    Rolling sums are computed with pandas (C-speed, O(n) memory), so the
    cost is linear in ``n`` instead of ``n * window`` Python iterations.

    Parameters
    ----------
    returns, factor_returns : pd.Series
        Already-aligned series with equal length.
    window : int
        Rolling window size.
    risk_free : float, optional
        Risk-free rate subtracted from both series (default 0.0).
    ann_factor : float, optional
        Alpha annualisation factor (default 252).

    Returns
    -------
    tuple of np.ndarray
        ``(alpha, beta)``, each of length ``len(returns) - window + 1``.
    """
    r = returns.to_numpy(dtype=np.float64)
    f = factor_returns.to_numpy(dtype=np.float64)
    valid = ~(np.isnan(r) | np.isnan(f))
    r_adj = np.where(valid, r - risk_free, 0.0)
    f_adj = np.where(valid, f - risk_free, 0.0)
    index = returns.index

    sum_r = pd.Series(r_adj, index=index).rolling(window, min_periods=window).sum()
    sum_f = pd.Series(f_adj, index=index).rolling(window, min_periods=window).sum()
    sum_rf = pd.Series(r_adj * f_adj, index=index).rolling(window, min_periods=window).sum()
    sum_ff = pd.Series(f_adj * f_adj, index=index).rolling(window, min_periods=window).sum()
    count = pd.Series(valid, index=index).rolling(window, min_periods=window).sum()

    with np.errstate(divide="ignore", invalid="ignore"):
        mean_r = sum_r / count
        mean_f = sum_f / count
        cov_num = sum_rf - mean_f * sum_r
        var_num = sum_ff - sum_f * mean_f
        var_clean = var_num.where(var_num >= 1.0e-30)
        beta = (cov_num / var_clean).to_numpy(dtype=np.float64)
        mean_alpha = mean_r.to_numpy(dtype=np.float64) - beta * mean_f.to_numpy(dtype=np.float64)
        alpha = np.power(np.add(mean_alpha, 1.0), ann_factor) - 1.0

    return alpha[window - 1 :], beta[window - 1 :]


def roll_max_drawdown_chunked(values: np.ndarray, window: int, *, block_rows: int = 1024) -> np.ndarray:
    """Bounded-memory rolling max drawdown with legacy numeric parity.

    Processes windows in blocks so peak scratch memory is
    ``O(block_rows * window)`` instead of ``O(n * window)``.  Every window
    receives the identical cumprod/accumulate/nanmin operation sequence as
    the legacy vectorised implementation, so results are bit-identical.

    Parameters
    ----------
    values : np.ndarray
        Simple (non-cumulative) returns.
    window : int
        Rolling window size.
    block_rows : int, optional
        Number of output rows computed per block (default 1024).

    Returns
    -------
    np.ndarray
        Max drawdown per complete window, length ``len(values) - window + 1``.
    """
    from numpy.lib.stride_tricks import sliding_window_view

    ret_arr = np.asanyarray(values, dtype=np.float64)
    n = len(ret_arr) - window + 1
    out = np.empty(n, dtype=np.float64)

    for start in range(0, n, block_rows):
        stop = min(start + block_rows, n)
        windows = sliding_window_view(ret_arr[start : stop + window - 1], window)
        cum = np.empty((stop - start, window + 1), dtype=np.float64)
        cum[:, 0] = 1.0
        np.cumprod(1.0 + windows, axis=1, out=cum[:, 1:])
        run_max = np.maximum.accumulate(cum, axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            cum -= run_max
            cum /= run_max
        out[start:stop] = np.nanmin(cum, axis=1)

    return out
