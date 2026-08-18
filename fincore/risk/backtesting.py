"""Out-of-sample risk backtesting.

This module implements deterministic VaR/ES backtest statistics on aligned
forecast/realized loss series.  It is independent of ``fincore.risk.evt`` and
``fincore.risk.garch`` and of ``fincore.risk.models``, and records small-sample
status so inconclusive results are data, not silent passes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import pairwise
from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy import stats

STATUS_PASS = "pass"
STATUS_FAIL = "fail"
STATUS_INCONCLUSIVE = "inconclusive"
STATUS_EXPERIMENTAL = "experimental"

_MIN_OBSERVATIONS = 3
_MIN_EXPECTED_EXCEPTIONS = 5.0


def _align(forecast: pd.Series, realized: pd.Series) -> pd.DataFrame:
    if not isinstance(forecast.index, pd.DatetimeIndex) or not isinstance(realized.index, pd.DatetimeIndex):
        raise ValueError("forecast and realized must be indexed by DatetimeIndex")
    if forecast.index.has_duplicates or realized.index.has_duplicates:
        raise ValueError("forecast and realized must not contain duplicate timestamps")
    frame = pd.concat([forecast.rename("forecast"), realized.rename("realized")], axis=1).dropna()
    if frame.empty:
        raise ValueError("forecast and realized share no overlapping timestamps")
    return frame


def _chi2_pvalue(lr: float, df: int) -> float:
    return float(stats.chi2.sf(lr, df))


def kupiec_lr(observations: int, exceptions: int, confidence_level: float) -> float:
    """Kupiec proportion-of-failures likelihood-ratio statistic."""
    if observations <= 0:
        return 0.0
    p = 1.0 - confidence_level
    observed = exceptions / observations
    if observed in (0.0, 1.0):
        return math.inf
    term1 = (observations - exceptions) * math.log((1.0 - observed) / (1.0 - p))
    term2 = exceptions * math.log(observed / p)
    return -2.0 * (term1 + term2)


def _transition_counts(exceptions: np.ndarray) -> tuple[int, int, int, int]:
    n00 = n01 = n10 = n11 = 0
    for prev, curr in pairwise(exceptions):
        if prev == 0 and curr == 0:
            n00 += 1
        elif prev == 0 and curr == 1:
            n01 += 1
        elif prev == 1 and curr == 0:
            n10 += 1
        else:
            n11 += 1
    return n00, n01, n10, n11


def christoffersen_lr(exceptions: np.ndarray) -> float:
    """Christoffersen independence likelihood-ratio statistic.

    Uses the multinomial form so zero cell counts degrade gracefully
    (``0 * log(0) = 0``) instead of raising a math domain error.
    """
    if len(exceptions) < 2:
        return 0.0
    n00, n01, n10, n11 = _transition_counts(exceptions)
    total = n00 + n01 + n10 + n11
    if total == 0:
        return 0.0

    def xlog(x: float) -> float:
        return x * math.log(x) if x > 0.0 else 0.0

    return 2.0 * (
        xlog(n00)
        + xlog(n01)
        + xlog(n10)
        + xlog(n11)
        - xlog(n00 + n01)
        - xlog(n10 + n11)
        - xlog(n00 + n10)
        - xlog(n01 + n11)
        + xlog(total)
    )


@dataclass(frozen=True)
class RiskBacktestResult:
    """Result of an out-of-sample VaR backtest."""

    method: str
    confidence_level: float
    observations: int
    exceptions: int
    expected_exceptions: float
    aligned_index: pd.DatetimeIndex
    inputs_digest: str
    exception_rate: float
    kupiec_lr: float
    kupiec_pvalue: float
    christoffersen_lr: float
    christoffersen_pvalue: float
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    status: str = STATUS_INCONCLUSIVE


def backtest_var(
    forecast: pd.Series,
    realized: pd.Series,
    *,
    confidence_level: float = 0.99,
    significance: float = 0.05,
) -> RiskBacktestResult:
    """Backtest a VaR forecast against realized returns.

    Under the ``losses_negative`` convention, a VaR forecast is a negative
    threshold; an exception occurs when the realized return falls strictly
    below it.
    """
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be in (0, 1)")
    frame = _align(forecast, realized)
    exceptions_mask = frame["realized"] < frame["forecast"]
    n = len(frame)
    x = int(exceptions_mask.sum())
    p = 1.0 - confidence_level
    kupiec = kupiec_lr(n, x, confidence_level)
    kupiec_p = _chi2_pvalue(kupiec, 1) if math.isfinite(kupiec) else 0.0
    chris = christoffersen_lr(exceptions_mask.to_numpy(dtype=int))
    chris_p = _chi2_pvalue(chris, 1) if math.isfinite(chris) else 0.0

    if n < _MIN_OBSERVATIONS or n * p < _MIN_EXPECTED_EXCEPTIONS:
        status = STATUS_INCONCLUSIVE
    elif kupiec_p < significance or chris_p < significance:
        status = STATUS_FAIL
    else:
        status = STATUS_PASS

    return RiskBacktestResult(
        method="var",
        confidence_level=confidence_level,
        observations=n,
        exceptions=x,
        expected_exceptions=n * p,
        aligned_index=pd.DatetimeIndex(frame.index),
        inputs_digest=_sha256_frame(frame),
        exception_rate=x / n if n else 0.0,
        kupiec_lr=kupiec,
        kupiec_pvalue=kupiec_p,
        christoffersen_lr=chris,
        christoffersen_pvalue=chris_p,
        diagnostics={"significance": significance, "small_sample": n * p < _MIN_EXPECTED_EXCEPTIONS},
        status=status,
    )


def _sha256_frame(frame: pd.DataFrame) -> str:
    import hashlib

    payload = frame.to_csv(index=True, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def backtest_es(
    forecast: pd.Series,
    realized: pd.Series,
    *,
    confidence_level: float = 0.975,
    n_bootstrap: int = 1000,
    seed: int = 20260817,
) -> RiskBacktestResult:
    """Backtest an Expected Shortfall forecast.

    ES backtesting is an open problem; this first version uses a bootstrap
    calibration score (mean realised shortfall in the exception tail vs the
    forecast ES) and is reported with ``experimental`` status.
    """
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be in (0, 1)")
    frame = _align(forecast, realized)
    var_threshold = float(np.quantile(frame["realized"].to_numpy(), 1.0 - confidence_level))
    exceptions_mask = frame["realized"] < var_threshold
    tail = frame.loc[exceptions_mask, "realized"]
    realized_es = float(tail.mean()) if len(tail) else 0.0
    forecast_es = float(forecast.reindex(frame.index).mean())

    rng = np.random.default_rng(seed)
    boot_means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(tail.to_numpy(), size=len(tail), replace=True) if len(tail) else np.array([0.0])
        boot_means.append(float(sample.mean()))
    boot_std = float(np.std(boot_means)) if boot_means else 0.0
    z_score = (realized_es - forecast_es) / boot_std if boot_std > 0 else 0.0

    return RiskBacktestResult(
        method="es",
        confidence_level=confidence_level,
        observations=len(frame),
        exceptions=int(exceptions_mask.sum()),
        expected_exceptions=len(frame) * (1.0 - confidence_level),
        aligned_index=pd.DatetimeIndex(frame.index),
        inputs_digest=_sha256_frame(frame),
        exception_rate=float(exceptions_mask.mean()) if len(frame) else 0.0,
        kupiec_lr=0.0,
        kupiec_pvalue=1.0,
        christoffersen_lr=0.0,
        christoffersen_pvalue=1.0,
        diagnostics={
            "realized_es": realized_es,
            "forecast_es": forecast_es,
            "z_score": z_score,
            "n_bootstrap": n_bootstrap,
            "seed": seed,
        },
        status=STATUS_EXPERIMENTAL,
    )


__all__ = [
    "STATUS_EXPERIMENTAL",
    "STATUS_FAIL",
    "STATUS_INCONCLUSIVE",
    "STATUS_PASS",
    "RiskBacktestResult",
    "backtest_es",
    "backtest_var",
    "christoffersen_lr",
    "kupiec_lr",
]
