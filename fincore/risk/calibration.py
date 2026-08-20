"""Risk model calibration and statistical tests.

Provides the Basel traffic-light reference (250 observations) and an ES
calibration score.  These are reference implementations for model validation,
not regulatory certification.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "basel_traffic_light",
    "es_calibration_score",
    "expected_exception_count",
]


def expected_exception_count(observations: int, confidence_level: float) -> float:
    """Expected number of VaR exceptions under correct coverage."""
    return observations * (1.0 - confidence_level)


def basel_traffic_light(exceptions: int, observations: int, confidence_level: float = 0.99) -> str:
    """Basel traffic-light zone for a 250-observation VaR backtest.

    For ``n = 250`` and 99% coverage the zones are: green 0–4, yellow 5–9,
    red 10+.  For other sample sizes the thresholds are the 95% and 99.99%
    cumulative-binomial quantiles, which reproduces the Basel reference at
    ``n = 250``.
    """
    from scipy import stats

    if observations <= 0:
        return "green"
    p = 1.0 - confidence_level
    green_max = int(stats.binom.ppf(0.95, observations, p))
    red_min = int(stats.binom.ppf(0.9999, observations, p))
    if exceptions >= red_min:
        return "red"
    if exceptions >= green_max:
        return "yellow"
    return "green"


def es_calibration_score(
    forecast_es: float,
    realized: np.ndarray,
    confidence_level: float,
) -> float:
    """A simple ES calibration score.

    Returns the relative difference between the forecast ES and the realized
    mean shortfall in the exception tail.  A value near 0 indicates a
    well-calibrated ES; negative values indicate the forecast overstates the
    tail loss.
    """
    alpha = 1.0 - confidence_level
    var_threshold = float(np.quantile(realized, alpha))
    tail = realized[realized <= var_threshold]
    realized_es = float(tail.mean()) if len(tail) else 0.0
    if abs(forecast_es) < 1e-15:
        return float("nan")
    return float((realized_es - forecast_es) / abs(forecast_es))
