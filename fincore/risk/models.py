"""Enhanced risk result contracts and forecast adapters.

``RiskEstimate`` is an immutable result container: it records *what* was
forecast, under which method, confidence level, horizon, sign convention and
timestamp, together with an inputs digest for reproducibility.  ``forecast_var``
and ``forecast_es`` are enhanced adapters that reuse the existing EVT/GARCH
kernels without changing their legacy signatures.

This module is additive: it does not modify ``fincore.risk.evt`` or
``fincore.risk.garch``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Mapping, cast

import numpy as np
import pandas as pd

SIGN_LOSSES_NEGATIVE = "losses_negative"
STATUS_OK = "ok"
STATUS_INSUFFICIENT_DATA = "insufficient_data"
STATUS_FAILED = "failed"

_METHODS = ("historical", "evt", "garch")


def _validate_forecast(forecast: pd.Series | None) -> None:
    if forecast is None:
        return
    if not isinstance(forecast.index, pd.DatetimeIndex):
        raise ValueError("forecast must be indexed by a DatetimeIndex")
    if forecast.index.has_duplicates:
        raise ValueError("forecast index must not contain duplicates")
    if not forecast.index.is_monotonic_increasing:
        raise ValueError("forecast index must be sorted in ascending order")


def _sha256_inputs(returns: pd.Series) -> str:
    payload = returns.to_frame(name="returns").to_csv(index=True, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class RiskEstimate:
    """An immutable risk estimate with full provenance."""

    method: str
    confidence_level: float
    horizon: int
    sign_convention: str
    estimate: float
    forecast_timestamp: pd.Timestamp
    inputs_digest: str
    forecast: pd.Series | None = None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    status: str = STATUS_OK

    def __post_init__(self) -> None:
        if self.method not in _METHODS:
            raise ValueError(f"method must be one of {_METHODS}")
        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError("confidence_level must be in (0, 1)")
        if self.horizon < 1:
            raise ValueError("horizon must be at least 1")
        _validate_forecast(self.forecast)


def _validate_returns(returns: pd.Series) -> pd.Series:
    if not isinstance(returns, pd.Series):
        raise TypeError("returns must be a pandas Series")
    if returns.index.has_duplicates:
        raise ValueError("returns index must not contain duplicates")
    if not returns.index.is_monotonic_increasing:
        raise ValueError("returns index must be sorted in ascending order")
    return returns.dropna()


def forecast_var(
    returns: pd.Series,
    *,
    method: str = "historical",
    confidence_level: float = 0.95,
    horizon: int = 1,
    **kwargs: Any,
) -> RiskEstimate:
    """Forecast Value-at-Risk under a chosen method.

    Parameters
    ----------
    returns : pd.Series
        Historical returns.
    method : str, default "historical"
        One of ``historical`` (empirical quantile), ``evt`` (extreme-value
        theory) or ``garch`` (conditional volatility).
    confidence_level : float, default 0.95
        Coverage level; the VaR is the quantile at ``1 - confidence_level``.
    horizon : int, default 1
        Forecast horizon.
    **kwargs
        Method-specific arguments forwarded to the underlying kernel.

    Returns
    -------
    RiskEstimate
        A negative VaR under the ``losses_negative`` sign convention.
    """
    clean = _validate_returns(returns)
    alpha = 1.0 - confidence_level

    if method == "historical":
        value = float(np.quantile(clean.to_numpy(), alpha))
        diagnostics: dict[str, Any] = {"n_observations": len(clean)}
    elif method == "evt":
        from fincore.risk.evt import evt_var

        value = float(evt_var(clean.to_numpy(), alpha=alpha, **kwargs))
        diagnostics = {"kernel": "evt_var", "alpha": alpha}
    elif method == "garch":
        from fincore.risk.garch import conditional_var

        result = conditional_var(clean, alpha=alpha, **kwargs)
        value = cast("float", result["var"])
        diagnostics = {"kernel": "conditional_var", "alpha": alpha}
    else:
        raise ValueError(f"unknown method: {method}")

    timestamp = clean.index[-1] if len(clean) else pd.Timestamp("NaT")
    status = STATUS_OK if len(clean) >= 2 else STATUS_INSUFFICIENT_DATA
    return RiskEstimate(
        method=method,
        confidence_level=confidence_level,
        horizon=horizon,
        sign_convention=SIGN_LOSSES_NEGATIVE,
        estimate=value,
        forecast_timestamp=timestamp,
        inputs_digest=_sha256_inputs(clean),
        diagnostics=diagnostics,
        status=status,
    )


def forecast_es(
    returns: pd.Series,
    *,
    method: str = "historical",
    confidence_level: float = 0.95,
    horizon: int = 1,
    **kwargs: Any,
) -> RiskEstimate:
    """Forecast Expected Shortfall under a chosen method.

    ES is the average loss beyond the VaR threshold.  Under the
    ``losses_negative`` convention the returned value is negative.
    """
    clean = _validate_returns(returns)
    alpha = 1.0 - confidence_level

    if method == "historical":
        var_value = float(np.quantile(clean.to_numpy(), alpha))
        tail = clean[clean <= var_value]
        value = float(tail.mean()) if len(tail) else var_value
        diagnostics: dict[str, Any] = {"n_tail_observations": len(tail)}
    elif method == "evt":
        from fincore.risk.evt import evt_cvar

        value = float(evt_cvar(clean.to_numpy(), alpha=alpha, **kwargs))
        diagnostics = {"kernel": "evt_cvar", "alpha": alpha}
    elif method == "garch":
        from fincore.risk.garch import conditional_var

        result = conditional_var(clean, alpha=alpha, **kwargs)
        value = cast("float", result["var"])
        diagnostics = {"kernel": "conditional_var", "alpha": alpha, "note": "normal-distribution ES approximation"}
    else:
        raise ValueError(f"unknown method: {method}")

    timestamp = clean.index[-1] if len(clean) else pd.Timestamp("NaT")
    status = STATUS_OK if len(clean) >= 2 else STATUS_INSUFFICIENT_DATA
    return RiskEstimate(
        method=method,
        confidence_level=confidence_level,
        horizon=horizon,
        sign_convention=SIGN_LOSSES_NEGATIVE,
        estimate=value,
        forecast_timestamp=timestamp,
        inputs_digest=_sha256_inputs(clean),
        diagnostics=diagnostics,
        status=status,
    )


__all__ = [
    "SIGN_LOSSES_NEGATIVE",
    "STATUS_FAILED",
    "STATUS_INSUFFICIENT_DATA",
    "STATUS_OK",
    "RiskEstimate",
    "forecast_es",
    "forecast_var",
]
