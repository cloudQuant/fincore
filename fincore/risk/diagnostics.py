"""Auditable, strictly out-of-sample risk-forecast diagnostics.

The enhanced walk-forward VaR API deliberately supports a small, documented
method set: finite-sample calibrated historical quantiles and a Normal
distribution fitted from sample mean and sample standard deviation.  The
historical method is Hyndman-Fan type 6 (NumPy ``"weibull"``), for which an iid
continuous sample has the requested expected quantile coverage.  This avoids
the systematic excess exceptions from NumPy's default ``"linear"`` method at
the short rolling windows used by VaR backtests.  The API does not route to
legacy GARCH or EVT kernels because those models do not yet have this
production walk-forward validation contract.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from statistics import NormalDist
from typing import Any, Final, Literal, Mapping

import numpy as np
import pandas as pd

from fincore.risk.backtesting import RiskBacktestResult, backtest_var
from fincore.risk.specs import RiskModelSpec

STATUS_OK: Final[str] = "ok"
STATUS_INSUFFICIENT_DATA: Final[str] = "insufficient_data"
STATUS_UNSUPPORTED: Final[str] = "unsupported"
HISTORICAL_QUANTILE_METHOD: Final[Literal["weibull"]] = "weibull"

__all__ = [
    "HISTORICAL_QUANTILE_METHOD",
    "STATUS_INSUFFICIENT_DATA",
    "STATUS_OK",
    "STATUS_UNSUPPORTED",
    "WalkForwardVaRResult",
    "walk_forward_var",
]


@dataclass(frozen=True)
class WalkForwardVaRResult:
    """A reproducible one-step VaR forecast path and its out-of-sample check.

    ``forecast`` and ``realized`` always share the same sorted,
    duplicate-free :class:`~pandas.DatetimeIndex`.  Each forecast is fitted at
    the latest timestamp in ``refit_timestamps`` at or before it, using a
    fixed rolling window that ends *strictly before* that forecast timestamp.
    ``backtest`` is therefore calculated only from the returned out-of-sample
    pairs; it is ``None`` when the path cannot be produced.
    """

    spec: RiskModelSpec
    forecast: pd.Series
    realized: pd.Series
    refit_timestamps: pd.DatetimeIndex
    inputs_digest: str
    status: str
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    backtest: RiskBacktestResult | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.spec, RiskModelSpec):
            raise TypeError("spec must be a RiskModelSpec")
        if not isinstance(self.forecast, pd.Series) or not isinstance(self.realized, pd.Series):
            raise TypeError("forecast and realized must be pandas Series")
        if not self.forecast.index.equals(self.realized.index):
            raise ValueError("forecast and realized must have the same index")
        if not isinstance(self.forecast.index, pd.DatetimeIndex) or not isinstance(
            self.realized.index, pd.DatetimeIndex
        ):
            raise ValueError("forecast and realized must be indexed by a DatetimeIndex")
        if self.forecast.index.has_duplicates or not self.forecast.index.is_monotonic_increasing:
            raise ValueError("forecast index must be sorted and duplicate-free")
        if self.status not in (STATUS_OK, STATUS_INSUFFICIENT_DATA, STATUS_UNSUPPORTED):
            raise ValueError("status must be ok, insufficient_data, or unsupported")
        if not _is_sha256_hex_digest(self.inputs_digest):
            raise ValueError("inputs_digest must be a lowercase SHA-256 hex digest")
        if not isinstance(self.refit_timestamps, pd.DatetimeIndex):
            raise ValueError("refit_timestamps must be a DatetimeIndex")
        if self.refit_timestamps.has_duplicates or not self.refit_timestamps.is_monotonic_increasing:
            raise ValueError("refit_timestamps must be sorted and duplicate-free")

        if self.status == STATUS_OK:
            self._validate_ok_state()
        else:
            self._validate_empty_state()

    def _validate_ok_state(self) -> None:
        if self.forecast.empty or self.realized.empty:
            raise ValueError("ok result must contain a non-empty forecast path")
        if self.refit_timestamps.empty:
            raise ValueError("ok result must contain at least one refit timestamp")
        if not self.refit_timestamps.isin(self.forecast.index).all():
            raise ValueError("refit_timestamps must belong to the forecast index")
        if self.refit_timestamps[0] != self.forecast.index[0]:
            raise ValueError("the first refit timestamp must equal the first forecast timestamp")
        if self.backtest is None:
            raise ValueError("ok result must contain a backtest")
        if not isinstance(self.backtest, RiskBacktestResult):
            raise TypeError("backtest must be a RiskBacktestResult")
        if not self.backtest.aligned_index.equals(self.forecast.index):
            raise ValueError("backtest aligned_index must equal forecast index")
        if self.backtest.observations != len(self.forecast):
            raise ValueError("backtest observations must equal forecast length")
        if not _is_sha256_hex_digest(self.backtest.inputs_digest):
            raise ValueError("backtest inputs_digest must be a lowercase SHA-256 hex digest")

    def _validate_empty_state(self) -> None:
        if not self.forecast.empty or not self.realized.empty or not self.refit_timestamps.empty:
            raise ValueError(
                f"{self.status} result must have an empty forecast path, realized path, and refit timestamps"
            )
        if self.backtest is not None:
            raise ValueError(f"{self.status} result must not contain a backtest")


def walk_forward_var(returns: pd.Series, spec: RiskModelSpec) -> WalkForwardVaRResult:
    """Build an auditable, one-step, lower-tail VaR forecast path.

    The first forecast occurs at position ``spec.window``.  Its fit uses
    ``returns.iloc[:spec.window]``; at every later timestamp, the fitter sees
    only observations positioned strictly before that timestamp.  A fit is
    refreshed at the first forecast and every ``spec.refit_cadence`` forecasts
    after it.  Forecasts between refits reuse those fitted parameters.

    Parameters
    ----------
    returns
        Finite return observations with a sorted, unique
        :class:`~pandas.DatetimeIndex`.
    spec
        A :class:`RiskModelSpec`.  This function currently supports only
        ``forecast_target="var"``, ``tail="lower"``,
        ``sign_convention="losses_negative"``, and ``horizon=1``.

    Returns
    -------
    WalkForwardVaRResult
        Includes aligned out-of-sample forecasts and realized returns, refit
        timestamps, input/spec digest, fit provenance, and an ordinary
        :func:`fincore.risk.backtest_var` result.  Unsupported contracts and
        insufficient history return an empty, structured result rather than
        estimating a forecast from the evaluation window.
    """
    clean_returns = _validate_returns(returns)
    if not isinstance(spec, RiskModelSpec):
        raise TypeError("spec must be a RiskModelSpec")

    inputs_digest = _inputs_digest(clean_returns, spec)
    unsupported_reason = _unsupported_reason(spec)
    if unsupported_reason is not None:
        return _empty_result(
            clean_returns,
            spec,
            inputs_digest,
            status=STATUS_UNSUPPORTED,
            diagnostics={"reason": unsupported_reason},
        )

    if len(clean_returns) <= spec.window:
        return _empty_result(
            clean_returns,
            spec,
            inputs_digest,
            status=STATUS_INSUFFICIENT_DATA,
            diagnostics={
                "reason": "at least one realized observation after the estimation window is required",
                "n_observations": len(clean_returns),
                "window": spec.window,
            },
        )

    forecast_values: list[float] = []
    forecast_timestamps: list[pd.Timestamp] = []
    refit_timestamps: list[pd.Timestamp] = []
    fit_parameters: dict[str, Mapping[str, float | str]] = {}
    current_forecast: float | None = None

    for position in range(spec.window, len(clean_returns)):
        timestamp = pd.Timestamp(clean_returns.index[position])
        is_refit = current_forecast is None or (position - spec.window) % spec.refit_cadence == 0
        if is_refit:
            estimation_window = clean_returns.iloc[position - spec.window : position]
            current_forecast, parameters = _fit_one_step_var(estimation_window, spec)
            refit_timestamps.append(timestamp)
            fit_parameters[timestamp.isoformat()] = parameters

        assert current_forecast is not None
        forecast_values.append(current_forecast)
        forecast_timestamps.append(timestamp)

    forecast_index = pd.DatetimeIndex(forecast_timestamps)
    forecast = pd.Series(forecast_values, index=forecast_index, name="var", dtype=float)
    realized = clean_returns.reindex(forecast_index).rename("realized")
    backtest = backtest_var(forecast, realized, confidence_level=spec.confidence_level)

    return WalkForwardVaRResult(
        spec=spec,
        forecast=forecast,
        realized=realized,
        refit_timestamps=pd.DatetimeIndex(refit_timestamps),
        inputs_digest=inputs_digest,
        status=STATUS_OK,
        diagnostics={
            "method": spec.distribution,
            "quantile_method": HISTORICAL_QUANTILE_METHOD if spec.distribution == "historical" else "normal_ppf",
            "window": spec.window,
            "refit_cadence": spec.refit_cadence,
            "fit_parameters": fit_parameters,
            "forecast_count": len(forecast),
            "backtest_status": backtest.status,
        },
        backtest=backtest,
    )


def _validate_returns(returns: pd.Series) -> pd.Series:
    if not isinstance(returns, pd.Series):
        raise TypeError("returns must be a pandas Series")
    if not isinstance(returns.index, pd.DatetimeIndex):
        raise ValueError("returns must be indexed by a DatetimeIndex")
    if returns.index.has_duplicates:
        raise ValueError("returns index must not contain duplicates")
    if not returns.index.is_monotonic_increasing:
        raise ValueError("returns index must be sorted in ascending order")
    try:
        clean = returns.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("returns must contain numeric values") from exc
    if not np.isfinite(clean.to_numpy()).all():
        raise ValueError("returns must contain only finite values")
    return clean


def _unsupported_reason(spec: RiskModelSpec) -> str | None:
    if spec.forecast_target != "var":
        return "walk_forward_var currently supports forecast_target='var' only"
    if spec.tail != "lower":
        return "walk_forward_var currently supports tail='lower' only"
    if spec.sign_convention != "losses_negative":
        return "walk_forward_var currently supports sign_convention='losses_negative' only"
    if spec.horizon != 1:
        return "walk_forward_var currently supports horizon=1 only"
    return None


def _fit_one_step_var(returns: pd.Series, spec: RiskModelSpec) -> tuple[float, Mapping[str, float | str]]:
    values = returns.to_numpy(dtype=float)
    alpha = 1.0 - spec.confidence_level
    if spec.distribution == "historical":
        return float(np.quantile(values, alpha, method=HISTORICAL_QUANTILE_METHOD)), {
            "n_observations": float(len(values)),
            "quantile_method": HISTORICAL_QUANTILE_METHOD,
        }

    mean = float(np.mean(values))
    standard_deviation = float(np.std(values, ddof=1))
    value = mean if standard_deviation == 0.0 else NormalDist(mu=mean, sigma=standard_deviation).inv_cdf(alpha)
    return value, {"mean": mean, "standard_deviation": standard_deviation, "n_observations": float(len(values))}


def _empty_result(
    returns: pd.Series,
    spec: RiskModelSpec,
    inputs_digest: str,
    *,
    status: str,
    diagnostics: Mapping[str, Any],
) -> WalkForwardVaRResult:
    index = pd.DatetimeIndex(returns.index[:0])
    forecast = pd.Series(index=index, dtype=float, name="var")
    realized = pd.Series(index=index, dtype=float, name="realized")
    return WalkForwardVaRResult(
        spec=spec,
        forecast=forecast,
        realized=realized,
        refit_timestamps=index,
        inputs_digest=inputs_digest,
        status=status,
        diagnostics=diagnostics,
    )


def _inputs_digest(returns: pd.Series, spec: RiskModelSpec) -> str:
    payload = returns.to_frame(name="returns").to_csv(index=True, lineterminator="\n").encode("utf-8")
    spec_payload = json.dumps(asdict(spec), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload + b"\n" + spec_payload).hexdigest()


def _is_sha256_hex_digest(value: object) -> bool:
    return (
        isinstance(value, str) and len(value) == 64 and all("0" <= char <= "9" or "a" <= char <= "f" for char in value)
    )
