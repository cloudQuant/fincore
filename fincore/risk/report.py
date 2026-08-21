"""Portable audit reports for the enhanced walk-forward VaR boundary.

The report intentionally represents only the validated one-step, lower-tail
VaR path produced by :func:`fincore.risk.walk_forward_var`.  It records every
out-of-sample forecast/realisation pair, exception, refit and input digest in
a deterministic JSON artifact.  It is a Basel-oriented reference aid, not a
regulatory approval or compliance certification.
"""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import timezone as datetime_timezone
from numbers import Integral, Real
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final, cast

import numpy as np
import pandas as pd

from fincore.risk.calibration import basel_traffic_light
from fincore.risk.diagnostics import WalkForwardVaRResult

if TYPE_CHECKING:
    from fincore.risk.backtesting import RiskBacktestResult

BASEL_REFERENCE_DISCLOSURE: Final[str] = (
    "Basel traffic-light and VaR backtesting fields are a reference implementation; "
    "they are not regulatory approval or compliance certification."
)
RISK_VALIDATION_REPORT_SCHEMA_VERSION: Final[int] = 1

__all__ = [
    "BASEL_REFERENCE_DISCLOSURE",
    "RISK_VALIDATION_REPORT_SCHEMA_VERSION",
    "RiskValidationReport",
    "build_risk_validation_report",
]


@dataclass(frozen=True)
class RiskValidationReport:
    """A JSON-serializable audit record for one walk-forward VaR result.

    ``forecast_events`` has one record for every out-of-sample timestamp.  A
    record includes the forecast, realised return, exception flag and whether
    the model was refitted at that timestamp.  ``refits`` holds the fitted
    parameters recorded by the underlying walk-forward run.  Along with the
    input and backtest digests, this is sufficient to reconstruct the
    validation evidence without exposing the in-sample return history.
    """

    status: str
    inputs_digest: str
    specification: Mapping[str, Any]
    forecast_events: tuple[Mapping[str, Any], ...]
    refits: tuple[Mapping[str, Any], ...]
    diagnostics: Mapping[str, Any]
    backtest: Mapping[str, Any] | None
    timestamp_index_name: str | int | float | bool | None = None
    timestamp_timezone: str | None = None
    disclosure: str = BASEL_REFERENCE_DISCLOSURE
    schema_version: int = RISK_VALIDATION_REPORT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Freeze all nested JSON values after validating their shape."""
        if not isinstance(self.status, str):
            raise TypeError("status must be a string")
        if not isinstance(self.inputs_digest, str):
            raise TypeError("inputs_digest must be a string")
        object.__setattr__(
            self,
            "timestamp_index_name",
            _normalize_timestamp_index_name(self.timestamp_index_name),
        )
        if self.timestamp_timezone is not None and not isinstance(self.timestamp_timezone, str):
            raise TypeError("timestamp_timezone must be a string or None")
        if not isinstance(self.disclosure, str):
            raise TypeError("disclosure must be a string")
        if isinstance(self.schema_version, bool) or not isinstance(self.schema_version, int):
            raise TypeError("schema_version must be an integer")
        object.__setattr__(self, "specification", _freeze_json_object(self.specification, context="specification"))
        object.__setattr__(
            self,
            "forecast_events",
            tuple(_freeze_json_object(event, context="forecast event") for event in self.forecast_events),
        )
        object.__setattr__(self, "refits", tuple(_freeze_json_object(refit, context="refit") for refit in self.refits))
        object.__setattr__(self, "diagnostics", _freeze_json_object(self.diagnostics, context="diagnostics"))
        object.__setattr__(
            self,
            "backtest",
            None if self.backtest is None else _freeze_json_object(self.backtest, context="backtest"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a detached JSON-compatible representation of this report."""
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "inputs_digest": self.inputs_digest,
            "timestamp_index_name": self.timestamp_index_name,
            "timestamp_timezone": self.timestamp_timezone,
            "specification": _json_object(self.specification, context="specification"),
            "forecast_events": [_json_object(event, context="forecast event") for event in self.forecast_events],
            "refits": [_json_object(refit, context="refit") for refit in self.refits],
            "diagnostics": _json_object(self.diagnostics, context="diagnostics"),
            "backtest": None if self.backtest is None else _json_object(self.backtest, context="backtest"),
            "disclosure": self.disclosure,
        }

    def to_json(self) -> str:
        """Serialize the report as deterministic, human-readable JSON."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n"

    def write_json(self, path: str | Path) -> Path:
        """Write the deterministic JSON artifact to ``path`` and return it.

        The caller owns directory creation and output-retention policy.  This
        method deliberately does not create missing parent directories.
        """
        target = Path(path)
        target.write_text(self.to_json(), encoding="utf-8")
        return target


def build_risk_validation_report(result: WalkForwardVaRResult) -> RiskValidationReport:
    """Create an auditable report from a validated walk-forward VaR result.

    Parameters
    ----------
    result
        A :class:`~fincore.risk.diagnostics.WalkForwardVaRResult` created by
        :func:`~fincore.risk.diagnostics.walk_forward_var`.  Its status and
        path invariants have already been checked by that immutable result.

    Returns
    -------
    RiskValidationReport
        A portable record of forecast timestamps, exceptions, refits, model
        parameters, inputs digest, and backtest statistics.  An
        ``"insufficient_data"`` or ``"unsupported"`` result remains a
        structured report with empty event/refit lists and no backtest.

    Notes
    -----
    The Basel traffic-light fields are a reference implementation.  This
    function does not certify a regulatory model, approve capital treatment,
    or extend validation to legacy GARCH/EVT APIs.
    """
    if not isinstance(result, WalkForwardVaRResult):
        raise TypeError("result must be a WalkForwardVaRResult")
    result = _validated_snapshot(result)
    forecast_index = result.forecast.index
    assert isinstance(forecast_index, pd.DatetimeIndex)

    fit_parameters = _fit_parameters(result)
    forecast_events = tuple(
        {
            "timestamp": timestamp.isoformat(),
            "forecast": float(result.forecast.loc[timestamp]),
            "realized": float(result.realized.loc[timestamp]),
            "exception": bool(result.realized.loc[timestamp] < result.forecast.loc[timestamp]),
            "refit": bool(timestamp in result.refit_timestamps),
        }
        for timestamp in result.forecast.index
    )
    refits = tuple(
        {
            "timestamp": timestamp.isoformat(),
            "parameters": _json_object(
                _parameters_for_refit(fit_parameters, timestamp),
                context=f"refit parameters for {timestamp.isoformat()}",
            ),
        }
        for timestamp in result.refit_timestamps
    )

    diagnostics = {
        key: value
        for key, value in _json_object(result.diagnostics, context="walk-forward diagnostics").items()
        if key != "fit_parameters"
    }
    timestamp_index_name = _normalize_timestamp_index_name(result.forecast.index.name)
    return RiskValidationReport(
        status=result.status,
        inputs_digest=result.inputs_digest,
        specification=_json_object(asdict(result.spec), context="specification"),
        forecast_events=forecast_events,
        refits=refits,
        diagnostics=diagnostics,
        backtest=None if result.backtest is None else _backtest_payload(result.backtest),
        timestamp_index_name=timestamp_index_name,
        timestamp_timezone=_timestamp_timezone(forecast_index),
    )


def _fit_parameters(result: WalkForwardVaRResult) -> Mapping[str, Any]:
    raw = result.diagnostics.get("fit_parameters", {})
    if not isinstance(raw, Mapping):
        raise ValueError("walk-forward diagnostics fit_parameters must be a mapping")
    if result.refit_timestamps.empty:
        return raw
    if not raw:
        raise ValueError("ok walk-forward result must record fit_parameters for every refit")
    return raw


def _validated_snapshot(result: WalkForwardVaRResult) -> WalkForwardVaRResult:
    """Copy mutable result members and re-run their audit invariants.

    Dataclass freezing does not make a pandas Series or a nested diagnostics
    mapping immutable.  Reconstructing the result from deep copies prevents a
    caller-mutated path from being serialized beside stale backtest evidence.
    """
    return WalkForwardVaRResult(
        spec=result.spec,
        forecast=result.forecast.copy(deep=True),
        realized=result.realized.copy(deep=True),
        refit_timestamps=result.refit_timestamps.copy(),
        inputs_digest=result.inputs_digest,
        status=result.status,
        diagnostics=copy.deepcopy(result.diagnostics),
        backtest=copy.deepcopy(result.backtest),
    )


def _parameters_for_refit(parameters: Mapping[str, Any], timestamp: pd.Timestamp) -> Mapping[str, Any]:
    key = timestamp.isoformat()
    try:
        value = parameters[key]
    except KeyError as exc:
        raise ValueError(f"missing fit_parameters for refit {key}") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"fit_parameters for refit {key} must be a mapping")
    return value


def _backtest_payload(backtest: RiskBacktestResult) -> dict[str, Any]:
    """Return the stable, serializable subset of backtest evidence."""
    return {
        "method": backtest.method,
        "confidence_level": backtest.confidence_level,
        "observations": backtest.observations,
        "exceptions": backtest.exceptions,
        "expected_exceptions": backtest.expected_exceptions,
        "inputs_digest": backtest.inputs_digest,
        "exception_rate": backtest.exception_rate,
        "kupiec_lr": backtest.kupiec_lr,
        "kupiec_pvalue": backtest.kupiec_pvalue,
        "christoffersen_lr": backtest.christoffersen_lr,
        "christoffersen_pvalue": backtest.christoffersen_pvalue,
        "diagnostics": _json_object(backtest.diagnostics, context="backtest diagnostics"),
        "status": backtest.status,
        "traffic_light": {
            "zone": basel_traffic_light(
                backtest.exceptions,
                backtest.observations,
                backtest.confidence_level,
            ),
            "observations": backtest.observations,
            "confidence_level": backtest.confidence_level,
        },
    }


def _json_object(value: Mapping[str, Any], *, context: str) -> dict[str, Any]:
    safe = _json_safe(value, context=context)
    assert isinstance(safe, dict)
    return safe


def _freeze_json_object(value: Mapping[str, Any], *, context: str) -> Mapping[str, Any]:
    return cast("Mapping[str, Any]", _freeze_json_value(_json_object(value, context=context)))


def _normalize_timestamp_index_name(value: object) -> str | int | float | bool | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return value
    if isinstance(value, np.generic):
        return _normalize_timestamp_index_name(value.item())
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        normalized = float(value)
        if math.isfinite(normalized):
            return normalized
    raise TypeError("timestamp_index_name must be a native JSON scalar")


def _timestamp_timezone(index: pd.DatetimeIndex) -> str | None:
    if index.tz is None:
        return None

    timezone = index.tz
    for attribute in ("key", "zone"):
        candidate = getattr(timezone, attribute, None)
        if isinstance(candidate, str) and _timezone_token_replays(index, candidate):
            return candidate

    filename = getattr(timezone, "_filename", None)
    candidate = _iana_token_from_zoneinfo_filename(filename)
    if candidate is not None and _timezone_token_replays(index, candidate):
        return candidate

    candidate = _fixed_offset_token(index)
    if candidate is not None and _timezone_token_replays(index, candidate):
        return candidate

    if index.empty:
        return None
    raise ValueError("timestamp timezone cannot be represented as a portable IANA or UTC-offset token")


def _iana_token_from_zoneinfo_filename(filename: object) -> str | None:
    if not isinstance(filename, str):
        return None
    normalized = filename.replace("\\", "/")
    marker = "/zoneinfo/"
    if marker not in normalized:
        return None
    token = normalized.split(marker, maxsplit=1)[1].lstrip("/")
    if not token or token.startswith(("posix/", "right/")):
        return None
    return token


def _fixed_offset_token(index: pd.DatetimeIndex) -> str | None:
    if index.empty:
        return None
    offsets = {timestamp.utcoffset() for timestamp in index}
    if len(offsets) != 1:
        return None
    offset = offsets.pop()
    if offset is None:
        return None
    return str(datetime_timezone(offset))


def _timezone_token_replays(index: pd.DatetimeIndex, token: str) -> bool:
    try:
        replayed = index.tz_convert("UTC").tz_convert(token)
    except (KeyError, TypeError, ValueError):
        return False
    return [timestamp.isoformat() for timestamp in replayed] == [timestamp.isoformat() for timestamp in index]


def _freeze_json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze_json_value(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_json_value(item) for item in value)
    return value


def _json_safe(value: Any, *, context: str) -> Any:
    """Convert known audit values to JSON primitives or fail closed."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, np.generic):
        return _json_safe(value.item(), context=context)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        normalized = float(value)
        if not math.isfinite(normalized):
            raise ValueError(f"{context} must not contain non-finite numeric values")
        return normalized
    if isinstance(value, Mapping):
        output: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{context} mapping keys must be strings")
            output[key] = _json_safe(item, context=f"{context}.{key}")
        return output
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        return [_json_safe(item, context=context) for item in value]
    raise TypeError(f"{context} contains a value that is not JSON-compatible: {type(value).__name__}")
