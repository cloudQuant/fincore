"""Versioned contracts for the auditable risk-forecast boundary.

``RiskModelSpec`` is intentionally conservative.  It names the values that
the public walk-forward VaR implementation can honestly support today rather
than accepting a broad, ambiguous set of model labels.  GARCH and EVT remain
available through their dedicated APIs; they are not silently selected by this
contract until the corresponding out-of-sample validation path exists.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal
from numbers import Real
from typing import Final

__all__ = [
    "SUPPORTED_DISTRIBUTIONS",
    "SUPPORTED_FORECAST_TARGETS",
    "SUPPORTED_MODEL_VERSIONS",
    "SUPPORTED_SIGN_CONVENTIONS",
    "SUPPORTED_TAILS",
    "RiskModelSpec",
]


# These values are part of the enhanced, versioned public contract.  The
# walk-forward VaR path currently implements only the two distributions below.
SUPPORTED_FORECAST_TARGETS: Final[tuple[str, ...]] = ("var", "es", "pair")
SUPPORTED_DISTRIBUTIONS: Final[tuple[str, ...]] = ("historical", "normal")
SUPPORTED_TAILS: Final[tuple[str, ...]] = ("lower", "upper")
SUPPORTED_SIGN_CONVENTIONS: Final[tuple[str, ...]] = ("losses_negative",)
SUPPORTED_MODEL_VERSIONS: Final[tuple[str, ...]] = ("1.0",)


@dataclass(frozen=True)
class RiskModelSpec:
    """Immutable specification for an auditable risk forecast.

    Supported values
    ----------------
    ``distribution``
        ``"historical"`` (empirical quantile) or ``"normal"`` (sample mean
        and sample standard deviation under a Normal distribution).
    ``sign_convention``
        Only ``"losses_negative"`` is supported: lower-tail VaR is expressed
        as a return threshold and an exception is a realized return below it.
    ``model_version``
        ``"1.0"`` is the currently supported enhanced contract version.
    ``confidence_level``
        A finite real number in ``(0, 1)``.  It is normalized to a built-in
        :class:`float` before validation so the immutable specification can
        always be serialized into its audit digest.  Booleans, strings, and
        non-finite values are rejected.

    ``forecast_target`` may express ``"var"``, ``"es"`` or ``"pair"`` for
    cross-component metadata.  The public walk-forward path currently returns
    only one-period, lower-tail VaR; unsupported combinations are represented
    by its structured ``"unsupported"`` result status rather than an
    in-sample substitute.
    """

    forecast_target: str = "var"
    confidence_level: float = 0.99
    horizon: int = 1
    distribution: str = "normal"
    tail: str = "lower"
    sign_convention: str = "losses_negative"
    window: int = 252
    refit_cadence: int = 1
    model_version: str = "1.0"

    def __post_init__(self) -> None:
        _require_supported("forecast_target", self.forecast_target, SUPPORTED_FORECAST_TARGETS)
        object.__setattr__(self, "confidence_level", _normalize_confidence_level(self.confidence_level))
        _require_positive_int("horizon", self.horizon, minimum=1)
        _require_supported("distribution", self.distribution, SUPPORTED_DISTRIBUTIONS)
        _require_supported("tail", self.tail, SUPPORTED_TAILS)
        _require_supported("sign_convention", self.sign_convention, SUPPORTED_SIGN_CONVENTIONS)
        _require_positive_int("window", self.window, minimum=2)
        _require_positive_int("refit_cadence", self.refit_cadence, minimum=1)
        _require_supported("model_version", self.model_version, SUPPORTED_MODEL_VERSIONS)


def _require_positive_int(name: str, value: object, *, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer at least {minimum}")


def _normalize_confidence_level(value: object) -> float:
    """Return a JSON-safe finite coverage level or raise a contract error."""
    if isinstance(value, bool) or not isinstance(value, (Real, Decimal)):
        raise ValueError("confidence_level must be a finite real number in (0, 1)")
    try:
        normalized = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError("confidence_level must be a finite real number in (0, 1)") from exc
    if not math.isfinite(normalized) or not 0.0 < normalized < 1.0:
        raise ValueError("confidence_level must be a finite real number in (0, 1)")
    return normalized


def _require_supported(name: str, value: object, supported: tuple[str, ...]) -> None:
    if not isinstance(value, str) or value not in supported:
        values = ", ".join(repr(item) for item in supported)
        raise ValueError(f"{name} must be one of: {values}")
