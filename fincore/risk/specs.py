"""Risk model specification.

``RiskModelSpec`` declares the forecast target, coverage, horizon, distribution,
tail, sign convention, estimation window, refit cadence and model version — the
metadata a risk forecast must carry so it can be reproduced and audited.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["RiskModelSpec"]


@dataclass(frozen=True)
class RiskModelSpec:
    """An immutable specification for a risk forecast."""

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
        if self.forecast_target not in ("var", "es", "pair"):
            raise ValueError("forecast_target must be var, es, or pair")
        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError("confidence_level must be in (0, 1)")
        if self.horizon < 1:
            raise ValueError("horizon must be at least 1")
        if self.tail not in ("lower", "upper"):
            raise ValueError("tail must be 'lower' or 'upper'")
