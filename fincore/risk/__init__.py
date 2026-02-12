"""Advanced risk models for fincore.

Provides sophisticated risk measurement techniques beyond standard
volatility and VaR:
- EVT (Extreme Value Theory) models for tail risk
- GARCH models for conditional volatility
- Skew-t and other heavy-tailed distributions
"""

from __future__ import annotations

from fincore.risk.evt import (
    hill_estimator,
    gev_fit,
    gpd_fit,
    evt_var,
    evt_cvar,
    extreme_risk,
)

from fincore.risk.garch import (
    GARCH,
    EGARCH,
    GJRGARCH,
    forecast_volatility,
    conditional_var,
)

__all__ = [
    # EVT functions
    "hill_estimator",
    "gev_fit",
    "gpd_fit",
    "evt_var",
    "evt_cvar",
    "extreme_risk",
    # GARCH classes
    "GARCH",
    "EGARCH",
    "GJRGARCH",
    "forecast_volatility",
    "conditional_var",
]
