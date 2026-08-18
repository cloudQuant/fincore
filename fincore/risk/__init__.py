"""Advanced risk models for fincore.

Provides sophisticated risk measurement techniques beyond standard
volatility and VaR:
- EVT (Extreme Value Theory) models for tail risk
- GARCH models for conditional volatility
- Skew-t and other heavy-tailed distributions

Public capability states are declared in :mod:`fincore.capabilities` and
rendered into ``docs/quality/capability-inventory.md``.
"""

from __future__ import annotations

from fincore.risk.backtesting import (
    RiskBacktestResult,
    backtest_es,
    backtest_var,
)
from fincore.risk.evt import (
    evt_cvar,
    evt_var,
    extreme_risk,
    gev_fit,
    gpd_fit,
    hill_estimator,
)
from fincore.risk.garch import (
    EGARCH,
    GARCH,
    GJRGARCH,
    conditional_var,
    forecast_volatility,
)
from fincore.risk.models import (
    RiskEstimate,
    forecast_es,
    forecast_var,
)

__all__ = [
    "EGARCH",
    # GARCH classes
    "GARCH",
    "GJRGARCH",
    "RiskBacktestResult",
    "RiskEstimate",
    "backtest_es",
    "backtest_var",
    "conditional_var",
    "evt_cvar",
    "evt_var",
    "extreme_risk",
    "forecast_es",
    "forecast_var",
    "forecast_volatility",
    "gev_fit",
    "gpd_fit",
    # EVT functions
    "hill_estimator",
]
