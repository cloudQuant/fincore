"""Migration-only location for pre-0.5 rolling-moment imports.

The canonical implementation lives in :mod:`fincore.metrics._rolling_moments`.
This module is an oracle-only bridge during the staged refactor and is removed
in the atomic breaking cutover.
"""

from fincore.metrics._rolling_moments import (
    MOMENT_NEEDS,
    RollingMoments,
    beta_from_moments,
    mean_return_from_moments,
    roll_alpha_beta_vectorized,
    roll_max_drawdown_chunked,
    sharpe_from_moments,
    sortino_from_moments,
    volatility_from_moments,
)

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
