"""The explicit NumPy reference backend for dense numerical kernels."""

from __future__ import annotations

from typing import cast

import numpy as np

__all__ = ["NumPyBackend"]


class NumPyBackend:
    """The pandas/NumPy reference backend.

    It is intentionally small and explicit: strict compatibility surfaces use
    their frozen implementations, while enhanced dense kernels can record this
    backend's name/version as provenance and compare any optional backend
    against its canonical semantics.
    """

    name = "numpy"
    version = np.__version__

    def cum_returns(self, returns: np.ndarray) -> np.ndarray:
        return np.asarray(np.cumprod(1.0 + np.asarray(returns, dtype=float)) - 1.0, dtype=float)

    def max_drawdown(self, returns: np.ndarray) -> float | np.ndarray:
        """Return maximum drawdown using the canonical reference semantics.

        Drawdown is not the absolute drop in cumulative return.  Its public
        contract includes an initial wealth baseline, finite-input validation,
        and defined behaviour after a zero or negative wealth path.  Delegating
        to the canonical metric keeps the reference backend on that contract
        instead of maintaining a divergent dense approximation.

        A one-dimensional input returns a scalar; a two-dimensional input
        returns one drawdown per column.
        """
        from fincore.metrics.drawdown import max_drawdown as canonical_max_drawdown

        return cast("float | np.ndarray", canonical_max_drawdown(returns))

    def sharpe_ratio(self, returns: np.ndarray, periods_per_year: int = 252) -> float:
        r = np.asarray(returns, dtype=float)
        std = float(np.std(r, ddof=1))
        if std < 1e-15:
            return float("nan")
        return float(np.mean(r) / std * np.sqrt(periods_per_year))
