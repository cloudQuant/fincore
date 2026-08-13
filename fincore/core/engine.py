"""RollingEngine — batch rolling metric computation.

Computes multiple rolling metrics in a single pass where possible,
avoiding redundant rolling-window iteration.

The engine's metric inventory is the single extension registry
(:mod:`fincore.plugin.registry`): each built-in rolling metric is
registered under the ``"rolling"`` family with
:class:`~fincore.plugin.specs.Scope.BUILTIN` scope at import time, and
:attr:`RollingEngine.available_metrics` reads the family back from the
registry.

Usage::

    from fincore.core.engine import RollingEngine
    engine = RollingEngine(returns, window=60)
    results = engine.compute(['sharpe', 'volatility', 'max_drawdown'])
    # results is a dict[str, pd.Series]
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from fincore.constants import DAILY
from fincore.metrics.basic import annualization_factor as _ann_factor
from fincore.plugin.registry import register_metric, registry
from fincore.plugin.specs import ROLLING_FAMILY, ExtensionKind, Scope

if TYPE_CHECKING:
    import pandas as pd

__all__ = ["RollingEngine"]


# =============================================================================
# Built-in rolling metrics (registered in the extension registry under the
# "rolling" family; the single source of truth for available_metrics).
# =============================================================================


@register_metric("sharpe", family=ROLLING_FAMILY, scope=Scope.BUILTIN)
def _rolling_sharpe(
    returns: pd.Series,
    *,
    factor_returns: pd.Series | None = None,
    window: int,
    ann: float,
    sqrt_ann: float,
) -> pd.Series:
    rolling_mean = returns.rolling(window, min_periods=window).mean()
    rolling_std = returns.rolling(window, min_periods=window).std(ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = (rolling_mean / rolling_std) * sqrt_ann
    return result.dropna()


@register_metric("volatility", family=ROLLING_FAMILY, scope=Scope.BUILTIN)
def _rolling_volatility(
    returns: pd.Series,
    *,
    factor_returns: pd.Series | None = None,
    window: int,
    ann: float,
    sqrt_ann: float,
) -> pd.Series:
    result = returns.rolling(window, min_periods=window).std(ddof=1) * sqrt_ann
    return result.dropna()


@register_metric("max_drawdown", family=ROLLING_FAMILY, scope=Scope.BUILTIN)
def _rolling_max_drawdown(
    returns: pd.Series,
    *,
    factor_returns: pd.Series | None = None,
    window: int,
    ann: float,
    sqrt_ann: float,
) -> pd.Series:
    from fincore.metrics.rolling import roll_max_drawdown

    return roll_max_drawdown(returns, window=window)


@register_metric("beta", family=ROLLING_FAMILY, scope=Scope.BUILTIN)
def _rolling_beta(
    returns: pd.Series,
    *,
    factor_returns: pd.Series | None = None,
    window: int,
    ann: float,
    sqrt_ann: float,
) -> pd.Series:
    if factor_returns is None:
        raise ValueError("factor_returns required to compute 'beta'")
    from fincore.metrics.rolling import roll_beta

    return roll_beta(returns, factor_returns, window=window)


@register_metric("sortino", family=ROLLING_FAMILY, scope=Scope.BUILTIN)
def _rolling_sortino(
    returns: pd.Series,
    *,
    factor_returns: pd.Series | None = None,
    window: int,
    ann: float,
    sqrt_ann: float,
) -> pd.Series:
    rolling_mean = returns.rolling(window, min_periods=window).mean()
    # downside deviation: std of returns below 0
    downside = returns.copy()
    downside[downside > 0] = 0.0
    rolling_downside_std = downside.rolling(window, min_periods=window).std(ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = (rolling_mean / rolling_downside_std) * sqrt_ann
    return result.dropna()


@register_metric("mean_return", family=ROLLING_FAMILY, scope=Scope.BUILTIN)
def _rolling_mean_return(
    returns: pd.Series,
    *,
    factor_returns: pd.Series | None = None,
    window: int,
    ann: float,
    sqrt_ann: float,
) -> pd.Series:
    result = returns.rolling(window, min_periods=window).mean() * ann
    return result.dropna()


class RollingEngine:
    """Batch rolling metric computation engine.

    Parameters
    ----------
    returns : pd.Series
        Non-cumulative simple returns with a DatetimeIndex.
    factor_returns : pd.Series, optional
        Benchmark returns (required for ``beta``).
    window : int, optional
        Rolling window size.  Default 252 (approx. 1 year of daily data).
    period : str, optional
        Data frequency.  Default ``DAILY``.
    """

    def __init__(
        self,
        returns: pd.Series,
        *,
        factor_returns: pd.Series | None = None,
        window: int = 252,
        period: str = DAILY,
    ) -> None:
        self._returns = returns
        self._factor_returns = factor_returns
        self._window = window
        self._period = period
        self._ann = _ann_factor(period, None)
        self._sqrt_ann = float(np.sqrt(self._ann))

    @property
    def available_metrics(self) -> frozenset:
        """Return the set of metric names supported by this engine.

        Sourced from the ``"rolling"`` family of the single extension
        registry — the same store that user extensions register into.
        """
        return registry.metric_names(family=ROLLING_FAMILY)

    # ------------------------------------------------------------------
    # Core compute
    # ------------------------------------------------------------------

    def compute(
        self,
        metrics: list[str] | str = "all",
    ) -> dict[str, pd.Series]:
        """Compute the requested rolling metrics.

        Parameters
        ----------
        metrics : list of str or ``'all'``
            Which metrics to compute.  Pass ``'all'`` to compute every
            available metric.

        Returns
        -------
        dict[str, pd.Series]
            Mapping from metric name to rolling values.
        """
        if metrics == "all":
            metrics = sorted(self.available_metrics)

        results: dict[str, pd.Series] = {}
        for name in metrics:
            entry = registry.get(ExtensionKind.METRIC, name, family=ROLLING_FAMILY)
            if entry is None:
                raise ValueError(f"Unknown metric {name!r}. Available: {sorted(self.available_metrics)}")
            results[name] = entry.target(
                self._returns,
                factor_returns=self._factor_returns,
                window=self._window,
                ann=self._ann,
                sqrt_ann=self._sqrt_ann,
            )
        return results
