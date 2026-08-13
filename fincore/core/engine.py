"""RollingEngine — batch rolling metric computation.

Computes multiple rolling metrics in a single pass where possible,
avoiding redundant rolling-window iteration.

The engine's metric inventory is the single extension registry
(:mod:`fincore.plugin.registry`): each built-in rolling metric is
registered under the ``"rolling"`` family with
:class:`~fincore.plugin.specs.Scope.BUILTIN` scope at import time, and
:attr:`RollingEngine.available_metrics` reads the family back from the
registry.

First/second moments are computed once per window by
:mod:`fincore.core.rolling_moments` and shared across the built-in
moment-based metrics, so ``compute(['sharpe', 'volatility', 'sortino',
'mean_return', 'beta'])`` pays for each rolling pass once instead of
once per metric.  Only the moments a requested metric set actually
needs are built (:data:`fincore.core.rolling_moments.MOMENT_NEEDS`).
Plugin-registered metrics continue to be called with the standard
target signature.

Usage::

    from fincore.core.engine import RollingEngine
    engine = RollingEngine(returns, window=60)
    results = engine.compute(['sharpe', 'volatility', 'max_drawdown'])
    # results is a dict[str, pd.Series]
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np

from fincore.constants import DAILY
from fincore.core.rolling_moments import (
    MOMENT_NEEDS,
    RollingMoments,
    beta_from_moments,
    mean_return_from_moments,
    sharpe_from_moments,
    sortino_from_moments,
    volatility_from_moments,
)
from fincore.metrics.basic import annualization_factor as _ann_factor
from fincore.plugin.registry import register_metric, registry
from fincore.plugin.specs import ROLLING_FAMILY, ExtensionKind, Scope

if TYPE_CHECKING:
    import pandas as pd

__all__ = ["RollingEngine"]


# =============================================================================
# Built-in rolling metrics (registered in the extension registry under the
# "rolling" family; the single source of truth for available_metrics).  The
# moment-based targets below are thin wrappers for plugin consumption; the
# engine itself shares one moment build across the requested metric set.
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
    moments = RollingMoments.build(returns, factor_returns=factor_returns, window=window, needs=MOMENT_NEEDS["sharpe"])
    return sharpe_from_moments(moments, ann, sqrt_ann)


@register_metric("volatility", family=ROLLING_FAMILY, scope=Scope.BUILTIN)
def _rolling_volatility(
    returns: pd.Series,
    *,
    factor_returns: pd.Series | None = None,
    window: int,
    ann: float,
    sqrt_ann: float,
) -> pd.Series:
    moments = RollingMoments.build(
        returns, factor_returns=factor_returns, window=window, needs=MOMENT_NEEDS["volatility"]
    )
    return volatility_from_moments(moments, ann, sqrt_ann)


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
    moments = RollingMoments.build(returns, factor_returns=factor_returns, window=window, needs=MOMENT_NEEDS["beta"])
    return beta_from_moments(moments, ann, sqrt_ann)


@register_metric("sortino", family=ROLLING_FAMILY, scope=Scope.BUILTIN)
def _rolling_sortino(
    returns: pd.Series,
    *,
    factor_returns: pd.Series | None = None,
    window: int,
    ann: float,
    sqrt_ann: float,
) -> pd.Series:
    moments = RollingMoments.build(returns, factor_returns=factor_returns, window=window, needs=MOMENT_NEEDS["sortino"])
    return sortino_from_moments(moments, ann, sqrt_ann)


@register_metric("mean_return", family=ROLLING_FAMILY, scope=Scope.BUILTIN)
def _rolling_mean_return(
    returns: pd.Series,
    *,
    factor_returns: pd.Series | None = None,
    window: int,
    ann: float,
    sqrt_ann: float,
) -> pd.Series:
    moments = RollingMoments.build(
        returns, factor_returns=factor_returns, window=window, needs=MOMENT_NEEDS["mean_return"]
    )
    return mean_return_from_moments(moments, ann, sqrt_ann)


# Builtin moment-based metrics and their moment consumers.  ``compute``
# unions the moment needs of every requested builtin metric, builds each
# moment once, and feeds the shared moments to these consumers.
_BUILTIN_MOMENT_METRICS: dict[str, Callable[[RollingMoments, float, float], pd.Series]] = {
    "sharpe": sharpe_from_moments,
    "volatility": volatility_from_moments,
    "sortino": sortino_from_moments,
    "mean_return": mean_return_from_moments,
    "beta": beta_from_moments,
}

_BUILTIN_TARGETS: dict[str, Callable] = {
    "sharpe": _rolling_sharpe,
    "volatility": _rolling_volatility,
    "sortino": _rolling_sortino,
    "mean_return": _rolling_mean_return,
    "beta": _rolling_beta,
    "max_drawdown": _rolling_max_drawdown,
}


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

        requested_needs: frozenset[str] = frozenset()
        for name in metrics:
            if name in MOMENT_NEEDS:
                requested_needs |= MOMENT_NEEDS[name]

        moments: RollingMoments | None = None
        results: dict[str, pd.Series] = {}
        for name in metrics:
            entry = registry.get(ExtensionKind.METRIC, name, family=ROLLING_FAMILY)
            if entry is None:
                raise ValueError(f"Unknown metric {name!r}. Available: {sorted(self.available_metrics)}")
            moment_impl = _BUILTIN_MOMENT_METRICS.get(name)
            if moment_impl is not None and entry.target is _BUILTIN_TARGETS[name]:
                # Builtin moment-based metric: share one moment build
                # across the whole requested metric set.
                if moments is None:
                    moments = RollingMoments.build(
                        self._returns,
                        factor_returns=self._factor_returns,
                        window=self._window,
                        needs=requested_needs,
                    )
                results[name] = moment_impl(moments, self._ann, self._sqrt_ann)
            else:
                # Plugin-registered (or replaced) targets get the plain
                # registry contract.
                results[name] = entry.target(
                    self._returns,
                    factor_returns=self._factor_returns,
                    window=self._window,
                    ann=self._ann,
                    sqrt_ann=self._sqrt_ann,
                )
        return results
