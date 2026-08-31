"""Dependency-neutral visualization contracts shared by all backends."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    import pandas as pd


@runtime_checkable
class VizBackend(Protocol):
    """Protocol implemented by each visualization backend."""

    def plot_returns(self, cum_returns: pd.Series, **kwargs: Any) -> Any:
        """Plot cumulative returns."""

    def plot_drawdown(self, drawdown: pd.Series, **kwargs: Any) -> Any:
        """Plot drawdown underwater chart."""

    def plot_rolling_sharpe(
        self,
        sharpe: pd.Series,
        benchmark_sharpe: pd.Series | None = None,
        window: int = 252,
        **kwargs: Any,
    ) -> Any:
        """Plot rolling Sharpe ratio."""

    def plot_monthly_heatmap(self, returns: pd.Series | pd.DataFrame, **kwargs: Any) -> Any:
        """Plot monthly returns heatmap."""
