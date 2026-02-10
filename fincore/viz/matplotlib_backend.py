"""Matplotlib visualization backend.

Provides a concrete :class:`MatplotlibBackend` implementing the
:class:`~fincore.viz.base.VizBackend` protocol using matplotlib.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd


class MatplotlibBackend:
    """Visualization backend powered by matplotlib.

    All methods return the ``matplotlib.axes.Axes`` object so callers
    can further customise the plot.
    """

    def _import_plt(self):
        import matplotlib.pyplot as plt

        return plt

    def plot_returns(
        self,
        cum_returns: pd.Series,
        *,
        title: str = "Cumulative Returns",
        ax: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        plt = self._import_plt()
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 4))
        ax.plot(cum_returns.index, cum_returns.values, **kwargs)
        ax.set_title(title)
        ax.set_ylabel("Cumulative Return")
        ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
        ax.grid(True, alpha=0.3)
        return ax

    def plot_drawdown(
        self,
        drawdown: pd.Series,
        *,
        title: str = "Drawdown",
        ax: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        plt = self._import_plt()
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 3))
        ax.fill_between(drawdown.index, drawdown.values, 0, color="red", alpha=0.3, **kwargs)
        ax.plot(drawdown.index, drawdown.values, color="red", linewidth=0.8)
        ax.set_title(title)
        ax.set_ylabel("Drawdown")
        ax.grid(True, alpha=0.3)
        return ax

    def plot_rolling_sharpe(
        self,
        rolling_sharpe: pd.Series,
        *,
        title: str = "Rolling Sharpe Ratio",
        ax: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        plt = self._import_plt()
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 3))
        ax.plot(rolling_sharpe.index, rolling_sharpe.values, **kwargs)
        ax.set_title(title)
        ax.set_ylabel("Sharpe Ratio")
        ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
        ax.grid(True, alpha=0.3)
        return ax

    def plot_monthly_heatmap(
        self,
        monthly_returns: pd.DataFrame,
        *,
        title: str = "Monthly Returns (%)",
        ax: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        plt = self._import_plt()
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 5))
        import matplotlib.colors as mcolors

        norm = mcolors.TwoSlopeNorm(vmin=monthly_returns.min().min(), vcenter=0, vmax=monthly_returns.max().max())
        im = ax.imshow(monthly_returns.values, cmap="RdYlGn", norm=norm, aspect="auto")
        ax.set_xticks(range(monthly_returns.shape[1]))
        ax.set_xticklabels(monthly_returns.columns, fontsize=8)
        ax.set_yticks(range(monthly_returns.shape[0]))
        ax.set_yticklabels(monthly_returns.index, fontsize=8)
        ax.set_title(title)
        plt.colorbar(im, ax=ax, shrink=0.8)
        return ax
