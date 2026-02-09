"""Visualization backend protocol and registry.

Defines the :class:`VizBackend` protocol that all visualization backends
must satisfy, plus a helper :func:`get_backend` to resolve a backend by
name.
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class VizBackend(Protocol):
    """Protocol that every visualization backend must implement."""

    def plot_returns(self, cum_returns: pd.Series, **kwargs: Any) -> Any:
        """Plot cumulative returns."""
        ...

    def plot_drawdown(self, drawdown: pd.Series, **kwargs: Any) -> Any:
        """Plot drawdown underwater chart."""
        ...

    def plot_rolling_sharpe(self, rolling_sharpe: pd.Series, **kwargs: Any) -> Any:
        """Plot rolling Sharpe ratio."""
        ...

    def plot_monthly_heatmap(self, monthly_returns: pd.DataFrame, **kwargs: Any) -> Any:
        """Plot monthly returns heatmap."""
        ...


def get_backend(name: str = "matplotlib") -> VizBackend:
    """Resolve a visualization backend by name.

    Parameters
    ----------
    name : str
        Backend identifier.  Currently supported: ``'matplotlib'``,
        ``'html'``.

    Returns
    -------
    VizBackend
        An instance satisfying the :class:`VizBackend` protocol.

    Raises
    ------
    ValueError
        If the requested backend is not recognized.
    """
    name = name.lower().strip()
    if name == "matplotlib":
        from fincore.viz.matplotlib_backend import MatplotlibBackend
        return MatplotlibBackend()
    if name == "html":
        from fincore.viz.html_backend import HtmlReportBuilder
        return HtmlReportBuilder()
    raise ValueError(
        f"Unknown viz backend {name!r}. Available: 'matplotlib', 'html'"
    )
