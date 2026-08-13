"""AnalysisContext — cached, lazy performance analytics.

Usage::

    import fincore
    ctx = fincore.analyze(returns, factor_returns=benchmark)
    print(ctx.sharpe_ratio)
    print(ctx.perf_stats())
"""

from __future__ import annotations

import json
from functools import cached_property
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from fincore.constants import DAILY
from fincore.contracts.validation import validate_context_inputs

__all__ = ["AnalysisContext", "analyze"]

_UNSET = object()


class AnalysisContext:
    """Lazy, cached container for performance analytics.

    All metrics are computed on first access and cached via
    :func:`functools.cached_property`.  Call :meth:`invalidate` to
    clear all cached values (e.g. after replacing the underlying data).

    Parameters
    ----------
    returns : pd.Series
        Non-cumulative simple returns with a DatetimeIndex.
    factor_returns : pd.Series, optional
        Benchmark / factor returns aligned to the same dates.
    positions : pd.DataFrame, optional
        Daily net position values.
    transactions : pd.DataFrame, optional
        Executed trades.
    period : str, optional
        Data frequency.  Default ``DAILY``.

    """

    def __init__(
        self,
        returns: pd.Series,
        *,
        factor_returns: pd.Series | None = None,
        positions: pd.DataFrame | None = None,
        transactions: pd.DataFrame | None = None,
        period: str = DAILY,
        normalize_tz: str | None = None,
    ) -> None:
        """Initialize the analysis context.

        Parameters
        ----------
        returns : pd.Series
            Portfolio returns.
        factor_returns : pd.Series, optional
            Benchmark or factor returns.
        positions : pd.DataFrame, optional
            Portfolio positions.
        transactions : pd.DataFrame, optional
            Executed trades.
        period : str, default DAILY
            Data frequency.
        normalize_tz : {None, "UTC"}, default None
            Explicit timezone normalization for datetime-indexed inputs. Mixed
            timezones are rejected unless UTC normalization is requested.
        """
        from fincore.validation import validate_period

        snapshot = validate_context_inputs(
            returns=returns,
            factor_returns=factor_returns,
            positions=positions,
            transactions=transactions,
            normalize_tz=normalize_tz,
        )
        self._returns = snapshot.returns
        self._factor_returns = snapshot.factor_returns
        self._positions = snapshot.positions
        self._transactions = snapshot.transactions
        self._period = validate_period(period)
        self._normalize_tz = normalize_tz

    def _metric(self, public_name: str, *args: Any, **kwargs: Any) -> Any:
        """Invoke a context registry entry on already-validated snapshot data."""

        from fincore._dispatch import invoke_prevalidated_metric

        return invoke_prevalidated_metric("context", public_name, "cached-property", *args, **kwargs)

    # ------------------------------------------------------------------
    # Extension registry access
    # ------------------------------------------------------------------

    def compute(self, name: str, *args: Any, **kwargs: Any) -> Any:
        """Compute an extension-registered metric by name on the stored returns.

        Resolves metrics registered through
        :func:`fincore.plugin.register_metric` (default family).  The metric
        receives the stored returns as its first argument, followed by any
        positional/keyword arguments given here.

        Raises
        ------
        ValueError
            If no extension metric with this name is registered.
        """
        from fincore.plugin.registry import get_metric, list_metrics

        fn = get_metric(name)
        if fn is None:
            available = sorted(list_metrics())
            raise ValueError(
                f"Unknown metric {name!r}. Registered extension metrics: {available} "
                f"(core metrics are available as cached attributes, e.g. ctx.sharpe_ratio)."
            )
        return fn(self._returns, *args, **kwargs)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Core metrics (cached_property)
    # ------------------------------------------------------------------

    @cached_property
    def annual_return(self) -> float:
        """Annualized return."""
        return float(self._metric("annual_return", self._returns, period=self._period))

    @cached_property
    def cumulative_returns(self) -> float:
        """Total cumulative return."""
        return float(self._metric("cumulative_returns", self._returns, starting_value=0))

    @cached_property
    def annual_volatility(self) -> float:
        """Annualized volatility."""
        return float(self._metric("annual_volatility", self._returns, period=self._period))

    @cached_property
    def sharpe_ratio(self) -> float:
        """Sharpe ratio (annualized)."""
        return float(self._metric("sharpe_ratio", self._returns, period=self._period))

    @cached_property
    def calmar_ratio(self) -> float:
        """Calmar ratio (annual return / max drawdown)."""
        return float(self._metric("calmar_ratio", self._returns, period=self._period))

    @cached_property
    def stability(self) -> float:
        """R-squared of linear fit to cumulative returns."""
        return float(self._metric("stability", self._returns))

    @cached_property
    def max_drawdown(self) -> float:
        """Maximum drawdown."""
        return float(self._metric("max_drawdown", self._returns))

    @cached_property
    def omega_ratio(self) -> float:
        """Omega ratio."""
        return float(self._metric("omega_ratio", self._returns))

    @cached_property
    def sortino_ratio(self) -> float:
        """Sortino ratio (annualized)."""
        return float(self._metric("sortino_ratio", self._returns, period=self._period))

    @cached_property
    def skew(self) -> float:
        """Return skewness."""
        return float(self._metric("skew", self._returns))

    @cached_property
    def kurtosis(self) -> float:
        """Return kurtosis."""
        return float(self._metric("kurtosis", self._returns))

    @cached_property
    def tail_ratio(self) -> float:
        """Tail ratio (95th percentile / 5th percentile)."""
        return float(self._metric("tail_ratio", self._returns))

    @cached_property
    def daily_value_at_risk(self) -> float:
        """Daily Value at Risk."""
        return float(self._metric("daily_value_at_risk", self._returns))

    # ------------------------------------------------------------------
    # Factor-dependent metrics
    # ------------------------------------------------------------------

    @cached_property
    def _alpha_beta_pair(self) -> tuple[float, float]:
        """Compute the shared alpha/beta kernel exactly once."""

        if self._factor_returns is None:
            return np.nan, np.nan
        from fincore._dispatch import invoke_prevalidated_projections

        result = invoke_prevalidated_projections(
            "context",
            ("alpha", "beta"),
            "cached-property",
            self._returns,
            self._factor_returns,
            period=self._period,
        )
        return float(result["alpha"]), float(result["beta"])

    @cached_property
    def alpha(self) -> float:
        """Alpha (excess return over factor returns)."""
        return self._alpha_beta_pair[0]

    @cached_property
    def beta(self) -> float:
        """Beta (sensitivity to factor returns)."""
        return self._alpha_beta_pair[1]

    @cached_property
    def information_ratio(self) -> float:
        if self._factor_returns is None:
            return np.nan
        return float(self._metric("information_ratio", self._returns, self._factor_returns))

    @cached_property
    def gross_leverage(self) -> pd.Series:
        """Gross leverage series for the stored positions snapshot."""

        if self._positions is None:
            return pd.Series(dtype=float)
        # Registry dispatch returns Any; the gross_leverage kernel yields a Series.
        return cast("pd.Series", self._metric("gross_leverage", self._positions))

    @cached_property
    def turnover(self) -> pd.Series:
        """Turnover series for the stored portfolio and transaction snapshots."""

        if self._positions is None or self._transactions is None:
            return pd.Series(dtype=float)
        # Registry dispatch returns Any; the turnover kernel yields a Series.
        return cast("pd.Series", self._metric("turnover", self._positions, self._transactions))

    # ------------------------------------------------------------------
    # Aggregate helpers
    # ------------------------------------------------------------------

    def perf_stats(self) -> pd.Series:
        """Return a :class:`pd.Series` of key performance metrics.

        This method assembles the cached sub-metrics so that repeated
        calls are essentially free after the first computation.
        """
        from collections import OrderedDict

        stats: dict[str, Any] = OrderedDict()
        stats["Annual return"] = self.annual_return
        stats["Cumulative returns"] = self.cumulative_returns
        stats["Annual volatility"] = self.annual_volatility
        stats["Sharpe ratio"] = self.sharpe_ratio
        stats["Calmar ratio"] = self.calmar_ratio
        stats["Stability"] = self.stability
        stats["Max drawdown"] = self.max_drawdown
        stats["Omega ratio"] = self.omega_ratio
        stats["Sortino ratio"] = self.sortino_ratio
        stats["Skew"] = self.skew
        stats["Kurtosis"] = self.kurtosis
        stats["Tail ratio"] = self.tail_ratio
        stats["Daily value at risk"] = self.daily_value_at_risk

        if self._factor_returns is not None:
            stats["Alpha"] = self.alpha
            stats["Beta"] = self.beta

        if self._positions is not None:
            stats["Average gross leverage"] = float(self.gross_leverage.mean())
        if self._positions is not None and self._transactions is not None:
            stats["Average turnover"] = float(self.turnover.mean())

        return pd.Series(stats)

    def to_dict(self) -> dict[str, Any]:
        """Return metrics as a plain dict (JSON-friendly values)."""
        s = self.perf_stats()
        return {str(k): (float(v) if np.isfinite(v) else None) for k, v in s.items()}

    def to_json(self, path: str | Path | None = None, **kwargs: Any) -> str:
        """Serialize metrics and optionally write the exact payload to ``path``."""

        payload = json.dumps(self.to_dict(), **kwargs)
        if path is not None:
            Path(path).write_text(payload, encoding="utf-8")
        return payload

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def plot(self, backend: str = "matplotlib", **kwargs: Any) -> Any:
        """Plot key performance charts using the specified backend.

        Parameters
        ----------
        backend : str
            Visualization backend name (``'matplotlib'`` or ``'html'``, or a
            backend registered through :func:`fincore.plugin.register_viz_backend`).

        Returns
        -------
        Depends on the backend (e.g. matplotlib Figure or HTML string).
        Custom backends that define ``render(model, **kwargs)`` receive a
        :class:`~fincore.viz.base.RenderModel` and control the returned
        artifacts.
        """
        from fincore.viz.base import get_backend

        viz = get_backend(backend)

        render = getattr(viz, "render", None)
        if callable(render):
            from fincore.viz.base import RenderModel

            model = RenderModel(
                returns=self._returns,
                factor_returns=self._factor_returns,
                period=self._period,
            )
            return render(model, **kwargs)

        from fincore._dispatch import resolve_raw_metric

        cum_returns = resolve_raw_metric("fincore.metrics.returns:cum_returns")
        cum_ret = cum_returns(self._returns, starting_value=0)
        running_max = (1 + cum_ret).cummax()  # type: ignore[union-attr]
        drawdown = (1 + cum_ret) / running_max - 1

        rendered = [
            viz.plot_returns(cum_ret, **kwargs),
            viz.plot_drawdown(drawdown, **kwargs),
        ]
        from fincore.report.artifacts import ReportArtifacts

        build = getattr(viz, "build", None)
        html = build() if backend.lower().strip() == "html" and callable(build) else None
        return ReportArtifacts(backend=backend.lower().strip(), figures=rendered, html=html)

    def to_html(self, path: str | None = None) -> str:
        """Generate a self-contained HTML performance report.

        Parameters
        ----------
        path : str, optional
            If given, write the HTML to this file path.

        Returns
        -------
        str
            The HTML report as a string.
        """
        from fincore.viz.html_backend import HtmlReportBuilder

        builder = HtmlReportBuilder()
        builder.add_title("Performance Report")
        builder.add_metric_cards(
            self.perf_stats(),
            keys=["Annual return", "Sharpe ratio", "Max drawdown", "Annual volatility", "Calmar ratio"],
        )
        builder.add_heading("Performance Statistics")
        builder.add_stats_table(self.perf_stats())
        html = builder.build()
        if path is not None:
            Path(path).write_text(html, encoding="utf-8")
        return html

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def invalidate(self) -> None:
        """Clear all cached metric values."""
        cls = type(self)
        for attr in list(self.__dict__):
            if (
                attr.startswith("_")
                and not attr.startswith("__")
                and attr
                in (
                    "_returns",
                    "_factor_returns",
                    "_positions",
                    "_transactions",
                    "_period",
                    "_normalize_tz",
                )
            ):
                continue
            if isinstance(getattr(cls, attr, None), cached_property):
                del self.__dict__[attr]

    def replace_data(
        self,
        *,
        returns: pd.Series | object = _UNSET,
        factor_returns: pd.Series | object | None = _UNSET,
        positions: pd.DataFrame | object | None = _UNSET,
        transactions: pd.DataFrame | object | None = _UNSET,
        period: str | object = _UNSET,
        normalize_tz: str | object | None = _UNSET,
    ) -> None:
        """Atomically replace snapshot inputs and invalidate every cached metric."""

        from fincore.validation import validate_period

        next_returns = self._returns if returns is _UNSET else returns
        next_factor = self._factor_returns if factor_returns is _UNSET else factor_returns
        next_positions = self._positions if positions is _UNSET else positions
        next_transactions = self._transactions if transactions is _UNSET else transactions
        next_period = self._period if period is _UNSET else cast("str", period)
        next_normalize_tz = self._normalize_tz if normalize_tz is _UNSET else cast("str | None", normalize_tz)
        snapshot = validate_context_inputs(
            returns=next_returns,
            factor_returns=next_factor,
            positions=next_positions,
            transactions=next_transactions,
            normalize_tz=next_normalize_tz,
        )
        checked_period = validate_period(next_period)
        self.invalidate()
        self._returns = snapshot.returns
        self._factor_returns = snapshot.factor_returns
        self._positions = snapshot.positions
        self._transactions = snapshot.transactions
        self._period = checked_period
        self._normalize_tz = next_normalize_tz

    # ------------------------------------------------------------------
    # repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        n = len(self._returns)
        start = self._returns.index[0].strftime("%Y-%m-%d") if n else "?"
        end = self._returns.index[-1].strftime("%Y-%m-%d") if n else "?"
        bm = "yes" if self._factor_returns is not None else "no"
        return f"AnalysisContext({start} → {end}, {n} obs, benchmark={bm})"


# ------------------------------------------------------------------
# Convenience constructor
# ------------------------------------------------------------------


def analyze(
    returns: pd.Series,
    *,
    factor_returns: pd.Series | None = None,
    positions: pd.DataFrame | None = None,
    transactions: pd.DataFrame | None = None,
    period: str = DAILY,
    normalize_tz: str | None = None,
) -> AnalysisContext:
    """Create an :class:`AnalysisContext` — the recommended entry point.

    Example::

        import fincore
        ctx = fincore.analyze(returns, factor_returns=benchmark)
        print(ctx.sharpe_ratio)
        print(ctx.perf_stats())
    """
    return AnalysisContext(
        returns,
        factor_returns=factor_returns,
        positions=positions,
        transactions=transactions,
        period=period,
        normalize_tz=normalize_tz,
    )
