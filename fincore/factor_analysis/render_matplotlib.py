"""Lazy, data-only Matplotlib rendering primitives for factor analysis.

The module deliberately imports only NumPy and pandas.  Matplotlib, seaborn,
SciPy, and statsmodels are resolved at the individual rendering boundary so a
compute-only consumer of :mod:`fincore.factor_analysis` has no visualization
dependency or backend side effect.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping, Sequence
from functools import wraps
from numbers import Integral
from typing import Any, Literal, cast

import numpy as np
import pandas as pd

from fincore.exceptions import DependencyError

DECIMAL_TO_BPS = 10_000.0
_DEFAULT_THEORETICAL_DIST = object()


def _plot_dependencies() -> tuple[Any, Any, Any, Any]:
    """Load the visualization stack only for an explicit renderer call."""

    try:
        pyplot = importlib.import_module("matplotlib.pyplot")
        cm = importlib.import_module("matplotlib.cm")
        ticker = importlib.import_module("matplotlib.ticker")
        seaborn = importlib.import_module("seaborn")
    except ModuleNotFoundError as error:
        raise DependencyError(
            "Factor-analysis plotting requires the optional 'alphalens' extra. "
            "Install it with:\n    pip install fincore[alphalens]",
            dependency="matplotlib/seaborn",
        ) from error
    return pyplot, cm, ticker, seaborn


def _require_statsmodels() -> Any:
    """Resolve the Q-Q plotting implementation without an eager import."""

    try:
        return importlib.import_module("statsmodels.api")
    except ModuleNotFoundError as error:
        raise DependencyError(
            "plot_ic_qq requires the optional 'alphalens' extra. Install it with:\n    pip install fincore[alphalens]",
            dependency="statsmodels",
        ) from error


def _as_frame(value: pd.DataFrame | pd.Series, name: str) -> pd.DataFrame:
    """Normalize renderer table inputs without changing caller-owned data."""

    if isinstance(value, pd.DataFrame):
        return value.copy(deep=True)
    if isinstance(value, pd.Series):
        return value.to_frame(name if value.name is None else value.name)
    raise TypeError(f"{name} must be a pandas Series or DataFrame")


def _as_series(value: pd.Series, name: str) -> pd.Series:
    """Copy a Series renderer input while preserving index metadata."""

    if not isinstance(value, pd.Series):
        raise TypeError(f"{name} must be a pandas Series")
    return value.copy(deep=True)


def _normalize_axes(
    ax: object,
    count: int,
    pyplot: Any,
    *,
    ncols: int = 1,
    figsize: tuple[float, float],
    sharey: bool = False,
) -> tuple[np.ndarray, bool]:
    """Return enough axes for a chart while retaining explicit ownership."""

    if count < 1:
        # Do not allocate an unreachable Figure for an empty multi-panel
        # chart. ``_axes_result`` will hand the original axes back to the
        # caller (or ``None`` when no axes were supplied).
        return np.asarray((), dtype=object), False
    if ax is None:
        rows = int(np.ceil(count / ncols))
        _, created = pyplot.subplots(rows, ncols, figsize=figsize, squeeze=False, sharey=sharey)
        return np.asarray(created, dtype=object).reshape(-1), True
    if isinstance(ax, np.ndarray):
        axes = np.asarray(ax, dtype=object).reshape(-1)
    elif isinstance(ax, Sequence) and not isinstance(ax, (str, bytes, bytearray)):
        axes = np.asarray(tuple(ax), dtype=object).reshape(-1)
    else:
        axes = np.asarray((ax,), dtype=object)
    if len(axes) < count:
        raise ValueError(f"expected at least {count} axes, received {len(axes)}")
    return axes, False


def _axes_result(
    original_ax: object,
    axes: np.ndarray,
    count: int,
    *,
    auto_scalar: bool = False,
    auto_full_grid: bool = False,
) -> Any:
    """Project renderer-owned axes into the public chart's source shape."""

    if count == 0:
        return original_ax
    if original_ax is not None:
        if count == 1 and not isinstance(original_ax, (list, tuple, np.ndarray)):
            return axes[0]
        return original_ax
    if auto_scalar:
        return axes[0]
    if auto_full_grid:
        return axes
    return axes[:count]


def _set_shared_ylim(axes: Sequence[Any]) -> None:
    """Use a shared finite range when a multi-panel chart supplies one."""

    limits = [axis.get_ylim() for axis in axes]
    if not limits:
        return
    lower = min(limit[0] for limit in limits)
    upper = max(limit[1] for limit in limits)
    if np.isfinite(lower) and np.isfinite(upper) and lower < upper:
        for axis in axes:
            axis.set_ylim(lower, upper)


def _finite_abs_percentile(values: np.ndarray, percentile: float = 95.0) -> float:
    """Return a safe symmetric range for NaN-only visual data."""

    finite = np.abs(values[np.isfinite(values)])
    if not len(finite):
        return 1.0
    result = float(np.nanpercentile(finite, percentile))
    return result if result > 0 else 1.0


def _percentile_limits(values: np.ndarray, percentiles: tuple[float, float] | None) -> tuple[float, float] | None:
    """Return finite percentile limits, or leave a NaN-only chart autoscaled."""

    if percentiles is None:
        return None
    finite = values[np.isfinite(values)]
    if not len(finite):
        return None
    lower, upper = np.nanpercentile(finite, percentiles)
    return float(lower), float(upper)


def _cumulative_growth(returns: pd.Series) -> pd.Series:
    """Prepare display-only growth values from already-computed returns."""

    return returns.astype(float).fillna(0.0).add(1.0).cumprod()


def plotting_context(context: str = "notebook", font_scale: float = 1.5, rc: Mapping[str, object] | None = None) -> Any:
    """Return Alphalens-style seaborn context without mutating global rcParams."""

    _, _, _, seaborn = _plot_dependencies()
    options = dict(rc or {})
    options.setdefault("lines.linewidth", 1.5)
    return seaborn.plotting_context(context=context, font_scale=font_scale, rc=options)


def axes_style(style: str = "darkgrid", rc: Mapping[str, object] | None = None) -> Any:
    """Return the matching seaborn axes style context manager."""

    _, _, _, seaborn = _plot_dependencies()
    return seaborn.axes_style(style=style, rc=dict(rc or {}))


def customize(function: Callable[..., Any]) -> Callable[..., Any]:
    """Decorate a plot callable with a reversible Alphalens visual context."""

    @wraps(function)
    def call_with_context(*args: Any, **kwargs: Any) -> Any:
        set_context = kwargs.pop("set_context", True)
        if not set_context:
            return function(*args, **kwargs)
        _, _, _, seaborn = _plot_dependencies()
        with plotting_context(), axes_style(), seaborn.color_palette("colorblind"):
            seaborn.despine(left=True)
            return function(*args, **kwargs)

    return call_with_context


def build_returns_table(
    alpha_beta: pd.DataFrame,
    mean_ret_quantile: pd.DataFrame,
    mean_ret_spread_quantile: pd.Series | pd.DataFrame,
) -> pd.DataFrame:
    """Build the strict returns summary as renderer-ready data, never display it."""

    table = _as_frame(alpha_beta, "alpha_beta")
    quantile = _as_frame(mean_ret_quantile, "mean_ret_quantile")
    if not isinstance(mean_ret_spread_quantile, (pd.Series, pd.DataFrame)):
        raise TypeError("mean_ret_spread_quantile must be a pandas Series or DataFrame")
    table.loc["Mean Period Wise Return Top Quantile (bps)"] = quantile.iloc[-1] * DECIMAL_TO_BPS
    table.loc["Mean Period Wise Return Bottom Quantile (bps)"] = quantile.iloc[0] * DECIMAL_TO_BPS
    # ``Series.mean`` is scalar in the pinned API and intentionally broadcasts
    # over the alpha/beta period columns. A DataFrame retains source's
    # period-wise alignment instead.
    spread = (
        mean_ret_spread_quantile.mean()
        if isinstance(mean_ret_spread_quantile, pd.Series)
        else mean_ret_spread_quantile.mean(axis=0)
    )
    table.loc["Mean Period Wise Spread (bps)"] = spread * DECIMAL_TO_BPS
    return table


def build_turnover_tables(
    autocorrelation_data: Mapping[object, pd.Series],
    quantile_turnover: Mapping[object, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build turnover and rank-autocorrelation summary tables from model data."""

    turnover_table = pd.DataFrame()
    for period in sorted(cast("list[Any]", list(quantile_turnover))):
        values = quantile_turnover[period]
        if not isinstance(values, pd.DataFrame):
            raise TypeError("quantile_turnover values must be pandas DataFrames")
        for quantile in values.columns:
            turnover_table.loc[f"Quantile {quantile} Mean Turnover ", f"{period}D"] = values[quantile].mean()
    autocorrelation_table = pd.DataFrame()
    for period, raw_values in autocorrelation_data.items():
        autocorrelation_values = _as_series(raw_values, "autocorrelation_data value")
        autocorrelation_table.loc["Mean Factor Rank Autocorrelation", f"{period}D"] = autocorrelation_values.mean()
    return turnover_table, autocorrelation_table


def build_information_table(ic_data: pd.DataFrame) -> pd.DataFrame:
    """Build Information Coefficient diagnostics from an already-computed IC table."""

    data = _as_frame(ic_data, "ic_data")
    table = pd.DataFrame(index=data.columns)
    table["IC Mean"] = data.mean()
    table["IC Std."] = data.std()
    table["Risk-Adjusted IC"] = table["IC Mean"] / table["IC Std."]
    try:
        stats = importlib.import_module("scipy.stats")
    except ModuleNotFoundError as error:
        raise DependencyError(
            "Information-table statistics require scipy. Install it with:\n    pip install fincore[alphalens]",
            dependency="scipy",
        ) from error
    t_stat, p_value = stats.ttest_1samp(data, 0.0, axis=0, nan_policy="propagate")
    table["t-stat(IC)"] = t_stat
    table["p-value(IC)"] = p_value
    table["IC Skew"] = stats.skew(data, axis=0, nan_policy="propagate")
    table["IC Kurtosis"] = stats.kurtosis(data, axis=0, nan_policy="propagate")
    return table


def build_quantile_statistics_table(factor_data: pd.DataFrame) -> pd.DataFrame:
    """Build quantile statistics directly from the frozen factor-data table."""

    if not isinstance(factor_data, pd.DataFrame):
        raise TypeError("factor_data must be a pandas DataFrame")
    if not {"factor", "factor_quantile"}.issubset(factor_data.columns):
        raise ValueError("factor_data must contain 'factor' and 'factor_quantile'")
    data = factor_data.copy(deep=True)
    if "group" in data.columns:
        data = data.drop(columns="group")
    data["factor_quantile"] = data["factor_quantile"].astype(float)
    result = data.groupby("factor_quantile", observed=False, sort=True)["factor"].agg(
        ["min", "max", "mean", "std", "count"]
    )
    total = result["count"].sum()
    result["count %"] = result["count"] / total * 100.0 if total else np.nan
    return result


def plot_ic_ts(ic: pd.DataFrame, ax: object = None) -> Any:
    """Plot period-wise IC values and their one-month rolling mean."""

    data = _as_frame(ic, "ic")
    pyplot, _, _, _ = _plot_dependencies()
    axes, _ = _normalize_axes(ax, len(data.columns), pyplot, figsize=(18, max(1, len(data.columns)) * 7))
    for axis, (period, values) in zip(axes, data.items(), strict=False):
        axis.plot(values.index, values.to_numpy(), alpha=0.7, lw=0.7, color="steelblue")
        axis.plot(values.index, values.rolling(window=22).mean().to_numpy(), color="forestgreen", lw=2)
        axis.set(ylabel="IC", xlabel="", title=f"{period} Period Forward Return Information Coefficient (IC)")
        axis.axhline(0.0, linestyle="-", color="black", lw=1, alpha=0.8)
        axis.legend(["IC", "1 month moving avg"], loc="upper right")
        axis.text(
            0.05,
            0.95,
            f"Mean {values.mean():.3f} \n Std. {values.std():.3f}",
            transform=axis.transAxes,
            verticalalignment="top",
            bbox={"facecolor": "white", "alpha": 1, "pad": 5},
        )
    _set_shared_ylim(tuple(axes[: len(data.columns)]))
    return _axes_result(ax, axes, len(data.columns))


def plot_ic_hist(ic: pd.DataFrame, ax: object = None) -> Any:
    """Plot one IC histogram per forward period."""

    data = _as_frame(ic, "ic")
    pyplot, _, _, seaborn = _plot_dependencies()
    count = len(data.columns)
    axes, _ = _normalize_axes(ax, count, pyplot, ncols=3, figsize=(18, max(1, int(np.ceil(count / 3))) * 6))
    for axis, (period, values) in zip(axes, data.items(), strict=False):
        seaborn.histplot(values.fillna(0.0), stat="density", ax=axis)
        axis.set(title=f"{period} Period IC", xlabel="IC", xlim=(-1, 1))
        axis.text(
            0.05,
            0.95,
            f"Mean {values.mean():.3f} \n Std. {values.std():.3f}",
            transform=axis.transAxes,
            verticalalignment="top",
            bbox={"facecolor": "white", "alpha": 1, "pad": 5},
        )
        axis.axvline(values.mean(), color="white", linestyle="dashed", linewidth=2)
    if count < len(axes):
        axes[-1].set_visible(False)
    return _axes_result(ax, axes, count, auto_full_grid=True)


def plot_ic_qq(ic: pd.DataFrame, theoretical_dist: object = _DEFAULT_THEORETICAL_DIST, ax: object = None) -> Any:
    """Plot each IC series against a lazily loaded theoretical distribution."""

    data = _as_frame(ic, "ic")
    pyplot, _, _, _ = _plot_dependencies()
    statsmodels = _require_statsmodels()
    if theoretical_dist is _DEFAULT_THEORETICAL_DIST:
        try:
            theoretical_dist = importlib.import_module("scipy.stats").norm
        except ModuleNotFoundError as error:
            raise DependencyError(
                "plot_ic_qq requires scipy. Install it with:\n    pip install fincore[alphalens]",
                dependency="scipy",
            ) from error
    count = len(data.columns)
    axes, _ = _normalize_axes(ax, count, pyplot, ncols=3, figsize=(18, max(1, int(np.ceil(count / 3))) * 6))
    distribution = getattr(theoretical_dist, "name", None)
    distribution_name = "Normal" if distribution == "norm" else "T" if distribution == "t" else "Theoretical"
    for axis, (period, values) in zip(axes, data.items(), strict=False):
        statsmodels.qqplot(values.fillna(0.0).to_numpy(), theoretical_dist, fit=True, line="45", ax=axis)
        axis.set(
            title=f"{period} Period IC {distribution_name} Dist. Q-Q",
            ylabel="Observed Quantile",
            xlabel=f"{distribution_name} Distribution Quantile",
        )
    return _axes_result(ax, axes, count, auto_full_grid=True)


def plot_quantile_returns_bar(
    mean_ret_by_q: pd.DataFrame,
    by_group: bool = False,
    ylim_percentiles: tuple[float, float] | None = None,
    ax: object = None,
) -> Any:
    """Plot mean forward returns by quantile, optionally one panel per group."""

    data = _as_frame(mean_ret_by_q, "mean_ret_by_q")
    pyplot, _, _, _ = _plot_dependencies()
    if data.empty:
        return ax
    scaled = data * DECIMAL_TO_BPS
    limits = _percentile_limits(scaled.to_numpy(dtype=float, copy=False), ylim_percentiles)
    if by_group:
        if not isinstance(scaled.index, pd.MultiIndex) or "group" not in scaled.index.names:
            raise ValueError("by_group=True requires a 'group' index level")
        # Match the pinned groupby default: unused categorical groups do not
        # receive an empty panel (and therefore cannot fail ``.xs`` below).
        groups = tuple(scaled.groupby(level="group", observed=True, sort=True).groups)
        axes, _ = _normalize_axes(
            ax,
            len(groups),
            pyplot,
            ncols=2,
            figsize=(18, max(1, int(np.ceil(len(groups) / 2))) * 6),
            sharey=True,
        )
        for axis, group in zip(axes, groups, strict=False):
            scaled.xs(group, level="group").plot(kind="bar", title=str(group), ax=axis)
            axis.set(xlabel="", ylabel="Mean Return (bps)")
            if limits is not None:
                axis.set_ylim(limits)
        for unused in axes[len(groups) :]:
            unused.set_visible(False)
        return _axes_result(ax, axes, len(groups), auto_full_grid=True)
    axes, _ = _normalize_axes(ax, 1, pyplot, figsize=(18, 6))
    scaled.plot(kind="bar", title="Mean Period Wise Return By Factor Quantile", ax=axes[0])
    axes[0].set(xlabel="", ylabel="Mean Return (bps)")
    if limits is not None:
        axes[0].set_ylim(limits)
    return _axes_result(ax, axes, 1, auto_scalar=True)


def plot_quantile_returns_violin(
    return_by_q: pd.DataFrame,
    ylim_percentiles: tuple[float, float] | None = None,
    ax: object = None,
) -> Any:
    """Plot period-wise return distributions by factor quantile."""

    data = _as_frame(return_by_q, "return_by_q") * DECIMAL_TO_BPS
    if not isinstance(data.index, pd.MultiIndex) or "factor_quantile" not in data.index.names:
        raise ValueError("return_by_q must have a 'factor_quantile' index level")
    pyplot, _, _, seaborn = _plot_dependencies()
    if data.empty:
        return ax
    axes, _ = _normalize_axes(ax, 1, pyplot, figsize=(18, 6))
    limits = _percentile_limits(data.to_numpy(dtype=float, copy=False), ylim_percentiles)
    columns = data.columns.rename("forward_periods")
    stacked = cast("pd.Series", data.set_axis(columns, axis="columns").stack())
    long = stacked.rename("return").reset_index()
    seaborn.violinplot(
        data=long,
        x="factor_quantile",
        hue="forward_periods",
        y="return",
        orient="v",
        cut=0,
        inner="quartile",
        ax=axes[0],
    )
    axes[0].set(xlabel="", ylabel="Return (bps)", title="Period Wise Return By Factor Quantile")
    if limits is not None:
        axes[0].set_ylim(limits)
    axes[0].axhline(0.0, linestyle="-", color="black", lw=0.7, alpha=0.6)
    return _axes_result(ax, axes, 1, auto_scalar=True)


def plot_mean_quantile_returns_spread_time_series(
    mean_returns_spread: pd.Series | pd.DataFrame,
    std_err: pd.Series | pd.DataFrame | None = None,
    bandwidth: float = 1,
    ax: object = None,
) -> Any:
    """Plot top-minus-bottom quantile spreads and optional standard-error bands."""

    if isinstance(mean_returns_spread, pd.DataFrame):
        data = mean_returns_spread.copy(deep=True)
        pyplot, _, _, _ = _plot_dependencies()
        if len(data.columns) == 1 and ax is not None and not isinstance(ax, (list, tuple, np.ndarray)):
            column = data.columns[0]
            standard_error = None if std_err is None else _as_frame(std_err, "std_err")[column]
            return plot_mean_quantile_returns_spread_time_series(data[column], standard_error, bandwidth, ax=ax)
        axes, _ = _normalize_axes(ax, len(data.columns), pyplot, figsize=(18, max(1, len(data.columns)) * 6))
        for axis, (name, values) in zip(axes, data.items(), strict=False):
            standard_error = None if std_err is None else _as_frame(std_err, "std_err")[name]
            plot_mean_quantile_returns_spread_time_series(values, standard_error, bandwidth, ax=axis)
        _set_shared_ylim(tuple(axes[: len(data.columns)]))
        if ax is None:
            return list(axes[: len(data.columns)])
        return _axes_result(ax, axes, len(data.columns))
    values = _as_series(mean_returns_spread, "mean_returns_spread")
    pyplot, _, _, _ = _plot_dependencies()
    axes, _ = _normalize_axes(ax, 1, pyplot, figsize=(18, 6))
    axis = axes[0]
    period = values.name or ""
    title = f"Top Minus Bottom Quantile Mean Return ({period} Period Forward Return)"
    displayed = values * DECIMAL_TO_BPS
    axis.plot(displayed.index, displayed.to_numpy(), alpha=0.4, lw=0.7, color="forestgreen")
    axis.plot(displayed.index, displayed.rolling(window=22).mean().to_numpy(), color="orangered", alpha=0.7)
    if std_err is not None:
        error = _as_series(cast("pd.Series", std_err), "std_err") * DECIMAL_TO_BPS
        axis.fill_between(
            displayed.index, displayed - error * bandwidth, displayed + error * bandwidth, alpha=0.3, color="steelblue"
        )
    limit = _finite_abs_percentile(displayed.to_numpy(dtype=float, copy=False))
    axis.set(
        ylabel="Difference In Quantile Mean Return (bps)",
        xlabel="",
        title=title,
        ylim=(-limit, limit),
    )
    axis.axhline(0.0, linestyle="-", color="black", lw=1, alpha=0.8)
    axis.legend(["mean returns spread", "1 month moving avg"], loc="upper right")
    return _axes_result(ax, axes, 1, auto_scalar=True)


def plot_ic_by_group(ic_group: pd.DataFrame, ax: object = None) -> Any:
    """Plot group-level mean information coefficients."""

    data = _as_frame(ic_group, "ic_group")
    pyplot, _, _, _ = _plot_dependencies()
    if data.empty:
        return ax
    axes, _ = _normalize_axes(ax, 1, pyplot, figsize=(18, 6))
    data.plot(kind="bar", ax=axes[0])
    axes[0].set(title="Information Coefficient By Group", xlabel="")
    axes[0].tick_params(axis="x", rotation=45)
    return _axes_result(ax, axes, 1, auto_scalar=True)


def plot_factor_rank_auto_correlation(factor_autocorrelation: pd.Series, period: int = 1, ax: object = None) -> Any:
    """Plot factor-rank autocorrelation without re-running its kernel."""

    values = _as_series(factor_autocorrelation, "factor_autocorrelation")
    pyplot, _, _, _ = _plot_dependencies()
    axes, _ = _normalize_axes(ax, 1, pyplot, figsize=(18, 6))
    axes[0].plot(values.index, values.to_numpy())
    axes[0].set(
        title=f"{period}D Period Factor Rank Autocorrelation",
        ylabel="Autocorrelation Coefficient",
        xlabel="",
    )
    axes[0].axhline(0.0, linestyle="-", color="black", lw=1)
    axes[0].text(
        0.05,
        0.95,
        f"Mean {values.mean():.3f}",
        transform=axes[0].transAxes,
        verticalalignment="top",
        bbox={"facecolor": "white", "alpha": 1, "pad": 5},
    )
    return _axes_result(ax, axes, 1, auto_scalar=True)


def plot_top_bottom_quantile_turnover(quantile_turnover: pd.DataFrame, period: int = 1, ax: object = None) -> Any:
    """Plot turnover for the highest and lowest available quantiles."""

    data = _as_frame(quantile_turnover, "quantile_turnover")
    if not len(data.columns):
        raise ValueError("quantile_turnover must contain at least one quantile")
    pyplot, _, _, _ = _plot_dependencies()
    axes, _ = _normalize_axes(ax, 1, pyplot, figsize=(18, 6))
    minimum, maximum = data.columns.min(), data.columns.max()
    displayed = pd.DataFrame(
        {"top quantile turnover": data[maximum], "bottom quantile turnover": data[minimum]}, index=data.index
    )
    displayed.plot(ax=axes[0], alpha=0.6, lw=0.8)
    axes[0].set(
        title=f"{period}D Period Top and Bottom Quantile Turnover",
        ylabel="Proportion Of Names New To Quantile",
        xlabel="",
    )
    return _axes_result(ax, axes, 1, auto_scalar=True)


def plot_monthly_ic_heatmap(mean_monthly_ic: pd.DataFrame, ax: object = None) -> Any:
    """Plot one monthly IC heatmap per forward period."""

    data = _as_frame(mean_monthly_ic, "mean_monthly_ic")
    pyplot, cm, _, seaborn = _plot_dependencies()
    if data.empty:
        return ax
    try:
        years = [date.year for date in data.index]
        months = [date.month for date in data.index]
    except AttributeError as error:
        raise TypeError("mean_monthly_ic index values must expose year and month") from error
    count = len(data.columns)
    axes, _ = _normalize_axes(ax, count, pyplot, ncols=3, figsize=(18, max(1, int(np.ceil(count / 3))) * 6))
    for axis, (period, values) in zip(axes, data.items(), strict=False):
        monthly = pd.DataFrame({"year": years, "month": months, "value": values.to_numpy()})
        # Pinned Alphalens constructs a (year, month) index then ``unstack``s
        # it, leaving years on rows and months on columns.
        heatmap = monthly.pivot(index="year", columns="month", values="value")
        seaborn.heatmap(
            heatmap,
            annot=True,
            alpha=1.0,
            center=0.0,
            annot_kws={"size": 7},
            linewidths=0.01,
            linecolor="white",
            cmap=cm.coolwarm_r,
            cbar=False,
            ax=axis,
        )
        axis.set(ylabel="", xlabel="", title=f"Monthly Mean {period} Period IC")
    if count < len(axes):
        axes[-1].set_visible(False)
    return _axes_result(ax, axes, count, auto_full_grid=True)


def plot_cumulative_returns(
    factor_returns: pd.Series,
    period: object,
    freq: object = None,
    title: str | None = None,
    ax: object = None,
) -> Any:
    """Plot cumulative growth for already-computed factor returns."""

    del freq
    values = _cumulative_growth(_as_series(factor_returns, "factor_returns"))
    pyplot, _, _, _ = _plot_dependencies()
    axes, _ = _normalize_axes(ax, 1, pyplot, figsize=(18, 6))
    axes[0].plot(values.index, values.to_numpy(), lw=3, color="forestgreen", alpha=0.6)
    axes[0].set(
        ylabel="Cumulative Returns",
        title=f"Portfolio Cumulative Return ({period} Fwd Period)" if title is None else title,
        xlabel="",
    )
    axes[0].axhline(1.0, linestyle="-", color="black", lw=1)
    return _axes_result(ax, axes, 1, auto_scalar=True)


def plot_cumulative_returns_by_quantile(
    quantile_returns: pd.DataFrame,
    period: object,
    freq: object = None,
    ax: object = None,
) -> Any:
    """Plot cumulative return curves for each factor quantile."""

    del freq
    data = _as_frame(quantile_returns, "quantile_returns")
    if not isinstance(data.index, pd.MultiIndex) or "factor_quantile" not in data.index.names:
        raise ValueError("quantile_returns must have a 'factor_quantile' index level")
    pyplot, cm, ticker, _ = _plot_dependencies()
    if data.empty:
        return ax
    axes, _ = _normalize_axes(ax, 1, pyplot, figsize=(18, 6))
    wide = data.unstack("factor_quantile")
    cumulative = wide.apply(_cumulative_growth, axis=0).loc[:, ::-1]
    cumulative.plot(lw=2, ax=axes[0], cmap=cm.coolwarm)
    finite = cumulative.to_numpy(dtype=float, copy=False)
    lower = float(np.nanmin(finite)) if np.isfinite(finite).any() else 0.0
    upper = float(np.nanmax(finite)) if np.isfinite(finite).any() else 1.0
    if lower == upper:
        lower, upper = lower - 1.0, upper + 1.0
    axes[0].set(
        ylabel="Log Cumulative Returns",
        title=f"Cumulative Return by Quantile\n({period} Period Forward Return)",
        xlabel="",
        yscale="symlog",
        yticks=np.linspace(lower, upper, 5),
        ylim=(lower, upper),
    )
    axes[0].yaxis.set_major_formatter(ticker.ScalarFormatter())
    axes[0].axhline(1.0, linestyle="-", color="black", lw=1)
    return _axes_result(ax, axes, 1, auto_scalar=True)


def _legacy_plot_cumulative_returns_values(
    cumulative: pd.Series,
    period: object,
    title: str | None,
    ax: object,
) -> Any:
    """Render a portfolio cumulative-return curve precomputed during assembly."""

    values = _as_series(cumulative, "cumulative")
    pyplot, _, _, _ = _plot_dependencies()
    axes, _ = _normalize_axes(ax, 1, pyplot, figsize=(18, 6))
    if len(values):
        values.name = None
    values.plot(ax=axes[0], lw=3, color="forestgreen", alpha=0.6)
    axes[0].set(
        ylabel="Cumulative Returns",
        title=f"Portfolio Cumulative Return ({period} Fwd Period)" if title is None else title,
        xlabel="",
    )
    axes[0].axhline(1.0, linestyle="-", color="black", lw=1)
    return _axes_result(ax, axes, 1, auto_scalar=True)


def _legacy_plot_cumulative_returns_by_quantile_values(
    cumulative: pd.DataFrame,
    period: object,
    ax: object,
) -> Any:
    """Render precomputed quantile cumulative curves without recomputing."""

    data = _as_frame(cumulative, "cumulative")
    pyplot, cm, ticker, _ = _plot_dependencies()
    axes, _ = _normalize_axes(ax, 1, pyplot, figsize=(18, 6))
    data.plot(lw=2, ax=axes[0], cmap=cm.coolwarm)
    axes[0].legend()
    finite = data.to_numpy(dtype=float, copy=False)
    lower = float(np.nanmin(finite)) if np.isfinite(finite).any() else 0.0
    upper = float(np.nanmax(finite)) if np.isfinite(finite).any() else 1.0
    if lower == upper:
        lower, upper = lower - 1.0, upper + 1.0
    axes[0].set(
        ylabel="Log Cumulative Returns",
        title=f"Cumulative Return by Quantile\n                    ({period} Period Forward Return)",
        xlabel="",
        yscale="symlog",
        yticks=np.linspace(lower, upper, 5),
        ylim=(lower, upper),
    )
    axes[0].yaxis.set_major_formatter(ticker.ScalarFormatter())
    axes[0].axhline(1.0, linestyle="-", color="black", lw=1)
    return _axes_result(ax, axes, 1, auto_scalar=True)


def _quantile_average_series(values: pd.DataFrame, quantile: object, statistic: str) -> pd.Series | None:
    """Select the one visual series for a quantile/statistic from source-shaped data."""

    if not isinstance(values.index, pd.MultiIndex) or "factor_quantile" not in values.index.names:
        raise ValueError("avg_cumulative_returns must include a 'factor_quantile' index level")
    # The real performance kernel emits exactly the pinned two-level layout:
    # ``(factor_quantile, 'mean'/'std')`` rows and event-window offsets as
    # columns. Select that row before supporting the richer named test/table
    # layout below.
    if values.index.nlevels == 2:
        try:
            selected = cast("pd.Series | pd.DataFrame", values.loc[cast("Any", (quantile, statistic))])
        except KeyError:
            return None
        if isinstance(selected, pd.DataFrame):
            return selected.iloc[0] if len(selected.index) else None
        return cast("pd.Series", selected)
    subset = values.xs(quantile, level="factor_quantile")
    if isinstance(subset.index, pd.MultiIndex) and "statistic" in subset.index.names:
        subset = subset.xs(statistic, level="statistic")
    elif isinstance(subset, pd.DataFrame) and statistic in subset.index:
        selected = cast("pd.Series | pd.DataFrame", subset.loc[cast("Any", statistic)])
        if isinstance(selected, pd.DataFrame):
            return selected.iloc[0] if len(selected.index) else None
        return cast("pd.Series", selected)
    elif statistic != "mean":
        return None
    if isinstance(subset, pd.DataFrame):
        if not len(subset.columns):
            return None
        return subset.iloc[:, 0]
    return cast("pd.Series", subset)


def plot_quantile_average_cumulative_return(
    avg_cumulative_returns: pd.DataFrame,
    by_quantile: bool = False,
    std_bar: bool = False,
    title: str | None = None,
    ax: object = None,
) -> Any:
    """Plot average event-window returns without recomputing event kernels."""

    data = _as_frame(avg_cumulative_returns, "avg_cumulative_returns") * DECIMAL_TO_BPS
    if not isinstance(data.index, pd.MultiIndex) or "factor_quantile" not in data.index.names:
        raise ValueError("avg_cumulative_returns must have a 'factor_quantile' index level")
    pyplot, cm, _, _ = _plot_dependencies()
    if data.empty:
        return ax
    quantiles = tuple(data.groupby(level="factor_quantile", observed=True, sort=True).groups)
    quantile_level = data.index.names.index("factor_quantile")
    # The source palette spans every declared quantile category, while the
    # groupby loop deliberately renders observed groups only.  Keeping those
    # two cardinalities separate preserves the low-to-high colour mapping for
    # a categorical index with unused levels.
    palette_size = len(data.index.levels[quantile_level].unique())
    count = len(quantiles) if by_quantile else 1
    axes, _ = _normalize_axes(ax, count, pyplot, ncols=2 if by_quantile else 1, figsize=(18, max(1, count) * 6))
    palette = [cm.coolwarm(value) for value in np.linspace(0, 1, max(1, palette_size))][::-1]
    for ordinal, quantile in enumerate(quantiles):
        target = axes[ordinal] if by_quantile else axes[0]
        mean = _quantile_average_series(data, quantile, "mean")
        if mean is None:
            continue
        target.plot(mean.index, mean.to_numpy(), color=palette[ordinal], label=f"Quantile {quantile}")
        if std_bar:
            standard_deviation = _quantile_average_series(data, quantile, "std")
            if standard_deviation is not None:
                target.errorbar(
                    mean.index, mean.to_numpy(), yerr=standard_deviation.to_numpy(), fmt="none", ecolor=palette[ordinal]
                )
        if by_quantile:
            target.axvline(x=0, color="k", linestyle="--")
            target.legend()
            target.set_ylabel("Mean Return (bps)")
    if by_quantile:
        return _axes_result(ax, axes, len(quantiles), auto_full_grid=True)
    axes[0].set(
        ylabel="Mean Return (bps)",
        title="Average Cumulative Returns by Quantile" if title is None else title,
        xlabel="Periods",
    )
    axes[0].axvline(x=0, color="k", linestyle="--")
    axes[0].legend()
    return _axes_result(ax, axes, 1, auto_scalar=True)


def plot_events_distribution(events: pd.Series, num_bars: int = 50, ax: object = None) -> Any:
    """Plot event counts over time for an event-indexed Series."""

    values = _as_series(events, "events")
    if "date" not in values.index.names:
        raise ValueError("events must use an index with a 'date' level")
    if not isinstance(num_bars, Integral) or isinstance(num_bars, bool) or num_bars <= 0:
        raise ValueError("num_bars must be a positive integer")
    pyplot, _, _, _ = _plot_dependencies()
    axes, _ = _normalize_axes(ax, 1, pyplot, figsize=(18, 6))
    num_bars = int(num_bars)
    dates = pd.DatetimeIndex(values.index.get_level_values("date"))
    if len(dates):
        start, end = dates.min(), dates.max()
        counts: pd.Series
        if start == end:
            counts = pd.Series([values.count()], index=pd.DatetimeIndex([start]))
        else:
            interval = (end - start) / num_bars
            counts = cast("pd.Series", values.groupby(pd.Grouper(level="date", freq=cast("Any", interval))).count())
        # ``Series.plot(kind='bar')`` intentionally treats time buckets as
        # categorical panels.  This is the pinned strict artist contract and
        # prevents narrow intraday buckets from overlapping at a fixed
        # date-unit Matplotlib width.
        counts.plot(kind="bar", grid=False, ax=axes[0])
    axes[0].set(ylabel="Number of events", title="Distribution of events in time", xlabel="Date")
    return _axes_result(ax, axes, 1, auto_scalar=True)


__all__ = [
    "DECIMAL_TO_BPS",
    "axes_style",
    "build_information_table",
    "build_quantile_statistics_table",
    "build_returns_table",
    "build_turnover_tables",
    "customize",
    "plot_cumulative_returns",
    "plot_cumulative_returns_by_quantile",
    "plot_events_distribution",
    "plot_factor_rank_auto_correlation",
    "plot_ic_by_group",
    "plot_ic_hist",
    "plot_ic_qq",
    "plot_ic_ts",
    "plot_mean_quantile_returns_spread_time_series",
    "plot_monthly_ic_heatmap",
    "plot_quantile_average_cumulative_return",
    "plot_quantile_returns_bar",
    "plot_quantile_returns_violin",
    "plot_top_bottom_quantile_turnover",
    "plotting_context",
]
