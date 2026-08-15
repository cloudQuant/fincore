"""Strict, import-safe projection of the pinned Alphalens plotting API.

This facade defines no Matplotlib or seaborn imports.  Every optional visual
dependency is resolved only once a chart, context, or legacy display helper is
explicitly called.
"""

from __future__ import annotations

import importlib
from functools import wraps
from typing import Any, Callable, Mapping, cast

import numpy as np
import pandas as pd

from fincore.contracts.factor_analysis import ALPHALENS_FUNCTION_SPECS, FactorFunctionSpec, function_specs_for_module

_PLOTTING_NAMES = tuple(spec.public_name for spec in function_specs_for_module("plotting"))
_DEFAULT_THEORETICAL_DIST = object()


def _spec(name: str) -> FactorFunctionSpec:
    return ALPHALENS_FUNCTION_SPECS[("plotting", name)]


def _legacy_require_index_level(value: object, level: str) -> None:
    """Project pandas' pinned missing-level ``KeyError`` before enhanced validation."""

    if isinstance(value, (pd.Series, pd.DataFrame)):
        # Delegate the error construction to pandas so name-sensitive
        # ``KeyError`` text remains identical to the pinned direct access.
        value.index.get_level_values(level)


def _legacy_scaled_percentile_limits(values: np.ndarray, percentiles: tuple[float, float]) -> tuple[float, float]:
    """Calculate source-scaled limits without hiding its NumPy warnings."""

    lower = np.nanpercentile(values, percentiles[0]) * 10_000.0
    upper = np.nanpercentile(values, percentiles[1]) * 10_000.0
    return float(lower), float(upper)


def _legacy_plot_ic_ts(ic: pd.DataFrame, ax: object) -> Any:
    """Render strict IC time series with source iterable-axis grammar."""

    renderer = _renderer()
    pyplot, _, _, _ = renderer._plot_dependencies()
    data: Any = ic.copy()
    target: Any = ax
    count = len(data.columns)
    if target is None:
        _, created = pyplot.subplots(count, 1, figsize=(18, count * 7))
        target = np.asarray([created]).flatten()
    lower, upper = None, None
    for axis, (period, values) in zip(target, data.items(), strict=False):
        values.plot(alpha=0.7, ax=axis, lw=0.7, color="steelblue")
        values.rolling(window=22).mean().plot(ax=axis, color="forestgreen", lw=2, alpha=0.8)
        axis.set(ylabel="IC", xlabel="")
        axis.set_title(f"{period} Period Forward Return Information Coefficient (IC)")
        axis.axhline(0.0, linestyle="-", color="black", lw=1, alpha=0.8)
        axis.legend(["IC", "1 month moving avg"], loc="upper right")
        axis.text(
            0.05,
            0.95,
            f"Mean {values.mean():.3f} \n Std. {values.std():.3f}",
            fontsize=16,
            bbox={"facecolor": "white", "alpha": 1, "pad": 5},
            transform=axis.transAxes,
            verticalalignment="top",
        )
        current_lower, current_upper = axis.get_ylim()
        lower = current_lower if lower is None else min(lower, current_lower)
        upper = current_upper if upper is None else max(upper, current_upper)
    for axis in target:
        axis.set_ylim([lower, upper])
    return target


def _legacy_plot_ic_hist(ic: pd.DataFrame, ax: object) -> Any:
    """Render strict IC histograms with source iterable-axis grammar."""

    renderer = _renderer()
    pyplot, _, _, seaborn = renderer._plot_dependencies()
    data: Any = ic.copy()
    target: Any = ax
    count = len(data.columns)
    rows = ((count - 1) // 3) + 1
    if target is None:
        _, created = pyplot.subplots(rows, 3, figsize=(18, rows * 6))
        target = created.flatten()
    for axis, (period, values) in zip(target, data.items(), strict=False):
        seaborn.histplot(values.replace(np.nan, 0.0), stat="density", ax=axis)
        axis.set(title=f"{period} Period IC", xlabel="IC")
        axis.set_xlim([-1, 1])
        axis.text(
            0.05,
            0.95,
            f"Mean {values.mean():.3f} \n Std. {values.std():.3f}",
            fontsize=16,
            bbox={"facecolor": "white", "alpha": 1, "pad": 5},
            transform=axis.transAxes,
            verticalalignment="top",
        )
        axis.axvline(values.mean(), color="w", linestyle="dashed", linewidth=2)
    if count < len(target):
        target[-1].set_visible(False)
    return target


def _legacy_plot_ic_qq(ic: pd.DataFrame, theoretical_dist: object, ax: object) -> Any:
    """Render source Q-Q axes, preserving explicit-``None`` and ax grammar."""

    renderer = _renderer()
    pyplot, _, _, _ = renderer._plot_dependencies()
    statsmodels = renderer._require_statsmodels()
    try:
        stats = importlib.import_module("scipy.stats")
    except ModuleNotFoundError as error:
        from fincore.exceptions import DependencyError

        raise DependencyError(
            "plot_ic_qq requires scipy. Install it with:\n    pip install fincore[alphalens]",
            dependency="scipy",
        ) from error
    data: Any = ic.copy()
    target: Any = ax
    count = len(data.columns)
    rows = ((count - 1) // 3) + 1
    if target is None:
        _, created = pyplot.subplots(rows, 3, figsize=(18, rows * 6))
        target = created.flatten()
    if theoretical_dist is _DEFAULT_THEORETICAL_DIST:
        theoretical_dist = stats.norm
    if isinstance(theoretical_dist, stats.norm.__class__):
        distribution_name = "Normal"
    elif isinstance(theoretical_dist, stats.t.__class__):
        distribution_name = "T"
    else:
        distribution_name = "Theoretical"
    for axis, (period, values) in zip(target, data.items(), strict=False):
        statsmodels.qqplot(values.replace(np.nan, 0.0).values, theoretical_dist, fit=True, line="45", ax=axis)
        axis.set(
            title=f"{period} Period IC {distribution_name} Dist. Q-Q",
            ylabel="Observed Quantile",
            xlabel=f"{distribution_name} Distribution Quantile",
        )
    return target


def _legacy_plot_quantile_returns_violin(
    return_by_q: pd.DataFrame,
    ylim_percentiles: tuple[float, float] | None,
    ax: object,
) -> Any:
    """Render the strict violin path with the pinned source's pandas grammar."""

    renderer = _renderer()
    pyplot, _, _, seaborn = renderer._plot_dependencies()
    data: Any = return_by_q.copy()
    target: Any = ax
    if ylim_percentiles is None:
        lower, upper = None, None
    else:
        lower, upper = _legacy_scaled_percentile_limits(data.to_numpy(dtype=float, copy=False), ylim_percentiles)
    if target is None:
        _, target = pyplot.subplots(1, 1, figsize=(18, 6))
    displayed = data.multiply(10_000.0)
    displayed.columns = displayed.columns.set_names("forward_periods")
    if displayed.columns.has_duplicates:
        # Pandas 3's ``stack`` rejects duplicate column labels, whereas the
        # pinned plotting path accepted each physical forward-return column.
        # Preserve that source-visible long form positionally for strict
        # duplicate-period tear sheets.
        pieces: list[pd.DataFrame] = []
        for position, period in enumerate(displayed.columns):
            piece = displayed.iloc[:, position].rename("return").reset_index()
            piece["forward_periods"] = period
            pieces.append(piece)
        long = pd.concat(pieces, ignore_index=True)
    else:
        stacked = displayed.stack()
        stacked.name = "return"
        long = stacked.reset_index()
    seaborn.violinplot(
        data=long,
        x="factor_quantile",
        hue="forward_periods",
        y="return",
        orient="v",
        cut=0,
        inner="quartile",
        ax=target,
    )
    target.set(
        xlabel="",
        ylabel="Return (bps)",
        title="Period Wise Return By Factor Quantile",
        ylim=(lower, upper),
    )
    target.axhline(0.0, linestyle="-", color="black", lw=0.7, alpha=0.6)
    return target


def _legacy_plot_quantile_returns_bar(
    mean_ret_by_q: pd.DataFrame,
    by_group: bool,
    ylim_percentiles: tuple[float, float] | None,
    ax: object,
) -> Any:
    """Render the source bar-chart grammar at the strict compatibility edge."""

    renderer = _renderer()
    pyplot, _, _, _ = renderer._plot_dependencies()
    data: Any = mean_ret_by_q.copy()
    target: Any = ax
    if ylim_percentiles is None:
        lower, upper = None, None
    else:
        lower, upper = _legacy_scaled_percentile_limits(data.to_numpy(dtype=float, copy=False), ylim_percentiles)

    if by_group:
        groups = data.index.get_level_values("group").unique()
        if target is None:
            rows = ((len(groups) - 1) // 2) + 1
            _, created = pyplot.subplots(rows, 2, sharex=False, sharey=True, figsize=(18, 6 * rows))
            target = created.flatten()
        for axis, (group, grouped) in zip(target, data.groupby(level="group"), strict=False):
            grouped.xs(group, level="group").multiply(10_000.0).plot(kind="bar", title=group, ax=axis)
            axis.set(xlabel="", ylabel="Mean Return (bps)", ylim=(lower, upper))
        if len(groups) < len(target):
            target[-1].set_visible(False)
        return target

    if target is None:
        _, target = pyplot.subplots(1, 1, figsize=(18, 6))
    data.multiply(10_000.0).plot(kind="bar", title="Mean Period Wise Return By Factor Quantile", ax=target)
    target.set(xlabel="", ylabel="Mean Return (bps)", ylim=(lower, upper))
    return target


def _legacy_plot_mean_quantile_returns_spread_time_series(
    mean_returns_spread: pd.Series | pd.DataFrame,
    std_err: pd.Series | pd.DataFrame | None,
    bandwidth: float,
    ax: object,
) -> Any:
    """Keep the pinned spread renderer's positional and empty-data semantics."""

    renderer = _renderer()
    pyplot, _, _, _ = renderer._plot_dependencies()
    source_values: Any = mean_returns_spread
    source_error: Any = std_err
    target: Any = ax
    if isinstance(mean_returns_spread, pd.DataFrame):
        axes: Any = [None for _ in mean_returns_spread.columns] if target is None else target
        lower, upper = None, None
        for (ordinal, axis), (name, values) in zip(enumerate(axes), source_values.items(), strict=False):
            error = None if source_error is None else source_error[name]
            rendered = _legacy_plot_mean_quantile_returns_spread_time_series(values, error, 1, axis)
            axes[ordinal] = rendered
            current_lower, current_upper = rendered.get_ylim()
            lower = current_lower if lower is None else min(lower, current_lower)
            upper = current_upper if upper is None else max(upper, current_upper)
        for axis in axes:
            axis.set_ylim([lower, upper])
        return axes

    if source_values.isnull().all():
        return target
    periods = source_values.name
    title = f"Top Minus Bottom Quantile Mean Return ({periods if periods is not None else ''} Period Forward Return)"
    if target is None:
        _, target = pyplot.subplots(figsize=(18, 6))
    displayed = source_values * 10_000.0
    displayed.plot(alpha=0.4, ax=target, lw=0.7, color="forestgreen")
    displayed.rolling(window=22).mean().plot(color="orangered", alpha=0.7, ax=target)
    target.legend(["mean returns spread", "1 month moving avg"], loc="upper right")
    if source_error is not None:
        error = source_error * 10_000.0
        upper = displayed.to_numpy() + error * bandwidth
        lower = displayed.to_numpy() - error * bandwidth
        target.fill_between(source_values.index, lower, upper, alpha=0.3, color="steelblue")
    limit = np.nanpercentile(abs(displayed.to_numpy()), 95)
    target.set(ylabel="Difference In Quantile Mean Return (bps)", xlabel="", title=title, ylim=(-limit, limit))
    target.axhline(0.0, linestyle="-", color="black", lw=1, alpha=0.8)
    return target


def _legacy_plot_ic_by_group(ic_group: pd.DataFrame, ax: object) -> Any:
    """Render group ICs with the source empty-frame and ownership behavior."""

    renderer = _renderer()
    pyplot, _, _, _ = renderer._plot_dependencies()
    source: Any = ic_group
    target: Any = ax
    if target is None:
        _, target = pyplot.subplots(1, 1, figsize=(18, 6))
    source.plot(kind="bar", ax=target)
    target.set(title="Information Coefficient By Group", xlabel="")
    target.set_xticklabels(source.index, rotation=45)
    return target


def _legacy_plot_factor_rank_auto_correlation(factor_autocorrelation: pd.Series, period: int, ax: object) -> Any:
    """Keep source Series-plot metadata and its visible mean annotation."""

    renderer = _renderer()
    pyplot, _, _, _ = renderer._plot_dependencies()
    source: Any = factor_autocorrelation
    target: Any = ax
    if target is None:
        _, target = pyplot.subplots(1, 1, figsize=(18, 6))
    source.plot(title=f"{period}D Period Factor Rank Autocorrelation", ax=target)
    target.set(ylabel="Autocorrelation Coefficient", xlabel="")
    target.axhline(0.0, linestyle="-", color="black", lw=1)
    target.text(
        0.05,
        0.95,
        f"Mean {source.mean():.3f}",
        fontsize=16,
        bbox={"facecolor": "white", "alpha": 1, "pad": 5},
        transform=target.transAxes,
        verticalalignment="top",
    )
    return target


def _legacy_plot_top_bottom_quantile_turnover(quantile_turnover: pd.DataFrame, period: int, ax: object) -> Any:
    """Use the source column-selection path, including its empty-frame error."""

    renderer = _renderer()
    pyplot, _, _, _ = renderer._plot_dependencies()
    source: Any = quantile_turnover
    target: Any = ax
    if target is None:
        _, target = pyplot.subplots(1, 1, figsize=(18, 6))
    maximum = source.columns.max()
    minimum = source.columns.min()
    turnover = pd.DataFrame()
    turnover["top quantile turnover"] = source[maximum]
    turnover["bottom quantile turnover"] = source[minimum]
    turnover.plot(title=f"{period}D Period Top and Bottom Quantile Turnover", ax=target, alpha=0.6, lw=0.8)
    target.set(ylabel="Proportion Of Names New To Quantile", xlabel="")
    return target


def _legacy_plot_monthly_ic_heatmap(mean_monthly_ic: pd.DataFrame, ax: object) -> Any:
    """Render the pinned year-by-month heatmaps, including source empty errors."""

    renderer = _renderer()
    pyplot, cm, _, seaborn = renderer._plot_dependencies()
    data: Any = mean_monthly_ic.copy()
    target: Any = ax
    count = len(data.columns)
    rows = ((count - 1) // 3) + 1
    if target is None:
        _, created = pyplot.subplots(rows, 3, figsize=(18, rows * 6))
        target = created.flatten()
    years = [date.year for date in data.index]
    months = [date.month for date in data.index]
    data.index = pd.MultiIndex.from_arrays([years, months], names=["year", "month"])
    for axis, (period, values) in zip(target, data.items(), strict=False):
        seaborn.heatmap(
            values.unstack(),
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
        axis.set(ylabel="", xlabel="")
        axis.set_title(f"Monthly Mean {period} Period IC")
    if count < len(target):
        target[-1].set_visible(False)
    return target


def _legacy_plot_cumulative_returns_by_quantile(quantile_returns: pd.DataFrame, period: object, ax: object) -> Any:
    """Render strict quantile curves with source limits, warnings, and ordering."""

    from fincore.alphalens.performance import cumulative_returns

    wide = quantile_returns.unstack("factor_quantile")
    cumulative = wide.apply(cumulative_returns).loc[:, ::-1]
    return _legacy_plot_cumulative_returns_by_quantile_values(cumulative, period, ax)


def _legacy_plot_cumulative_returns_by_quantile_values(
    cumulative: pd.DataFrame,
    period: object,
    ax: object,
) -> Any:
    """Render strict quantile curves that were precomputed during model assembly."""

    renderer = _renderer()
    pyplot, cm, ticker, _ = renderer._plot_dependencies()
    target: Any = ax
    if target is None:
        _, target = pyplot.subplots(1, 1, figsize=(18, 6))
    cumulative.plot(lw=2, ax=target, cmap=cm.coolwarm)
    target.legend()
    lower, upper = cumulative.min().min(), cumulative.max().max()
    target.set(
        ylabel="Log Cumulative Returns",
        title=f"Cumulative Return by Quantile\n                    ({period} Period Forward Return)",
        xlabel="",
        yscale="symlog",
        yticks=np.linspace(lower, upper, 5),
        ylim=(lower, upper),
    )
    target.yaxis.set_major_formatter(ticker.ScalarFormatter())
    target.axhline(1.0, linestyle="-", color="black", lw=1)
    return target


def _legacy_plot_cumulative_returns(factor_returns: pd.Series, period: object, title: str | None, ax: object) -> Any:
    """Render the source strict cumulative-return Series projection."""

    from fincore.alphalens.performance import cumulative_returns

    return _legacy_plot_cumulative_returns_values(cumulative_returns(factor_returns), period, title, ax)


def _legacy_plot_cumulative_returns_values(
    cumulative: pd.Series,
    period: object,
    title: str | None,
    ax: object,
) -> Any:
    """Render a strict portfolio curve precomputed during model assembly."""

    renderer = _renderer()
    pyplot, _, _, _ = renderer._plot_dependencies()
    target: Any = ax
    if target is None:
        _, target = pyplot.subplots(1, 1, figsize=(18, 6))
    values = cumulative.copy(deep=True)
    if len(values):
        # The strict empyrical projection constructs a fresh nonempty Series.
        values.name = None
    values.plot(ax=target, lw=3, color="forestgreen", alpha=0.6)
    target.set(
        ylabel="Cumulative Returns",
        title=f"Portfolio Cumulative Return ({period} Fwd Period)" if title is None else title,
        xlabel="",
    )
    target.axhline(1.0, linestyle="-", color="black", lw=1)
    return target


def _legacy_plot_quantile_average_cumulative_return(
    avg_cumulative_returns: pd.DataFrame,
    by_quantile: bool,
    std_bar: bool,
    title: str | None,
    ax: object,
) -> Any:
    """Render the strict source event-window chart rather than safe renderer data."""

    renderer = _renderer()
    pyplot, cm, _, _ = renderer._plot_dependencies()
    data: Any = avg_cumulative_returns.multiply(10_000.0)
    target: Any = ax
    quantile_count = len(data.index.levels[0].unique())
    palette = [cm.coolwarm(value) for value in np.linspace(0, 1, quantile_count)][::-1]
    if by_quantile:
        if target is None:
            rows = ((quantile_count - 1) // 2) + 1
            _, created = pyplot.subplots(rows, 2, sharex=False, sharey=False, figsize=(18, 6 * rows))
            target = created.flatten()
        for ordinal, (quantile, values) in enumerate(data.groupby(level="factor_quantile")):
            mean = values.loc[(quantile, "mean")]
            mean.name = f"Quantile {quantile}"
            mean.plot(ax=target[ordinal], color=palette[ordinal])
            target[ordinal].set_ylabel("Mean Return (bps)")
            if std_bar:
                standard_deviation = values.loc[(quantile, "std")]
                target[ordinal].errorbar(
                    standard_deviation.index,
                    mean,
                    yerr=standard_deviation,
                    fmt="none",
                    ecolor=palette[ordinal],
                    label="none",
                )
            target[ordinal].axvline(x=0, color="k", linestyle="--")
            target[ordinal].legend()
        return target

    if target is None:
        _, target = pyplot.subplots(1, 1, figsize=(18, 6))
    for ordinal, (quantile, values) in enumerate(data.groupby(level="factor_quantile")):
        mean = values.loc[(quantile, "mean")]
        mean.name = f"Quantile {quantile}"
        mean.plot(ax=target, color=palette[ordinal])
        if std_bar:
            standard_deviation = values.loc[(quantile, "std")]
            target.errorbar(
                standard_deviation.index,
                mean,
                yerr=standard_deviation,
                fmt="none",
                ecolor=palette[ordinal],
                label="none",
            )
    target.axvline(x=0, color="k", linestyle="--")
    target.legend()
    target.set(
        ylabel="Mean Return (bps)",
        title="Average Cumulative Returns by Quantile" if title is None else title,
        xlabel="Periods",
    )
    return target


def _legacy_plot_events_distribution(events: pd.Series, num_bars: object, ax: object) -> Any:
    """Use the source's permissive interval grammar and categorical bars."""

    renderer = _renderer()
    pyplot, _, _, _ = renderer._plot_dependencies()
    source: Any = events
    target: Any = ax
    bars: Any = num_bars
    if target is None:
        _, target = pyplot.subplots(1, 1, figsize=(18, 6))
    start = source.index.get_level_values("date").min()
    end = source.index.get_level_values("date").max()
    interval = (end - start) / bars
    counts = source.groupby(pd.Grouper(level="date", freq=interval)).count()
    counts.plot(kind="bar", grid=False, ax=target)
    target.set(ylabel="Number of events", title="Distribution of events in time", xlabel="Date")
    return target


def _renderer() -> Any:
    """Load the enhanced visual implementation at a deliberate call boundary."""

    return importlib.import_module("fincore.factor_analysis.render_matplotlib")


def _attach_spec(function: Callable[..., Any], name: str) -> Callable[..., Any]:
    """Expose the exact frozen source signature independently of annotations."""

    spec = _spec(name)
    function.__signature__ = spec.introspection_signature  # type: ignore[attr-defined]
    function.__fincore_source_signature__ = spec.source_signature  # type: ignore[attr-defined]
    function.__fincore_factor_spec__ = spec  # type: ignore[attr-defined]
    return function


def _display_table(heading: str | None, table: pd.DataFrame, *, round_values: bool = True) -> None:
    """Lazily retain the pinned IPython display side effect and ``None`` return."""

    if heading:
        print(heading)
    displayed = table.round(3) if round_values else table
    try:
        display = importlib.import_module("IPython.display").display
    except ModuleNotFoundError:
        # A terminal-only installation still receives readable output while
        # notebook consumers preserve the source display-hook behavior.
        print(displayed.to_string())
    else:
        display(displayed)


def customize(func: Callable[..., Any]) -> Callable[..., Any]:
    """Return a strict legacy context decorator accepting hidden ``set_context``."""

    decorated = _renderer().customize(func)

    @wraps(decorated)
    def strict_decorated(*args: Any, **kwargs: Any) -> Any:
        return decorated(*args, **kwargs)

    return strict_decorated


def plotting_context(context: str = "notebook", font_scale: float = 1.5, rc: Mapping[str, object] | None = None) -> Any:
    """Return the lazily-created strict Alphalens seaborn context."""

    return _renderer().plotting_context(context=context, font_scale=font_scale, rc=rc)


def axes_style(style: str = "darkgrid", rc: Mapping[str, object] | None = None) -> Any:
    """Return the lazily-created strict Alphalens axes-style context."""

    return _renderer().axes_style(style=style, rc=rc)


def plot_returns_table(
    alpha_beta: pd.DataFrame,
    mean_ret_quantile: pd.DataFrame,
    mean_ret_spread_quantile: pd.Series | pd.DataFrame,
) -> None:
    _display_table(
        "Returns Analysis", _renderer().build_returns_table(alpha_beta, mean_ret_quantile, mean_ret_spread_quantile)
    )


def plot_turnover_table(
    autocorrelation_data: Mapping[object, pd.Series], quantile_turnover: Mapping[object, pd.DataFrame]
) -> None:
    turnover, autocorrelation = _renderer().build_turnover_tables(autocorrelation_data, quantile_turnover)
    _display_table("Turnover Analysis", turnover)
    _display_table(None, autocorrelation)


def plot_information_table(ic_data: pd.DataFrame) -> None:
    _display_table("Information Analysis", _renderer().build_information_table(ic_data).T)


def plot_quantile_statistics_table(factor_data: pd.DataFrame) -> None:
    # The source prepares but does not round this table before its display
    # helper; unlike the other three strict table functions, full precision is
    # observable to notebook display hooks.
    _display_table(
        "Quantiles Statistics",
        _renderer().build_quantile_statistics_table(factor_data),
        round_values=False,
    )


def plot_ic_ts(ic: pd.DataFrame, ax: object = None) -> Any:
    return _legacy_plot_ic_ts(ic, ax)


def plot_ic_hist(ic: pd.DataFrame, ax: object = None) -> Any:
    return _legacy_plot_ic_hist(ic, ax)


def plot_ic_qq(ic: pd.DataFrame, theoretical_dist: object = _DEFAULT_THEORETICAL_DIST, ax: object = None) -> Any:
    return _legacy_plot_ic_qq(ic, theoretical_dist, ax)


def plot_quantile_returns_bar(
    mean_ret_by_q: pd.DataFrame,
    by_group: bool = False,
    ylim_percentiles: tuple[float, float] | None = None,
    ax: object = None,
) -> Any:
    return _legacy_plot_quantile_returns_bar(mean_ret_by_q, by_group, ylim_percentiles, ax)


def plot_quantile_returns_violin(
    return_by_q: pd.DataFrame,
    ylim_percentiles: tuple[float, float] | None = None,
    ax: object = None,
) -> Any:
    return _legacy_plot_quantile_returns_violin(return_by_q, ylim_percentiles, ax)


def plot_mean_quantile_returns_spread_time_series(
    mean_returns_spread: pd.Series | pd.DataFrame,
    std_err: pd.Series | pd.DataFrame | None = None,
    bandwidth: float = 1,
    ax: object = None,
) -> Any:
    return _legacy_plot_mean_quantile_returns_spread_time_series(mean_returns_spread, std_err, bandwidth, ax)


def plot_ic_by_group(ic_group: pd.DataFrame, ax: object = None) -> Any:
    return _legacy_plot_ic_by_group(ic_group, ax)


def plot_factor_rank_auto_correlation(factor_autocorrelation: pd.Series, period: int = 1, ax: object = None) -> Any:
    return _legacy_plot_factor_rank_auto_correlation(factor_autocorrelation, period, ax)


def plot_top_bottom_quantile_turnover(quantile_turnover: pd.DataFrame, period: int = 1, ax: object = None) -> Any:
    return _legacy_plot_top_bottom_quantile_turnover(quantile_turnover, period, ax)


def plot_monthly_ic_heatmap(mean_monthly_ic: pd.DataFrame, ax: object = None) -> Any:
    return _legacy_plot_monthly_ic_heatmap(mean_monthly_ic, ax)


def plot_cumulative_returns(
    factor_returns: pd.Series,
    period: object,
    freq: object = None,
    title: str | None = None,
    ax: object = None,
) -> Any:
    del freq
    return _legacy_plot_cumulative_returns(factor_returns, period, title, ax)


def plot_cumulative_returns_by_quantile(
    quantile_returns: pd.DataFrame, period: object, freq: object = None, ax: object = None
) -> Any:
    del freq
    return _legacy_plot_cumulative_returns_by_quantile(quantile_returns, period, ax)


def plot_quantile_average_cumulative_return(
    avg_cumulative_returns: pd.DataFrame,
    by_quantile: bool = False,
    std_bar: bool = False,
    title: str | None = None,
    ax: object = None,
) -> Any:
    return _legacy_plot_quantile_average_cumulative_return(avg_cumulative_returns, by_quantile, std_bar, title, ax)


def plot_events_distribution(events: pd.Series, num_bars: int = 50, ax: object = None) -> Any:
    return _legacy_plot_events_distribution(events, num_bars, ax)


for _name in _PLOTTING_NAMES:
    globals()[_name] = _attach_spec(cast("Callable[..., Any]", globals()[_name]), _name)

__all__ = _PLOTTING_NAMES

del _name
