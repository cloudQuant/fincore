"""Model-driven, lazy-rendered factor tear-sheet workflows.

The functions in this module are the enhanced side of the Alphalens tear
sheet boundary.  They consume an already-computed :class:`FactorAnalysisModel`
and never re-enter the numerical kernels.  Matplotlib is intentionally loaded
only when a workflow is rendered.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Mapping, cast

import pandas as pd

# These public annotations are deliberately runtime-resolvable for renderer
# consumers that call ``typing.get_type_hints``.
from fincore.factor_analysis.models import EventAnalysisModel, FactorAnalysisModel  # noqa: TC001

if TYPE_CHECKING:
    from matplotlib.figure import Figure
else:
    # Keep ``typing.get_type_hints`` usable without importing the optional
    # Matplotlib package at factor-analysis import time.
    Figure = Any


@dataclass(frozen=True, slots=True)
class FactorTearSheetArtifacts:
    """Caller-owned figures and renderer-ready tables for one workflow.

    Enhanced workflows do not display or close these figures by default.  The
    strict Alphalens facade uses the same artifact object internally, then
    applies its legacy show-and-close projection at the public boundary.
    """

    model: FactorAnalysisModel
    figures: tuple[Figure, ...]
    tables: Mapping[str, pd.DataFrame]

    def __post_init__(self) -> None:
        copied = {name: table.copy(deep=True) for name, table in self.tables.items()}
        object.__setattr__(self, "tables", MappingProxyType(copied))


class GridFigure:
    """One lazily-created figure with the pinned row/cell grid primitives."""

    def __init__(self, rows, cols):
        if type(rows) is object or type(cols) is object:
            raise NotImplementedError(
                "Legacy Alphalens symbol 'GridFigure' is available for C0/C1 compatibility, "
                "but its rendering kernel requires concrete row and column counts."
            )
        if isinstance(rows, bool) or not isinstance(rows, int) or rows < 1:
            raise ValueError("rows must be a positive integer")
        if isinstance(cols, bool) or not isinstance(cols, int) or cols < 1:
            raise ValueError("cols must be a positive integer")
        pyplot = importlib.import_module("matplotlib.pyplot")
        gridspec = importlib.import_module("matplotlib.gridspec")
        self.rows = rows
        self.cols = cols
        self.fig = pyplot.figure(figsize=(14, rows * 7))
        self.gs = gridspec.GridSpec(rows, cols, figure=self.fig, wspace=0.4, hspace=0.3)
        self.curr_row = 0
        self.curr_col = 0

    def create_new_figure(self) -> Figure:
        """Return the one figure owned by this grid for explicit ownership checks."""

        if self.fig is None:
            raise RuntimeError("GridFigure is closed")
        return cast("Figure", self.fig)

    def next_row(self) -> Any:
        """Allocate the next full-width row using the source cursor grammar."""

        if self.fig is None or self.gs is None:
            raise RuntimeError("GridFigure is closed")
        if self.curr_col != 0:
            self.curr_row += 1
            self.curr_col = 0
        axis = self.fig.add_subplot(self.gs[self.curr_row, :])
        self.curr_row += 1
        return axis

    def next_cell(self) -> Any:
        """Allocate the next cell, advancing to a new row when necessary."""

        if self.fig is None or self.gs is None:
            raise RuntimeError("GridFigure is closed")
        if self.curr_col >= self.cols:
            self.curr_row += 1
            self.curr_col = 0
        axis = self.fig.add_subplot(self.gs[self.curr_row, self.curr_col])
        self.curr_col += 1
        return axis

    def close(self) -> None:
        """Close only this grid's own figure and release its layout references."""

        if self.fig is None:
            return
        pyplot = importlib.import_module("matplotlib.pyplot")
        pyplot.close(self.fig)
        self.fig = None
        self.gs = None


def _renderer() -> Any:
    """Resolve the optional renderer strictly at workflow execution time."""

    return importlib.import_module("fincore.factor_analysis.render_matplotlib")


def _show_figures(figures: tuple[Figure, ...]) -> None:
    """Show each owned legacy figure once, matching source workflow sections."""

    pyplot = importlib.import_module("matplotlib.pyplot")
    for _ in figures:
        pyplot.show()


def close_owned_figures(artifacts: FactorTearSheetArtifacts) -> None:
    """Close exactly the figures returned by an enhanced workflow."""

    pyplot = importlib.import_module("matplotlib.pyplot")
    for figure in artifacts.figures:
        pyplot.close(figure)


def show_owned_figures(artifacts: FactorTearSheetArtifacts) -> None:
    """Display exactly the figures returned by one enhanced workflow."""

    _show_figures(artifacts.figures)


def _artifacts(
    model: FactorAnalysisModel,
    grids: list[GridFigure],
    tables: Mapping[str, pd.DataFrame],
    *,
    show: bool,
) -> FactorTearSheetArtifacts:
    """Freeze the visible workflow result without transferring close ownership."""

    figures = tuple(cast("Figure", grid.create_new_figure()) for grid in grids)
    return _artifacts_from_figures(model, figures, tables, show=show)


def _artifacts_from_figures(
    model: FactorAnalysisModel,
    figures: tuple[Figure, ...],
    tables: Mapping[str, pd.DataFrame],
    *,
    show: bool,
) -> FactorTearSheetArtifacts:
    """Freeze an explicit figure tuple without introducing a second owner."""

    artifacts = FactorTearSheetArtifacts(model=model, figures=figures, tables=tables)
    if show:
        show_owned_figures(artifacts)
    return artifacts


def _overall_quantile_frame(value: pd.DataFrame) -> pd.DataFrame:
    """Derive a non-group display frame from compute-once grouped model data."""

    if isinstance(value.index, pd.MultiIndex) and "group" in value.index.names:
        levels = [name for name in value.index.names if name != "group"]
        return value.groupby(level=levels, observed=True, sort=True).mean()
    return value.copy(deep=True)


def _overall_information(value: pd.DataFrame) -> pd.DataFrame:
    """Collapse only the optional group index for charts expecting date rows."""

    if isinstance(value.index, pd.MultiIndex) and "group" in value.index.names:
        return value.groupby(level="date", observed=True, sort=True).mean()
    return value.copy(deep=True)


def _overall_monthly(model: FactorAnalysisModel) -> pd.DataFrame:
    """Read the model's monthly IC snapshot without calling an aggregation kernel."""

    monthly = model.aggregate_time_aggregated_results.get("M")
    if monthly is None:
        return model.aggregate_information_coefficient.resample("ME").mean()
    if isinstance(monthly, pd.Series):
        monthly = monthly.to_frame()
    return monthly.copy(deep=True)


def _aggregate_monthly(model: FactorAnalysisModel) -> pd.DataFrame:
    """Read the model's source-equivalent aggregate monthly IC snapshot."""

    monthly = model.aggregate_time_aggregated_results.get("M")
    if monthly is None:
        return model.aggregate_information_coefficient.resample("ME").mean()
    if isinstance(monthly, pd.Series):
        monthly = monthly.to_frame()
    return monthly.copy(deep=True)


def _rank_series_by_period(model: FactorAnalysisModel) -> dict[int, pd.Series]:
    """Split the stored rank-autocorrelation table for table and chart helpers."""

    ranks: dict[int, pd.Series] = {}
    for period in model.rank_autocorrelation:
        if isinstance(period, bool) or not isinstance(period, int):  # pragma: no cover - model contract guard
            raise TypeError("rank-autocorrelation periods must be integers")
        ranks[period] = model.rank_autocorrelation[period].copy(deep=True)
    return ranks


def _group_mean_returns(model: FactorAnalysisModel) -> pd.DataFrame | None:
    """Compose source-shaped group display data from model-owned group snapshots."""

    direct = model.mean_returns_by_quantile
    if isinstance(direct.index, pd.MultiIndex) and "group" in direct.index.names:
        return direct.copy(deep=True)
    if not model.grouped_results:
        return None
    pieces = {group: result.mean_returns_by_quantile for group, result in model.grouped_results.items()}
    combined = pd.concat(pieces, names=["group"])
    if not isinstance(combined.index, pd.MultiIndex):  # pragma: no cover - model contract
        return None
    return combined.reorder_levels(["factor_quantile", "group"]).sort_index()


def _returns_group_section(
    model: FactorAnalysisModel,
    *,
    legacy_projection: bool,
    plotter: Any | None,
) -> FactorTearSheetArtifacts:
    """Render only returns' optional by-group source section.

    The strict facade already displayed the primary grid.  Keeping this
    separate avoids allocating a second hidden primary grid just to obtain
    the group panel from the enhanced composite helper.
    """

    renderer = _renderer()
    charts = renderer if plotter is None else plotter
    group_returns = _group_mean_returns(model)
    if group_returns is None or group_returns.empty:
        return _artifacts(model, [], {}, show=False)
    if legacy_projection:
        group_returns = _rate_of_return(group_returns)
    groups = tuple(group_returns.index.get_level_values("group").unique())
    group_rows = 1 + (((len(groups) - 1) // 2) + 1) if legacy_projection else max(1, (len(groups) + 1) // 2)
    grid = GridFigure(group_rows, 2)
    charts.plot_quantile_returns_bar(
        group_returns,
        by_group=True,
        ylim_percentiles=(5, 95),
        ax=[grid.next_cell() for _ in groups],
    )
    return _artifacts(model, [grid], {}, show=False)


def _event_data(model: FactorAnalysisModel) -> EventAnalysisModel:
    """Require the complete event snapshot only for event workflows."""

    event = model.event_returns
    if event is None:
        raise ValueError("event tear sheets require a model built with complete event_returns and event window bounds")
    return event


def _event_average_without_group(event: EventAnalysisModel) -> pd.DataFrame:
    """Derive the aggregate event display from a grouped compute-once snapshot."""

    return event.aggregate_quantile_average_returns.copy(deep=True)


def _rate_of_return(value: pd.DataFrame) -> pd.DataFrame:
    """Project each forward-return column onto the first source period."""

    if value.empty or not len(value.columns):
        return value.copy(deep=True)
    base_period = value.columns[0]
    if value.columns.has_duplicates:
        # ``DataFrame.apply`` aligns results by label and pandas rejects that
        # alignment for duplicate forward labels.  The source paths operate
        # column-by-column, so keep the same positional relationship without
        # re-entering an analytical kernel.
        converted = value.copy(deep=True)
        for position, period in enumerate(value.columns):
            converted.iloc[:, position] = (
                value.iloc[:, position].add(1.0).pow(pd.Timedelta(base_period) / pd.Timedelta(period)).sub(1.0)
            )
        return converted
    return cast(
        "pd.DataFrame",
        value.apply(
            lambda period_values: (
                period_values.add(1.0).pow(pd.Timedelta(base_period) / pd.Timedelta(period_values.name)).sub(1.0)
            ),
            axis=0,
        ),
    )


def _std_conversion(value: pd.DataFrame) -> pd.DataFrame:
    """Convert standard errors to the first source period without re-analysis."""

    if value.empty or not len(value.columns):
        return value.copy(deep=True)
    base_period = value.columns[0]
    return cast(
        "pd.DataFrame",
        value.apply(
            lambda period_values: period_values / (pd.Timedelta(period_values.name) / pd.Timedelta(base_period)) ** 0.5,
            axis=0,
        ),
    )


def _quantile_slice(value: pd.DataFrame, quantile: object) -> pd.DataFrame:
    """Select one source quantile while retaining its date-indexed table shape."""

    if isinstance(value.index, pd.MultiIndex):
        selected = value.xs(quantile, level="factor_quantile")
    else:
        selected = value.loc[[quantile]]
    if isinstance(selected, pd.Series):  # pragma: no cover - DataFrame selection invariant
        return selected.to_frame().T
    return selected


def _spread_from_converted_by_date(
    model: FactorAnalysisModel,
    mean_by_date: pd.DataFrame,
    std_error_by_date: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rebuild the source spread/error projection from model-owned daily fields."""

    quantiles = model.factor_data["factor_quantile"].dropna()
    if quantiles.empty:
        return pd.DataFrame(columns=mean_by_date.columns), pd.DataFrame(columns=mean_by_date.columns)
    upper = quantiles.max()
    lower = quantiles.min()
    upper_mean = _quantile_slice(mean_by_date, upper)
    lower_mean = _quantile_slice(mean_by_date, lower)
    upper_std = _quantile_slice(std_error_by_date, upper)
    lower_std = _quantile_slice(std_error_by_date, lower)
    return upper_mean - lower_mean, (upper_std**2 + lower_std**2) ** 0.5


def _returns_display_data(
    model: FactorAnalysisModel,
    *,
    legacy_projection: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    """Return chart/table values from one model, with an opt-in source projection."""

    mean_returns = model.aggregate_mean_returns_by_quantile.copy(deep=True)
    mean_by_date = model.aggregate_mean_returns_by_date.copy(deep=True)
    if not legacy_projection:
        spread_std = model.aggregate_mean_return_spread_std
        return mean_returns, mean_by_date, model.aggregate_mean_return_spread.copy(deep=True), spread_std

    converted_returns = _rate_of_return(mean_returns)
    converted_by_date = _rate_of_return(mean_by_date)
    converted_std = _std_conversion(model.aggregate_std_error_by_date.copy(deep=True))
    spread, spread_std = _spread_from_converted_by_date(model, converted_by_date, converted_std)
    return converted_returns, converted_by_date, spread, spread_std


def _returns_tables(
    model: FactorAnalysisModel,
    *,
    mean_returns: pd.DataFrame | None = None,
    mean_return_spread: pd.DataFrame | None = None,
    information_coefficient: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """Build all tabular sections from fields already stored on the model."""

    renderer = _renderer()
    if mean_returns is None:
        mean_returns = model.aggregate_mean_returns_by_quantile.copy(deep=True)
    if mean_return_spread is None:
        mean_return_spread = model.aggregate_mean_return_spread.copy(deep=True)
    if information_coefficient is None:
        information_coefficient = model.aggregate_information_coefficient.copy(deep=True)
    ranks = _rank_series_by_period(model)
    turnover, autocorrelation = renderer.build_turnover_tables(ranks, model.quantile_turnover)
    return {
        "quantile_statistics": renderer.build_quantile_statistics_table(model.factor_data),
        "returns": renderer.build_returns_table(model.alpha_beta, mean_returns, mean_return_spread),
        "information": renderer.build_information_table(information_coefficient),
        "turnover": turnover,
        "autocorrelation": autocorrelation,
    }


def create_summary_tear_sheet(
    model: FactorAnalysisModel,
    *,
    show: bool = False,
    legacy_projection: bool = False,
    plotter: Any | None = None,
) -> FactorTearSheetArtifacts:
    """Render the compact returns, information, and turnover sections from one model."""

    renderer = _renderer()
    charts = renderer if plotter is None else plotter
    mean_returns, _, mean_spread, _ = _returns_display_data(model, legacy_projection=legacy_projection)
    information = (
        model.summary_information_coefficient.copy(deep=True)
        if legacy_projection
        else model.aggregate_information_coefficient.copy(deep=True)
    )
    tables = _returns_tables(
        model,
        mean_returns=mean_returns,
        mean_return_spread=mean_spread,
        information_coefficient=information,
    )
    if legacy_projection:
        # The pinned summary sheet reserves a taller grid but renders only the
        # quantile-return bar; IC/rank diagnostics are table-only here.
        grid = GridFigure(2 + len(mean_returns.columns) * 3, 1)
        charts.plot_quantile_returns_bar(mean_returns, ax=grid.next_row())
        return _artifacts(model, [grid], tables, show=show)

    information = model.aggregate_information_coefficient.copy(deep=True)
    ranks = _rank_series_by_period(model)
    grid = GridFigure(2 + len(information.columns), 1)
    charts.plot_quantile_returns_bar(mean_returns, ax=grid.next_row())
    charts.plot_ic_ts(information, ax=[grid.next_row() for _ in information.columns])
    period = next(iter(ranks))
    charts.plot_factor_rank_auto_correlation(ranks[period], period=period, ax=grid.next_row())
    return _artifacts(model, [grid], tables, show=show)


def create_returns_tear_sheet(
    model: FactorAnalysisModel,
    *,
    by_group: bool | None = None,
    show: bool = False,
    legacy_projection: bool = False,
    plotter: Any | None = None,
) -> FactorTearSheetArtifacts:
    """Render portfolio, quantile, spread, and optional group returns sections."""

    renderer = _renderer()
    charts = renderer if plotter is None else plotter
    mean_returns, mean_by_date, mean_spread, spread_std = _returns_display_data(
        model, legacy_projection=legacy_projection
    )
    tables = _returns_tables(model, mean_returns=mean_returns, mean_return_spread=mean_spread)
    grids: list[GridFigure] = []
    count = max(1, len(model.forward_periods))
    grid = GridFigure((2 if legacy_projection else 3) + count * 3, 1)
    grids.append(grid)
    charts.plot_quantile_returns_bar(mean_returns, ax=grid.next_row())
    charts.plot_quantile_returns_violin(mean_by_date, ylim_percentiles=(1, 99), ax=grid.next_row())
    for period in model.forward_periods:
        title = None
        if legacy_projection:
            title = (
                "Factor Weighted "
                + ("Group Neutral " if model.config.group_neutral else "")
                + ("Long/Short " if model.config.long_short else "")
                + f"Portfolio Cumulative Return ({period}Period)"
            )
        if legacy_projection:
            charts._legacy_plot_cumulative_returns_values(
                model.factor_cumulative_returns[period],
                period,
                title,
                grid.next_row(),
            )
            charts._legacy_plot_cumulative_returns_by_quantile_values(
                model.legacy_quantile_cumulative_returns[period],
                period,
                grid.next_row(),
            )
        else:
            charts.plot_cumulative_returns(model.factor_returns[period], period=period, title=title, ax=grid.next_row())
            charts.plot_cumulative_returns_by_quantile(
                mean_by_date[period].to_frame(period), period=period, ax=grid.next_row()
            )
    charts.plot_mean_quantile_returns_spread_time_series(
        mean_spread,
        std_err=spread_std,
        bandwidth=0.5,
        ax=[grid.next_row() for _ in mean_spread.columns],
    )

    render_groups = model.config.by_group if by_group is None else by_group
    group_returns = _group_mean_returns(model) if render_groups else None
    if group_returns is not None and not group_returns.empty:
        if legacy_projection:
            group_returns = _rate_of_return(group_returns)
        groups = tuple(group_returns.index.get_level_values("group").unique())
        group_rows = 1 + (((len(groups) - 1) // 2) + 1) if legacy_projection else max(1, (len(groups) + 1) // 2)
        group_grid = GridFigure(group_rows, 2)
        grids.append(group_grid)
        charts.plot_quantile_returns_bar(
            group_returns,
            by_group=True,
            ylim_percentiles=(5, 95),
            ax=[group_grid.next_cell() for _ in groups],
        )
    return _artifacts(model, grids, tables, show=show)


def create_information_tear_sheet(
    model: FactorAnalysisModel,
    *,
    by_group: bool | None = None,
    show: bool = False,
    legacy_projection: bool = False,
    plotter: Any | None = None,
) -> FactorTearSheetArtifacts:
    """Render IC time-series, distributions, Q-Q, monthly, and group sections."""

    renderer = _renderer()
    charts = renderer if plotter is None else plotter
    tables = _returns_tables(model)
    information = model.aggregate_information_coefficient.copy(deep=True)
    count = max(1, len(information.columns))
    rows = count + 3 * (((count - 1) // 2) + 1) + 2 * count if legacy_projection else 2 + count * 3
    grid = GridFigure(rows, 2)
    charts.plot_ic_ts(information, ax=[grid.next_row() for _ in information.columns])
    if legacy_projection:
        # Pinned tears allocates each histogram/Q-Q pair beside one another
        # in a shared two-column row, not histogram rows followed by Q-Q rows.
        paired_axes = [grid.next_cell() for _ in range(count * 2)]
        histogram_axes = paired_axes[::2]
        qq_axes = paired_axes[1::2]
    else:
        histogram_axes = [grid.next_cell() for _ in range(count)]
        qq_axes = [grid.next_cell() for _ in range(count)]
    charts.plot_ic_hist(information, ax=histogram_axes)
    charts.plot_ic_qq(information, ax=qq_axes)
    render_groups = model.config.by_group if by_group is None else by_group
    if not legacy_projection or not render_groups:
        monthly = _aggregate_monthly(model) if legacy_projection else _overall_monthly(model)
        charts.plot_monthly_ic_heatmap(monthly, ax=[grid.next_cell() for _ in monthly.columns])
    if render_groups and isinstance(model.mean_information_coefficient, pd.DataFrame):
        charts.plot_ic_by_group(model.mean_information_coefficient, ax=grid.next_row())
    return _artifacts(model, [grid], tables, show=show)


def create_turnover_tear_sheet(
    model: FactorAnalysisModel,
    *,
    turnover_periods: tuple[int, ...] | None = None,
    show: bool = False,
    legacy_projection: bool = False,
    plotter: Any | None = None,
) -> FactorTearSheetArtifacts:
    """Render stored quantile-turnover and rank-autocorrelation sections."""

    renderer = _renderer()
    charts = renderer if plotter is None else plotter
    tables = _returns_tables(model)
    periods = tuple(model.quantile_turnover) if turnover_periods is None else turnover_periods
    grid = GridFigure(max(1, len(periods) * (6 if legacy_projection else 2)), 1)
    ranks = _rank_series_by_period(model)
    for period in periods:
        turnover = model.quantile_turnover[period]
        if not turnover.isna().all().all():
            charts.plot_top_bottom_quantile_turnover(turnover, period=period, ax=grid.next_row())
        rank = ranks.get(period)
        if rank is not None and not rank.isna().all():
            charts.plot_factor_rank_auto_correlation(rank, period=period, ax=grid.next_row())
    return _artifacts(model, [grid], tables, show=show)


def _combine(
    model: FactorAnalysisModel, sections: tuple[FactorTearSheetArtifacts, ...], *, show: bool
) -> FactorTearSheetArtifacts:
    """Combine child artifacts without changing their caller ownership semantics."""

    figures = tuple(figure for section in sections for figure in section.figures)
    tables: dict[str, pd.DataFrame] = {}
    for prefix, section in zip(("returns", "information", "turnover"), sections, strict=False):
        tables.update({f"{prefix}.{name}": table for name, table in section.tables.items()})
    tables["quantile_statistics"] = _renderer().build_quantile_statistics_table(model.factor_data)
    artifacts = FactorTearSheetArtifacts(model=model, figures=figures, tables=tables)
    if show:
        show_owned_figures(artifacts)
    return artifacts


def create_full_tear_sheet(
    model: FactorAnalysisModel,
    *,
    by_group: bool | None = None,
    show: bool = False,
    legacy_projection: bool = False,
    plotter: Any | None = None,
) -> FactorTearSheetArtifacts:
    """Compose the returns, information, and turnover artifacts from one model."""

    sections = (
        create_returns_tear_sheet(
            model, by_group=by_group, show=False, legacy_projection=legacy_projection, plotter=plotter
        ),
        create_information_tear_sheet(
            model, by_group=by_group, show=False, legacy_projection=legacy_projection, plotter=plotter
        ),
        create_turnover_tear_sheet(model, show=False, legacy_projection=legacy_projection, plotter=plotter),
    )
    return _combine(model, sections, show=show)


def create_event_returns_tear_sheet(
    model: FactorAnalysisModel,
    *,
    std_bar: bool = True,
    by_group: bool | None = None,
    show: bool = False,
    plotter: Any | None = None,
    _include_returns_tables: bool = True,
) -> FactorTearSheetArtifacts:
    """Render aggregate and optional group event-window return sections."""

    renderer = _renderer()
    charts = renderer if plotter is None else plotter
    event = _event_data(model)
    average = _event_average_without_group(event)
    quantiles = tuple(average.index.get_level_values("factor_quantile").unique())
    grid = GridFigure(
        1 + (len(quantiles) + 1) // 2 if std_bar else 1,
        2 if len(quantiles) != 1 else 1,
    )
    charts.plot_quantile_average_cumulative_return(
        average,
        by_quantile=False,
        std_bar=False,
        ax=grid.next_row(),
    )
    if std_bar:
        charts.plot_quantile_average_cumulative_return(
            average,
            by_quantile=True,
            std_bar=True,
            ax=[grid.next_cell() for _ in quantiles],
        )
    grids = [grid]
    render_groups = model.config.by_group if by_group is None else by_group
    if (
        render_groups
        and isinstance(event.quantile_average_returns.index, pd.MultiIndex)
        and "group" in event.quantile_average_returns.index.names
    ):
        grouped_averages = tuple(event.quantile_average_returns.groupby(level="group", observed=True, sort=True))
        group_grid = GridFigure(max(1, (len(grouped_averages) + 1) // 2), 2)
        for group, group_average in grouped_averages:
            group_average = group_average.droplevel("group")
            charts.plot_quantile_average_cumulative_return(
                group_average,
                by_quantile=False,
                std_bar=False,
                title=str(group),
                ax=group_grid.next_cell(),
            )
        grids.append(group_grid)
    tables = _returns_tables(model) if _include_returns_tables else {}
    tables["event_average"] = average
    tables["event_windows"] = event.event_windows
    return _artifacts(model, grids, tables, show=show)


def _event_returns_group_section(
    model: FactorAnalysisModel,
    *,
    plotter: Any | None,
) -> FactorTearSheetArtifacts:
    """Render only event returns' optional by-group source section."""

    renderer = _renderer()
    charts = renderer if plotter is None else plotter
    event = _event_data(model)
    if not (
        isinstance(event.quantile_average_returns.index, pd.MultiIndex)
        and "group" in event.quantile_average_returns.index.names
    ):
        return _artifacts(model, [], {}, show=False)
    grouped_averages = tuple(event.quantile_average_returns.groupby(level="group", observed=True, sort=True))
    if not grouped_averages:
        return _artifacts(model, [], {}, show=False)
    grid = GridFigure(max(1, (len(grouped_averages) + 1) // 2), 2)
    for group, group_average in grouped_averages:
        charts.plot_quantile_average_cumulative_return(
            group_average.droplevel("group"),
            by_quantile=False,
            std_bar=False,
            title=str(group),
            ax=grid.next_cell(),
        )
    return _artifacts(model, [grid], {}, show=False)


def create_event_study_tear_sheet(
    model: FactorAnalysisModel,
    *,
    avgretplot: tuple[int, int] | None = None,
    rate_of_ret: bool = True,
    n_bars: int = 50,
    show: bool = False,
    plotter: Any | None = None,
) -> FactorTearSheetArtifacts:
    """Render event distribution, optional event returns, and quantile-return sections."""

    distribution = _event_distribution_section(model, n_bars=n_bars, plotter=plotter)
    sections = [distribution]
    tables = dict(distribution.tables)

    event = model.event_returns
    if event is not None and avgretplot is not None:
        event_artifacts = create_event_returns_tear_sheet(
            model, std_bar=True, by_group=False, show=False, plotter=plotter
        )
        sections.append(event_artifacts)
        tables.update({f"event_returns.{name}": table for name, table in event_artifacts.tables.items()})

    returns = _event_return_section(model, rate_of_ret=rate_of_ret, plotter=plotter)
    sections.append(returns)
    figures = tuple(figure for section in sections for figure in section.figures)
    return _artifacts_from_figures(model, figures, tables, show=show)


def _event_distribution_section(
    model: FactorAnalysisModel,
    *,
    n_bars: int,
    plotter: Any | None,
    include_returns_tables: bool = True,
) -> FactorTearSheetArtifacts:
    """Render only the event-distribution source section without hidden grids."""

    renderer = _renderer()
    charts = renderer if plotter is None else plotter
    tables = (
        _returns_tables(model)
        if include_returns_tables
        else {"quantile_statistics": renderer.build_quantile_statistics_table(model.factor_data)}
    )
    distribution_grid = GridFigure(1, 1)
    charts.plot_events_distribution(model.factor_data["factor"], num_bars=n_bars, ax=distribution_grid.next_row())
    return _artifacts(model, [distribution_grid], tables, show=False)


def _event_return_section(
    model: FactorAnalysisModel,
    *,
    rate_of_ret: bool,
    plotter: Any | None,
    include_returns_tables: bool = True,
) -> FactorTearSheetArtifacts:
    """Render only event-study's final period-return source section."""

    renderer = _renderer()
    charts = renderer if plotter is None else plotter
    tables = _returns_tables(model) if include_returns_tables else {}
    mean_returns = model.aggregate_mean_returns_by_quantile.copy(deep=True)
    mean_by_date = model.aggregate_mean_returns_by_date.copy(deep=True)
    if rate_of_ret and len(mean_returns.columns):
        mean_returns = _rate_of_return(mean_returns)
        mean_by_date = _rate_of_return(mean_by_date)
    # The legacy event-study source reserves one row per forward period even
    # though its final projection currently fills only the bar/violin rows.
    return_grid = GridFigure(3 + len(model.factor_returns.columns), 1)
    charts.plot_quantile_returns_bar(mean_returns, ax=return_grid.next_row())
    charts.plot_quantile_returns_violin(mean_by_date, ylim_percentiles=(1, 99), ax=return_grid.next_row())
    return _artifacts(model, [return_grid], tables, show=False)


__all__ = [
    "FactorTearSheetArtifacts",
    "GridFigure",
    "close_owned_figures",
    "create_event_returns_tear_sheet",
    "create_event_study_tear_sheet",
    "create_full_tear_sheet",
    "create_information_tear_sheet",
    "create_returns_tear_sheet",
    "create_summary_tear_sheet",
    "create_turnover_tear_sheet",
    "show_owned_figures",
]
