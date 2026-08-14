"""Strict, lazy Alphalens tear-sheet projection backed by model workflows."""

from __future__ import annotations

import importlib
import operator
import warnings
from dataclasses import replace
from functools import wraps
from typing import Any, cast

import pandas as pd

from fincore.alphalens import performance as _strict_performance
from fincore.alphalens import plotting as _plotting
from fincore.alphalens._compat import export_deferred_functions
from fincore.contracts.factor_analysis import ALPHALENS_FUNCTION_SPECS, FactorFunctionSpec
from fincore.factor_analysis.analysis import _analyze_factor_for_strict_turnover, analyze_factor
from fincore.factor_analysis.calendar import get_forward_returns_columns, timedelta_strings_to_integers
from fincore.factor_analysis.tears import (
    GridFigure,
    _event_distribution_section,
    _event_return_section,
    _event_returns_group_section,
    _returns_group_section,
    close_owned_figures,
    show_owned_figures,
)
from fincore.factor_analysis.tears import (
    create_event_returns_tear_sheet as _create_event_returns,
)
from fincore.factor_analysis.tears import (
    create_information_tear_sheet as _create_information,
)
from fincore.factor_analysis.tears import (
    create_returns_tear_sheet as _create_returns,
)
from fincore.factor_analysis.tears import (
    create_summary_tear_sheet as _create_summary,
)
from fincore.factor_analysis.tears import (
    create_turnover_tear_sheet as _create_turnover,
)

_TEARS_NAMES = export_deferred_functions(globals(), "tears")


def _spec(name: str) -> FactorFunctionSpec:
    return ALPHALENS_FUNCTION_SPECS[("tears", name)]


def _deferred(name: str) -> None:
    raise NotImplementedError(
        f"Legacy Alphalens symbol '{name}' is available for C0/C1 compatibility, "
        "but its numerical or rendering kernel is not implemented yet."
    )


def _reject_opaque(name: str, *values: object) -> None:
    """Retain the C1 opaque-input boundary before model construction exists."""

    if any(type(value) is object for value in values):
        _deferred(name)


def _attach_spec(function: Any, name: str) -> Any:
    """Expose the frozen inspection signature while retaining hidden context grammar."""

    spec = _spec(name)
    function.__signature__ = spec.introspection_signature
    function.__fincore_source_signature__ = spec.source_signature
    function.__fincore_factor_spec__ = spec
    return function


def _model(
    factor_data: pd.DataFrame,
    *,
    long_short: bool,
    group_neutral: bool,
    by_group: bool,
    turnover_periods: tuple[int, ...] = (1,),
    event_returns: pd.DataFrame | None = None,
    event_before: object = None,
    event_after: object = None,
    allow_legacy_zero_turnover: bool = False,
    legacy_turnover_quantiles: tuple[int, ...] | None = None,
    preserve_empty_group_analysis: bool = False,
    legacy_source_factor_data: object | None = None,
    duplicate_information_projection: bool = False,
    duplicate_event_projection: bool = False,
) -> Any:
    """Build the one model shared by a strict workflow and its enhanced renderer."""

    normalized_event_before, strict_event_before = _legacy_event_window_bound(event_before)
    normalized_event_after, strict_event_after = _legacy_event_window_bound(event_after)
    needs_legacy_event_windows = strict_event_before or strict_event_after
    assembly_by_group = by_group and (
        preserve_empty_group_analysis or not _legacy_group_rows_empty(factor_data, by_group)
    )

    # The enhanced calendar helpers may emit a modern diagnostic while
    # assembling an irregular-date model.  Source tear sheets expose one
    # workflow-level legacy warning instead, projected below at the actual
    # returns/event-study call boundary.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="'freq' not set, using business day calendar")
        # Model assembly also materializes portfolio snapshots that the legacy
        # tear functions never ask the calendar layer to render.  Pandas emits
        # this implementation-detail warning for those vectorized DateOffset
        # operations; keep only the source-visible workflow warning below.
        warnings.filterwarnings(
            "ignore",
            message="Non-vectorized DateOffset being applied to Series or DatetimeIndex",
            category=pd.errors.PerformanceWarning,
        )
        needs_strict_turnover_bridge = (
            allow_legacy_zero_turnover or legacy_turnover_quantiles is not None or needs_legacy_event_windows
        )
        if needs_strict_turnover_bridge:
            model = _analyze_factor_for_strict_turnover(
                factor_data,
                long_short=long_short,
                group_neutral=group_neutral,
                by_group=assembly_by_group,
                turnover_periods=turnover_periods,
                include_pyfolio=False,
                event_returns=event_returns,
                event_before=normalized_event_before,
                event_after=normalized_event_after,
                allow_legacy_zero_turnover=allow_legacy_zero_turnover,
                legacy_turnover_quantiles=legacy_turnover_quantiles,
                allow_legacy_event_windows=needs_legacy_event_windows,
            )
        else:
            model = analyze_factor(
                factor_data,
                long_short=long_short,
                group_neutral=group_neutral,
                by_group=assembly_by_group,
                turnover_periods=turnover_periods,
                include_pyfolio=False,
                event_returns=event_returns,
                event_before=cast("int | None", normalized_event_before),
                event_after=cast("int | None", normalized_event_after),
            )

    # Keep source-only duplicate-column expansion in the one strict assembly
    # boundary.  The renderers receive a completed model and never invoke an
    # analytical kernel after this function returns.
    source_factor_data = factor_data if legacy_source_factor_data is None else legacy_source_factor_data
    if duplicate_information_projection:
        model = _strict_duplicate_information_model(
            model,
            source_factor_data,
            group_neutral=group_neutral,
            by_group=by_group,
        )
    if duplicate_event_projection:
        model = _strict_duplicate_event_model(model, source_factor_data)
    if isinstance(source_factor_data, pd.DataFrame) and not _has_forward_return_columns(source_factor_data):
        model = replace(model, factor_data=source_factor_data)
    return model


def _legacy_event_window_bound(value: object) -> tuple[object, bool]:
    """Normalize source-accepted integral event offsets only for strict calls."""

    if value is None:
        return None, False
    try:
        normalized = operator.index(cast("Any", value))
    except TypeError:
        return value, False
    return normalized, not isinstance(value, int) or isinstance(value, bool) or normalized < 0


def _legacy_frequency_warning(factor_data: object) -> None:
    """Replay the source returns/event-study warning for an unset date frequency."""

    if not isinstance(factor_data, pd.DataFrame) or not isinstance(factor_data.index, pd.MultiIndex):
        return
    dates = factor_data.index.levels[0]
    if isinstance(dates, pd.DatetimeIndex) and dates.freq is None:
        warnings.warn("'freq' not set in factor_data index: assuming business day", UserWarning, stacklevel=2)


def _legacy_require_group_for_by_group(factor_data: object, by_group: object) -> None:
    """Preserve the source's direct ``KeyError('group')`` validation path."""

    if by_group:
        # Do not manufacture a friendlier validation error: the pinned
        # workflow reaches pandas' column lookup before it can render a group
        # section, and that exception is part of the strict surface.
        cast("pd.DataFrame", factor_data)["group"]


def _legacy_group_rows_empty(factor_data: object, by_group: object) -> bool:
    """Identify the plain-null source branch with no categorical group universe."""

    if not (bool(by_group) and isinstance(factor_data, pd.DataFrame) and "group" in factor_data.columns):
        return False
    groups = factor_data["group"]
    # ``groupby(observed=False)`` retains unused categorical levels.  The
    # pinned returns/information paths therefore construct all-NaN rows for
    # such categories rather than following the plain empty-group GridSpec
    # branch.  Preserve that distinction through model assembly.
    return groups.dropna().empty and not isinstance(groups.dtype, pd.CategoricalDtype)


def _legacy_empty_returns_group_grid_error() -> None:
    """Replay the pinned zero-row return-group ``GridFigure`` failure.

    In the returns path the source derives zero observed group panels, then
    still instantiates ``GridFigure(rows=0, cols=2)``.  Its Figure is created
    first and Matplotlib's GridSpec raises afterwards; the enhanced grid
    validates earlier, so the strict facade performs this tiny source-shaped
    projection only for the legacy exceptional path.
    """

    pyplot = importlib.import_module("matplotlib.pyplot")
    gridspec = importlib.import_module("matplotlib.gridspec")
    pyplot.figure(figsize=(14, 0))
    gridspec.GridSpec(0, 2, wspace=0.4, hspace=0.3)


def _legacy_turnover_periods(factor_data: object, turnover_periods: object) -> tuple[int, ...]:
    """Mirror the source's day-lag normalization before model assembly."""

    if turnover_periods is not None:
        values = timedelta_strings_to_integers(cast("list[str]", turnover_periods))
        return tuple(values)
    # Preserve the source's direct ``factor_data.columns`` access for the
    # default branch.  In particular, ``None`` raises ``AttributeError`` here
    # before the later factor-quantile lookup can produce a different error.
    labels = get_forward_returns_columns(cast("Any", factor_data).columns, require_exact_day_multiple=True)
    values = timedelta_strings_to_integers(cast("list[str]", labels.tolist()))
    return tuple(values)


def _summary_turnover_periods(factor_data: object) -> tuple[int, ...]:
    """Match summary's permissive forward-period discovery without a warning."""

    if not isinstance(factor_data, pd.DataFrame):
        return (1,)
    labels = get_forward_returns_columns(factor_data.columns)
    # Source turns every timedelta into an integer day count before building
    # its turnover dict.  Its dict silently collapses duplicate whole-day
    # values, so do the same before handing a model its unique lag contract.
    values = tuple(dict.fromkeys(int(pd.Timedelta(label).days) for label in labels))
    return values or (1,)


def _summary_turnover_quantiles(factor_data: object) -> tuple[int, ...] | None:
    """Return source summary's contiguous ``1..max`` turnover-bin universe."""

    if not isinstance(factor_data, pd.DataFrame) or "factor_quantile" not in factor_data.columns:
        return None
    maximum = factor_data["factor_quantile"].max()
    if pd.isna(maximum):
        return ()
    return tuple(range(1, int(maximum) + 1))


def _legacy_reject_duplicate_forward_columns(factor_data: object) -> None:
    """Replay the source returns assignment failure for duplicate periods.

    The enhanced model deliberately validates selected forward labels early.
    Pinned full tear sheets instead display quantile statistics, enter the
    returns path, and let pandas reject assigning a duplicate-column result.
    Keep that strict-only timing and public error surface at the facade.
    """

    if not isinstance(factor_data, pd.DataFrame):
        return
    forward_columns = get_forward_returns_columns(factor_data.columns)
    if forward_columns.has_duplicates:
        raise ValueError("Columns must be same length as key")


def _has_duplicate_forward_columns(factor_data: object) -> bool:
    """Return whether a factor table carries repeated forward-return labels."""

    return isinstance(factor_data, pd.DataFrame) and get_forward_returns_columns(factor_data.columns).has_duplicates


def _deduplicated_forward_model_input(factor_data: object) -> object:
    """Give private strict model assembly one representative per forward label.

    The enhanced analysis model deliberately rejects duplicate selected periods.
    Several pinned tear workflows, however, either use only factor/quantile
    fields or expand duplicate returns at their own public boundary.  Their
    strict projections therefore assemble a private unique-label model and
    restore source-visible duplicate fields where needed.
    """

    if not _has_duplicate_forward_columns(factor_data):
        return factor_data
    assert isinstance(factor_data, pd.DataFrame)  # narrowed by the helper above
    forward_labels = set(get_forward_returns_columns(factor_data.columns).tolist())
    seen: set[object] = set()
    positions: list[int] = []
    for position, label in enumerate(factor_data.columns):
        if label not in forward_labels or label not in seen:
            positions.append(position)
            seen.add(label)
    return factor_data.iloc[:, positions].copy(deep=True)


def _has_forward_return_columns(factor_data: object) -> bool:
    """Return whether a source-shaped factor table contains any forward label."""

    return isinstance(factor_data, pd.DataFrame) and bool(len(get_forward_returns_columns(factor_data.columns)))


def _strict_event_model_input(factor_data: object) -> object:
    """Supply a private placeholder return only for source event-only paths.

    Pinned event returns are driven by a separate ``returns`` price frame and
    can render without factor forward-return columns.  The enhanced model
    deliberately requires one because its general-purpose snapshots include
    portfolio fields.  A zero placeholder keeps that private assembly valid;
    strict event projections never expose it as source data.
    """

    model_input = _deduplicated_forward_model_input(factor_data)
    if isinstance(model_input, pd.DataFrame) and not _has_forward_return_columns(model_input):
        model_input = model_input.copy(deep=True)
        model_input["1D"] = 0.0
    return model_input


def _strict_event_returns_fail_after_distribution(
    factor_data: object,
    returns: object,
    *,
    none_is_failure: bool = False,
) -> bool:
    """Identify source event inputs that fail only after the first section.

    The legacy event-study renders quantile statistics and its event
    distribution before asking the optional average-return machinery to align
    the price frame.  The enhanced one-shot model validates that input too
    early, so keep the small set of deterministic no-window cases out of its
    event assembly and replay their source failure afterwards.
    """

    if returns is None:
        return none_is_failure
    if not isinstance(returns, pd.DataFrame):
        return True
    if returns.empty:
        return True
    if not isinstance(factor_data, pd.DataFrame) or not isinstance(factor_data.index, pd.MultiIndex):
        return False
    try:
        assets = factor_data.index.get_level_values("asset")
    except (KeyError, IndexError):
        return False
    if not set(assets).issubset(returns.columns):
        return True
    dates = factor_data.index.get_level_values("date")
    if not bool(pd.Index(dates).isin(returns.index).any()):
        return True
    duplicate_dates = returns.index[returns.index.duplicated(keep=False)]
    return bool(pd.Index(dates).isin(duplicate_dates).any())


def _legacy_event_returns_failure(
    factor_data: object,
    returns: object,
    *,
    before: object,
    after: object,
) -> None:
    """Replay the source's late invalid-price-frame exceptions."""

    if isinstance(returns, pd.Series):
        # The pinned source reaches ``pd.concat(all_returns)`` after no
        # DataFrame event slice was collected for a plain Series.
        pd.concat([])
    if isinstance(returns, pd.DataFrame) and isinstance(factor_data, pd.DataFrame):
        # Recreate only the source's event-slice grammar, after the first
        # distribution section has already been displayed.  In particular do
        # not use the enhanced validation layer: duplicate dates must retain
        # its native ``slice - int`` TypeError and missing assets must retain
        # pandas' own ``.loc`` KeyError.
        factor_index = cast("pd.MultiIndex", factor_data.index)
        date_values = factor_index.get_level_values("date")
        asset_values = factor_index.get_level_values("asset")
        all_returns: list[pd.Series] = []
        for timestamp in pd.Index(date_values).unique().sort_values():
            try:
                day_zero: Any = returns.index.get_loc(timestamp)
            except KeyError:
                continue
            source_before: Any = before
            source_after: Any = after
            start: Any = max(day_zero - source_before, 0)
            stop: Any = min(day_zero + source_after + 1, len(returns.index))
            equities = list(set(asset_values[date_values == timestamp]))
            series = cast("pd.DataFrame", returns.loc[returns.index[start:stop], equities].copy())
            series.index = pd.RangeIndex(start - day_zero, stop - day_zero)
            all_returns.append(series.mean(axis=1))
        pd.concat(all_returns, axis=1)
        return
    # A string/list follows the source's direct ``returns.index.get_loc``
    # attribute path instead of enhanced type validation.
    source_get_loc = cast("Any", returns).index.get_loc
    del source_get_loc


def _strict_duplicate_information_model(
    model: Any,
    factor_data: object,
    *,
    group_neutral: bool,
    by_group: bool,
) -> Any:
    """Restore pinned duplicate-period IC columns on a private strict model."""

    if not _has_duplicate_forward_columns(factor_data):
        return model
    assert isinstance(factor_data, pd.DataFrame)
    information = _strict_performance.factor_information_coefficient(
        factor_data,
        group_adjust=group_neutral,
        by_group=by_group,
    )
    aggregate = (
        _strict_performance.factor_information_coefficient(
            factor_data,
            group_adjust=group_neutral,
            by_group=False,
        )
        if by_group
        else information
    )
    mean_information: pd.Series | pd.DataFrame
    if by_group:
        mean_information = information.groupby(level="group", observed=False, sort=True).mean()
    else:
        mean_information = information.mean()
    aggregate_mean = aggregate.mean()
    aggregate_time: dict[str, pd.DataFrame] = {}
    for frequency in model.config.time_aggregation:
        aggregate_time[frequency] = aggregate.resample("ME" if frequency == "M" else frequency).mean()
    return replace(
        model,
        factor_data=factor_data,
        forward_periods=tuple(cast("str", period) for period in information.columns),
        information_coefficient=information,
        mean_information_coefficient=mean_information,
        aggregate_information_coefficient=aggregate,
        aggregate_mean_information_coefficient=aggregate_mean,
        summary_information_coefficient=aggregate if not group_neutral else model.summary_information_coefficient,
        time_aggregated_results=aggregate_time,
        aggregate_time_aggregated_results=aggregate_time,
    )


def _strict_duplicate_event_model(model: Any, factor_data: object) -> Any:
    """Restore expanded source return columns for event-study grid sizing."""

    if not _has_duplicate_forward_columns(factor_data):
        return model
    assert isinstance(factor_data, pd.DataFrame)
    strict_returns = _strict_performance.factor_returns(
        factor_data,
        demeaned=model.config.long_short,
        group_adjust=model.config.group_neutral,
        equal_weight=model.config.equal_weight,
    )
    changes: dict[str, object] = {
        "factor_data": factor_data,
        "forward_periods": tuple(cast("str", period) for period in strict_returns.columns),
        "factor_returns": strict_returns,
    }
    # The strict event-study façade uses ``long_short=False``.  In that path
    # the pinned mean-return kernels preserve all duplicate columns, so retain
    # their bar/violin content in addition to the source-sized final grid.
    if not model.config.long_short:
        from fincore.factor_analysis import performance as enhanced_performance

        mean_returns, std_error = enhanced_performance.mean_return_by_quantile(factor_data, demeaned=False)
        mean_by_date, std_error_by_date = enhanced_performance.mean_return_by_quantile(
            factor_data,
            by_date=True,
            demeaned=False,
        )
        changes.update(
            mean_returns_by_quantile=mean_returns,
            std_error_by_quantile=std_error,
            mean_returns_by_date=mean_by_date,
            std_error_by_date=std_error_by_date,
            aggregate_mean_returns_by_quantile=mean_returns,
            aggregate_std_error_by_quantile=std_error,
            aggregate_mean_returns_by_date=mean_by_date,
            aggregate_std_error_by_date=std_error_by_date,
        )
    return replace(model, **changes)


def _legacy_close(artifacts: Any) -> None:
    """Legacy workflows display every owned section, then close only those figures."""

    close_owned_figures(artifacts)


def _legacy_show_and_close(artifacts: Any) -> None:
    """Display one rendered source section while only that section is open."""

    show_owned_figures(artifacts)
    _legacy_close(artifacts)


def _strict_returns_sections(model: Any, *, by_group: bool, factor_data: object | None = None) -> None:
    """Render the primary and optional group sections in source show/close order."""

    primary = _create_returns(
        model,
        by_group=False,
        show=False,
        legacy_projection=True,
        plotter=_plotting,
    )
    _legacy_display_tables(primary, "returns")
    if factor_data is not None:
        _legacy_frequency_warning(factor_data)
    _legacy_show_and_close(primary)
    if by_group:
        if factor_data is not None:
            _legacy_require_group_for_by_group(factor_data, by_group)
            if _legacy_group_rows_empty(factor_data, by_group):
                _legacy_empty_returns_group_grid_error()
        grouped = _returns_group_section(
            model,
            legacy_projection=True,
            plotter=_plotting,
        )
        _legacy_show_and_close(grouped)


def _strict_event_return_sections(
    model: Any,
    *,
    std_bar: bool,
    by_group: bool,
    factor_data: object | None = None,
    returns: object | None = None,
    before: object = None,
    after: object = None,
    long_short: bool = True,
    group_neutral: bool = False,
) -> None:
    """Render source event sections one Figure lifecycle at a time."""

    primary = _create_event_returns(
        model,
        std_bar=std_bar,
        by_group=False,
        show=False,
        plotter=_plotting,
        # Pinned event-return tear sheets do not build the unrelated regular
        # returns table.  Avoiding it also keeps duplicate forward labels on
        # their source event-only path.
        _include_returns_tables=False,
    )
    _legacy_show_and_close(primary)
    if by_group:
        if factor_data is not None:
            _legacy_require_group_for_by_group(factor_data, by_group)
            if _legacy_group_rows_empty(factor_data, by_group):
                pd.concat([])
        if factor_data is not None and returns is not None:
            _replay_legacy_event_stdout(
                factor_data,
                returns,
                before=before,
                after=after,
                long_short=long_short,
                group_neutral=group_neutral,
                by_group=True,
            )
        grouped = _event_returns_group_section(model, plotter=_plotting)
        _legacy_show_and_close(grouped)


def _replay_legacy_event_stdout(
    factor_data: object,
    returns: object,
    *,
    before: object,
    after: object,
    long_short: bool,
    group_neutral: bool,
    by_group: bool,
) -> None:
    """Replay the source event-window ``print('series = ', ...)`` side effect.

    The enhanced model deliberately computes its event snapshots silently.  A
    strict tear sheet nevertheless has to retain the CloudQuant source's
    visible diagnostic, including its per-quantile/group cadence.  This helper
    only recreates the already-computed index/column slices for printing; it
    does not invoke an analytical kernel or replace any model snapshot.
    """

    if not isinstance(factor_data, pd.DataFrame) or not isinstance(returns, pd.DataFrame):
        return
    before_value, _ = _legacy_event_window_bound(before)
    after_value, _ = _legacy_event_window_bound(after)
    if not isinstance(before_value, int) or not isinstance(after_value, int):
        return

    def replay_quantiles(data: pd.DataFrame, demean_by: pd.Series | None) -> None:
        quantiles = data["factor_quantile"]
        for _, quantile_data in quantiles.groupby(quantiles):
            _strict_performance._strict_print_common_start_return_slices(
                quantile_data,
                returns,
                before_value,
                after_value,
                cumulative=True,
                mean_by_date=True,
                demean_by=demean_by,
            )

    if by_group:
        for _, group_data in factor_data.groupby("group"):
            group_quantiles = group_data["factor_quantile"]
            replay_quantiles(
                group_data,
                group_quantiles if group_neutral else factor_data["factor_quantile"] if long_short else None,
            )
        return
    if group_neutral:
        for _, group_data in factor_data.groupby("group"):
            replay_quantiles(group_data, group_data["factor_quantile"])
        return
    replay_quantiles(factor_data, factor_data["factor_quantile"] if long_short else None)


def _legacy_display_tables(artifacts: Any, workflow: str) -> None:
    """Replay the source workflow's notebook-table side effects from snapshots.

    The enhanced workflow has already built these tables from the model, so
    this projection deliberately uses its frozen values instead of invoking a
    kernel a second time.
    """

    tables = artifacts.tables

    def display(
        key: str,
        heading: str | None,
        *,
        round_values: bool = True,
        transpose: bool = False,
    ) -> None:
        table = tables.get(key)
        if table is not None:
            if transpose:
                table = table.T
            _plotting._display_table(heading, table, round_values=round_values)

    if workflow in {"summary", "full", "event_study"}:
        display("quantile_statistics", "Quantiles Statistics", round_values=False)
    if workflow == "summary":
        display("returns", "Returns Analysis")
        display("information", "Information Analysis", transpose=True)
        display("turnover", "Turnover Analysis")
        display("autocorrelation", None)
    elif workflow == "returns":
        display("returns", "Returns Analysis")
    elif workflow == "information":
        display("information", "Information Analysis", transpose=True)
    elif workflow == "turnover":
        display("turnover", "Turnover Analysis")
        display("autocorrelation", None)
    elif workflow == "full":
        display("returns.returns", "Returns Analysis")
        display("information.information", "Information Analysis", transpose=True)
        display("turnover.turnover", "Turnover Analysis")
        display("turnover.autocorrelation", None)


def _legacy_display_turnover_tables(model: Any) -> None:
    """Display turnover tables without materializing a chart grid."""

    renderer = _plotting._renderer()
    ranks = {period: model.rank_autocorrelation[period] for period in model.rank_autocorrelation}
    turnover, autocorrelation = renderer.build_turnover_tables(ranks, model.quantile_turnover)
    _plotting._display_table("Turnover Analysis", turnover)
    _plotting._display_table(None, autocorrelation)


def _legacy_duplicate_turnover_period_error(model: Any, source_lags: tuple[int, ...]) -> None:
    """Replay source's duplicate-rank-column truth-value failure after its table."""

    grid = GridFigure(6 * len(source_lags), 1)
    try:
        # Source renders every top/bottom turnover row before it indexes the
        # duplicate-labelled autocorrelation table.  The latter lookup is what
        # raises, so these chart calls are observable even though no Figure is
        # shown afterwards.
        for period in source_lags:
            turnover = model.quantile_turnover[period]
            if not turnover.isna().all().all():
                _plotting.plot_top_bottom_quantile_turnover(turnover, period=period, ax=grid.next_row())
        autocorrelation = pd.concat(
            [model.rank_autocorrelation[period] for period in source_lags],
            axis=1,
        )
        # Source indexes its duplicate-labelled concat by one integer, which
        # yields a DataFrame.  Its boolean check raises pandas' exact public
        # ambiguity error before any chart or ``show`` call.
        if autocorrelation[source_lags[0]].isnull().all():
            return
    finally:
        grid.close()


def _close_legacy_context_figures(function: Any) -> Any:
    """Close figures introduced by the legacy seaborn context, not caller figures.

    ``seaborn.despine`` may allocate an otherwise unreachable current Figure
    before the workflow owns its first grid.  The source workflows are
    self-contained, so preserve caller figures while releasing that context
    artifact together with the grid figures closed by their function bodies.
    """

    @wraps(function)
    def call(*args: Any, **kwargs: Any) -> Any:
        pyplot = importlib.import_module("matplotlib.pyplot")
        before = set(pyplot.get_fignums())
        try:
            return function(*args, **kwargs)
        finally:
            for number in set(pyplot.get_fignums()) - before:
                pyplot.close(number)

    return call


@_close_legacy_context_figures
@_plotting.customize
def create_summary_tear_sheet(factor_data, long_short=True, group_neutral=False):
    """Render the pinned summary workflow from a single compute-once model."""

    _reject_opaque("create_summary_tear_sheet", factor_data)
    _legacy_reject_duplicate_forward_columns(factor_data)
    lags = _summary_turnover_periods(factor_data)
    artifacts = _create_summary(
        _model(
            factor_data,
            long_short=long_short,
            group_neutral=group_neutral,
            by_group=False,
            turnover_periods=lags,
            allow_legacy_zero_turnover=any(lag <= 0 for lag in lags),
            legacy_turnover_quantiles=_summary_turnover_quantiles(factor_data),
        ),
        show=False,
        legacy_projection=True,
        plotter=_plotting,
    )
    _legacy_display_tables(artifacts, "summary")
    _legacy_show_and_close(artifacts)
    return


@_close_legacy_context_figures
@_plotting.customize
def create_returns_tear_sheet(factor_data, long_short=True, group_neutral=False, by_group=False):
    """Render the pinned returns workflow and preserve its show/close lifecycle."""

    _reject_opaque("create_returns_tear_sheet", factor_data)
    _legacy_reject_duplicate_forward_columns(factor_data)
    _strict_returns_sections(
        _model(factor_data, long_short=long_short, group_neutral=group_neutral, by_group=by_group),
        by_group=by_group,
        factor_data=factor_data,
    )
    return


@_close_legacy_context_figures
@_plotting.customize
def create_information_tear_sheet(factor_data, group_neutral=False, by_group=False):
    """Render the pinned information workflow using the same model snapshot."""

    _reject_opaque("create_information_tear_sheet", factor_data)
    model = _model(
        _deduplicated_forward_model_input(factor_data),
        long_short=True,
        group_neutral=group_neutral,
        by_group=by_group,
        preserve_empty_group_analysis=True,
        legacy_source_factor_data=factor_data,
        duplicate_information_projection=True,
    )
    artifacts = _create_information(
        model,
        by_group=by_group,
        show=False,
        legacy_projection=True,
        plotter=_plotting,
    )
    _legacy_display_tables(artifacts, "information")
    _legacy_require_group_for_by_group(factor_data, by_group)
    _legacy_show_and_close(artifacts)
    return


@_close_legacy_context_figures
@_plotting.customize
def create_turnover_tear_sheet(factor_data, turnover_periods=None):
    """Render source-shaped turnover periods from their stored model results."""

    _reject_opaque("create_turnover_tear_sheet", factor_data)
    source_lags = _legacy_turnover_periods(factor_data, turnover_periods)
    # The source discovers default forward labels first, then looks up
    # quantiles before an explicitly empty period list reaches ``pd.concat``.
    # That order distinguishes ``None``/default (``.columns`` AttributeError)
    # from ``None``/explicit-empty (subscript TypeError).
    _ = factor_data["factor_quantile"]
    if not source_lags:
        # Pinned code reaches ``pd.concat([])`` after its empty turnover dict;
        # do not invent a one-day model merely to make the enhanced API happy.
        pd.concat([])
    # The pinned source uses a dict for the per-period turnover tables (so
    # duplicate lag keys collapse) but still iterates the original sequence
    # for rank-autocorrelation charts.  Keep the model's mapping keys valid
    # while giving the renderer that source sequence below.
    lags = tuple(dict.fromkeys(source_lags))
    model = _model(
        _deduplicated_forward_model_input(factor_data),
        long_short=True,
        group_neutral=False,
        by_group=False,
        turnover_periods=lags,
        allow_legacy_zero_turnover=any(lag <= 0 for lag in source_lags),
    )
    if len(lags) != len(source_lags):
        _legacy_display_turnover_tables(model)
        _legacy_duplicate_turnover_period_error(model, source_lags)
        return
    artifacts = _create_turnover(
        model,
        turnover_periods=source_lags,
        show=False,
        legacy_projection=True,
        plotter=_plotting,
    )
    _legacy_display_tables(artifacts, "turnover")
    _legacy_show_and_close(artifacts)
    return


@_close_legacy_context_figures
@_plotting.customize
def create_full_tear_sheet(factor_data, long_short=True, group_neutral=False, by_group=False):
    """Compose the three legacy sections from one shared model snapshot."""

    _reject_opaque("create_full_tear_sheet", factor_data)
    # Pinned ``create_full_tear_sheet`` displays its factor-quantile
    # statistics before it delegates to returns/period discovery.  Keeping
    # this tiny table projection ahead of model assembly preserves that
    # notebook-visible side effect even when later forward-column validation
    # fails.
    quantile_statistics = _plotting._renderer().build_quantile_statistics_table(factor_data)
    _plotting._display_table(
        "Quantiles Statistics",
        quantile_statistics,
        round_values=False,
    )
    _legacy_reject_duplicate_forward_columns(factor_data)
    source_lags = _legacy_turnover_periods(factor_data, None)
    lags = tuple(dict.fromkeys(source_lags)) or (1,)
    model = _model(
        _deduplicated_forward_model_input(factor_data),
        long_short=long_short,
        group_neutral=group_neutral,
        by_group=by_group,
        turnover_periods=lags,
        allow_legacy_zero_turnover=any(lag <= 0 for lag in source_lags),
    )
    # The source full sheet delegates its children sequentially.  Each helper
    # consumes the same model and releases its figure before the next one is
    # created, preserving the observable ``plt.show`` open-figure set.
    _strict_returns_sections(model, by_group=by_group, factor_data=factor_data)
    information = _create_information(model, by_group=by_group, show=False, legacy_projection=True, plotter=_plotting)
    _legacy_display_tables(information, "information")
    _legacy_show_and_close(information)
    if not source_lags:
        pd.concat([])
    if len(source_lags) != len(lags):
        _legacy_display_turnover_tables(model)
        _legacy_duplicate_turnover_period_error(model, source_lags)
        return
    turnover = _create_turnover(
        model,
        turnover_periods=source_lags,
        show=False,
        legacy_projection=True,
        plotter=_plotting,
    )
    _legacy_display_tables(turnover, "turnover")
    _legacy_show_and_close(turnover)
    return


@_close_legacy_context_figures
@_plotting.customize
def create_event_returns_tear_sheet(
    factor_data,
    returns,
    avgretplot=(5, 15),
    long_short=True,
    group_neutral=False,
    std_bar=True,
    by_group=False,
):
    """Render source event-return sections from one model-bound event window."""

    _reject_opaque("create_event_returns_tear_sheet", factor_data, returns)
    before, after = avgretplot
    if _strict_event_returns_fail_after_distribution(factor_data, returns, none_is_failure=True):
        _legacy_event_returns_failure(
            factor_data,
            returns,
            before=before,
            after=after,
        )
    _replay_legacy_event_stdout(
        factor_data,
        returns,
        before=before,
        after=after,
        long_short=long_short,
        group_neutral=group_neutral,
        by_group=False,
    )
    model = _model(
        _strict_event_model_input(factor_data),
        long_short=long_short,
        group_neutral=group_neutral,
        by_group=by_group,
        event_returns=returns,
        event_before=before,
        event_after=after,
        legacy_source_factor_data=factor_data,
        duplicate_event_projection=True,
    )
    _strict_event_return_sections(
        model,
        std_bar=std_bar,
        by_group=by_group,
        factor_data=factor_data,
        returns=returns,
        before=before,
        after=after,
        long_short=long_short,
        group_neutral=group_neutral,
    )
    return


@_close_legacy_context_figures
@_plotting.customize
def create_event_study_tear_sheet(factor_data, returns, avgretplot=(5, 15), rate_of_ret=True, n_bars=50):
    """Render the legacy event-study composition from a typed event model."""

    # Pinned source does not inspect ``returns`` at all when the optional
    # average-return section is disabled.  Retain that accepted-call grammar
    # rather than leaking enhanced event-input validation into strict mode.
    _reject_opaque(
        "create_event_study_tear_sheet",
        factor_data,
        *(returns,) if avgretplot is not None else (),
    )
    before: object = None
    after: object = None
    event_section_enabled = returns is not None and avgretplot is not None
    deferred_avgretplot_error: TypeError | ValueError | None = None
    if event_section_enabled:
        try:
            before, after = cast("Any", avgretplot)
        except (TypeError, ValueError) as error:
            # The source does this unpacking inside its optional event-return
            # child, after the statistics/distribution Figure has been shown.
            # Retain the exact Python error but defer it to that lifecycle
            # position below.
            deferred_avgretplot_error = error
    late_invalid_event_returns = (
        event_section_enabled
        and deferred_avgretplot_error is None
        and (_strict_event_returns_fail_after_distribution(factor_data, returns))
    )
    model = _model(
        _strict_event_model_input(factor_data),
        long_short=False,
        group_neutral=False,
        by_group=False,
        # The pinned event-study source renders its first distribution section
        # before it touches a malformed non-null ``returns`` value.  Build the
        # reusable regular-return model first, then replay that later failure
        # below; valid frames still get their event snapshot in this one
        # assembly pass.
        event_returns=returns if event_section_enabled and not late_invalid_event_returns else None,
        event_before=before,
        event_after=after,
        legacy_source_factor_data=factor_data,
        duplicate_event_projection=True,
    )
    # Display the first distribution Figure, optional event-return Figure,
    # and final return Figure in the same source order.  Build each section
    # directly so no hidden GridFigure is allocated merely to obtain another
    # section from the enhanced composition.
    first = _event_distribution_section(
        model,
        n_bars=n_bars,
        plotter=_plotting,
        include_returns_tables=False,
    )
    _legacy_display_tables(first, "event_study")
    _legacy_show_and_close(first)
    if deferred_avgretplot_error is not None:
        raise deferred_avgretplot_error
    if late_invalid_event_returns:
        _legacy_event_returns_failure(
            factor_data,
            returns,
            before=before,
            after=after,
        )
    if model.event_returns is not None and event_section_enabled:
        _replay_legacy_event_stdout(
            factor_data,
            returns,
            before=before,
            after=after,
            long_short=False,
            group_neutral=False,
            by_group=False,
        )
        _strict_event_return_sections(
            model,
            std_bar=True,
            by_group=False,
            factor_data=factor_data,
            returns=returns,
            before=before,
            after=after,
            long_short=False,
            group_neutral=False,
        )
    if not _has_forward_return_columns(factor_data):
        # Source reaches its final regular-return composition only after the
        # distribution and optional event-return sections.  With no factor
        # forward columns it then hits ``pd.concat([])``.
        pd.concat([])
    final = _event_return_section(
        model,
        rate_of_ret=rate_of_ret,
        plotter=_plotting,
        include_returns_tables=False,
    )
    _legacy_frequency_warning(factor_data)
    _legacy_show_and_close(final)
    return


for _name in _TEARS_NAMES:
    _attach_spec(globals()[_name], _name)


__all__ = ("GridFigure", *_TEARS_NAMES)

del export_deferred_functions
