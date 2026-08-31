"""C4 workflow tests for enhanced and strict Alphalens tear-sheet projections."""

from __future__ import annotations

import importlib
import warnings
from functools import cache, lru_cache
from typing import Any, cast, get_type_hints

import numpy as np
import pandas as pd
import pytest

# Pinned tear-sheet fixtures include constant return paths.  Keep their modern
# SciPy diagnostics out of C4 pass/fail semantics while source-visible strict
# warnings are characterized independently below.
pytestmark = [
    pytest.mark.filterwarnings("ignore:Precision loss occurred in moment calculation:RuntimeWarning"),
    pytest.mark.filterwarnings("ignore:An input array is constant:RuntimeWarning"),
]


def _pyplot() -> object:
    """Resolve pyplot only in an explicitly visual C4 test."""

    return importlib.import_module("matplotlib.pyplot")


@lru_cache(maxsize=4)
def _workflow_model(*, event: bool, by_group: bool) -> object:
    """Cache real compute-once models while each C4 row renders fresh figures."""

    from fincore.factor_analysis.analysis import analyze_factor
    from tests.compat.alphalens.conftest import _shared_clean_factor_data, _shared_inputs

    factor_data = _shared_clean_factor_data().copy(deep=True)
    event_returns = _shared_inputs()[1].copy(deep=True) if event else None
    return analyze_factor(
        factor_data,
        periods=("1D",),
        turnover_periods=(1,),
        by_group=by_group,
        include_portfolio_inputs=False,
        event_returns=event_returns,
        event_before=1 if event else None,
        event_after=2 if event else None,
    )


_EVENT_RETURNS_CASES: dict[int, tuple[int, tuple[int, ...], float | None, str | None]] = {
    0: (2, (1, 5, 10), None, None),
    1: (3, (2, 4, 6), 20.0, None),
    2: (4, (3, 4), None, "US/Eastern"),
    3: (1, (2, 3, 6, 9), 20.0, "US/Eastern"),
}
_EVENT_STUDY_CASES: dict[int, tuple[tuple[int, int], float | None, str | None]] = {
    0: ((6, 8), None, None),
    1: ((6, 8), None, None),
    2: ((6, 3), 20.0, None),
    3: ((6, 3), 20.0, "US/Eastern"),
    4: ((0, 3), None, None),
    5: ((3, 0), 20.0, "US/Eastern"),
}

# Literal ``all_events`` source fixture from the pinned Alphalens tear-sheet
# suite.  Keep the sparse shape (including the zero-valued event) rather than
# deriving a mask from our dense shared factor: event distribution, binning,
# and common-start windows are all sensitive to this exact event topology.
_PINNED_EVENT_ROWS: tuple[tuple[int | None, ...], ...] = (
    (1, None, None, None, None, None),
    (4, None, None, 7, None, None),
    (None, None, None, None, None, None),
    (None, 3, None, 2, None, None),
    (1, None, None, None, None, None),
    (None, None, 2, None, None, None),
    (None, None, None, 2, None, None),
    (None, None, None, 1, None, None),
    (2, None, None, None, None, None),
    (None, None, None, None, 5, None),
    (None, None, None, 2, None, None),
    (None, None, None, None, None, None),
    (2, None, None, None, None, None),
    (None, None, None, None, None, 5),
    (None, None, None, 1, None, None),
    (None, None, None, None, 4, None),
    (5, None, None, 4, None, None),
    (None, None, None, 3, None, None),
    (None, None, None, 4, None, None),
    (None, None, 2, None, None, None),
    (5, None, None, None, None, None),
    (None, 1, None, None, None, None),
    (None, None, None, None, 4, None),
    (0, None, None, None, None, None),
    (None, 5, None, None, None, 4),
    (None, None, None, None, None, None),
    (None, None, 5, None, None, 3),
    (None, None, 1, 2, 3, None),
    (None, None, None, 5, None, None),
    (None, None, 1, None, 3, None),
)


def _pinned_event_study_inputs(
    input_ordinal: int,
    timezone: str | None,
) -> tuple[pd.Series, pd.DataFrame, pd.DatetimeIndex]:
    """Rebuild the pinned daily/BDay ``all_events`` fixture literally."""

    tickers = ("A", "B", "C", "D", "E", "F")
    price_rows = [[1.25**row, 1.50**row, 1.00**row, 0.50**row, 1.50**row, 1.00**row] for row in range(1, 51)]
    if input_ordinal == 0:
        price_dates = pd.date_range("2015-01-10", "2015-02-28", freq="D", name="date")
        factor_dates = pd.date_range("2015-01-15", "2015-02-13", freq="D", name="date")
    else:
        price_dates = pd.date_range("2015-01-10", "2015-03-22", freq="B", name="date")
        factor_dates = pd.date_range("2015-01-15", "2015-02-25", freq="B", name="date")
    prices = pd.DataFrame(price_rows, index=price_dates, columns=tickers)
    events = pd.DataFrame(_PINNED_EVENT_ROWS, index=factor_dates, columns=tickers)
    if timezone is not None:
        prices.index = pd.DatetimeIndex(prices.index.tz_localize(timezone), freq=price_dates.freq, name="date")
        events.index = pd.DatetimeIndex(events.index.tz_localize(timezone), freq=factor_dates.freq, name="date")
    # ``DataFrame.stack()`` in the pinned suite drops non-events.  Pandas 3's
    # ``future_stack=True`` retains them, so make the sparse source contract
    # explicit rather than passing a dense NaN matrix into preparation.
    factor = events.stack(future_stack=True).dropna()
    factor.index = factor.index.set_names(("date", "asset"))
    assert len(factor) == 35
    assert not factor.isna().any()
    return factor, prices, events.index


def _restore_source_calendar_levels(data: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Retain the source factor calendar even when sparse events omit dates."""

    copied = data.copy(deep=True)
    index = copied.index
    assert isinstance(index, pd.MultiIndex)
    observed_dates = pd.DatetimeIndex(index.get_level_values("date"))
    assets = index.get_level_values("asset")
    asset_levels = index.levels[index.names.index("asset")]
    copied.index = pd.MultiIndex(
        levels=(dates, asset_levels),
        codes=(dates.get_indexer(observed_dates), asset_levels.get_indexer(assets)),
        names=("date", "asset"),
        verify_integrity=True,
    )
    return copied


def _source_invocation_parts(source_invocation_id: str) -> tuple[int, int, int]:
    """Parse the literal C4 id back into its source parameter dimensions."""

    prefix, input_part, call_part = source_invocation_id.rsplit("/", 2)
    return (
        int(prefix.rsplit("#", 1)[1]),
        int(input_part.removeprefix("input-")),
        int(call_part.removeprefix("call-")),
    )


@cache
def _source_event_model(
    *,
    study: bool,
    ordinal: int,
    input_ordinal: int,
    long_short: bool,
    group_neutral: bool,
    by_group: bool,
) -> object:
    """Run each pinned event profile through factor cleaning and model assembly."""

    from fincore.factor_analysis.analysis import analyze_factor
    from fincore.factor_analysis.data import prepare_factor_data
    from tests.compat.alphalens.conftest import _pinned_tears_dense_inputs

    if study:
        _, filter_zscore, timezone = _EVENT_STUDY_CASES[ordinal]
        quantiles: int | None = None
        bins: int | None = 1
        periods = (1, 2)
        factor, price_frame, source_factor_dates = _pinned_event_study_inputs(input_ordinal, timezone)
        groups = None
    else:
        quantiles, periods, filter_zscore, timezone = _EVENT_RETURNS_CASES[ordinal]
        bins = None
        # Preserve the exact dense ``all_factors`` / ``all_prices`` source
        # fixture rather than relabeling the generic 2024 compatibility data.
        factor, price_frame, groups, source_factor_dates = _pinned_tears_dense_inputs(input_ordinal, timezone)
        factor = factor.copy(deep=True)
        price_frame = price_frame.copy(deep=True)
    prepared = prepare_factor_data(
        factor,
        price_frame,
        groupby=None if study else groups,
        quantiles=quantiles,
        bins=bins,
        periods=periods,
        filter_zscore=filter_zscore,
        max_loss=1.0,
    )
    before, after = _EVENT_STUDY_CASES[ordinal][0] if study else (5, 11)
    prepared_data = _restore_source_calendar_levels(prepared.data, source_factor_dates)
    # The pinned literal price fixture intentionally contains constant asset
    # paths.  SciPy emits a ConstantInputWarning while model assembly records
    # IC snapshots; upstream's own tests treat that data condition as a valid
    # render profile, whereas this repository promotes RuntimeWarning-derived
    # diagnostics to errors globally.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="An input array is constant; the correlation coefficient is not defined"
        )
        return analyze_factor(
            prepared_data,
            long_short=long_short,
            group_neutral=group_neutral,
            by_group=by_group,
            turnover_periods=periods,
            include_portfolio_inputs=False,
            event_returns=price_frame,
            event_before=before,
            event_after=after,
        )


def _event_model_for_invocation(source_invocation_id: str, *, study: bool) -> tuple[object, dict[str, object]]:
    """Bind every source invocation to its source options before rendering it."""

    ordinal, input_ordinal, call_ordinal = _source_invocation_parts(source_invocation_id)
    long_short = False
    group_neutral = False
    by_group = False
    if not study:
        long_short, group_neutral, by_group = (
            (False, False, False),
            (True, False, False),
            (False, True, False),
            (False, False, True),
            (True, False, True),
            (False, True, True),
        )[call_ordinal]
    model = _source_event_model(
        study=study,
        ordinal=ordinal,
        input_ordinal=input_ordinal,
        long_short=long_short,
        group_neutral=group_neutral,
        by_group=by_group,
    )
    avgretplot = _EVENT_STUDY_CASES[ordinal][0] if study else (5, 11)
    return model, {
        "avgretplot": avgretplot,
        "by_group": by_group,
        "calendar": "D" if input_ordinal == 0 else "B",
    }


def assert_figure_artifacts(
    artifacts: object,
    *,
    expected_figures: int,
    required_tables: frozenset[str],
) -> None:
    """Bind each frozen source row to fixed figure/table sections."""

    figures = artifacts.figures
    assert len(figures) == expected_figures
    assert all(figure.axes for figure in figures)
    assert required_tables.issubset(artifacts.tables)


def assert_show_called(calls: list[object], artifacts: object, *, expected_show: int) -> None:
    """Pin the source lifecycle rather than deriving it from artifacts."""

    del artifacts
    assert len(calls) == expected_show


def assert_event_snapshot_values(artifacts: object, model: object) -> None:
    """Compare C4 event tables with the stored aggregate event snapshots."""

    event = model.event_returns
    assert event is not None
    if "event_average" in artifacts.tables:
        pd.testing.assert_frame_equal(artifacts.tables["event_average"], event.aggregate_quantile_average_returns)
    if "event_windows" in artifacts.tables:
        pd.testing.assert_frame_equal(artifacts.tables["event_windows"], event.event_windows)
    if "event_returns.event_average" in artifacts.tables:
        pd.testing.assert_frame_equal(
            artifacts.tables["event_returns.event_average"], event.aggregate_quantile_average_returns
        )
    if "event_returns.event_windows" in artifacts.tables:
        pd.testing.assert_frame_equal(artifacts.tables["event_returns.event_windows"], event.event_windows)


def assert_strict_event_source_invocation(
    source_invocation_id: str,
    model: object,
    options: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
    *,
    study: bool,
) -> None:
    """Run every frozen event profile through the strict public workflow too."""

    from fincore.alphalens import tears as strict_tears

    returns = model.event_input_snapshot
    assert returns is not None
    pyplot = _pyplot()
    calls: list[object] = []
    monkeypatch.setattr(pyplot, "show", lambda *args, **kwargs: calls.append((args, kwargs)))
    monkeypatch.setattr(strict_tears._plotting, "_display_table", lambda *args, **kwargs: None)
    with warnings.catch_warnings():
        # The pinned literal price fixture contains two constant paths.  Its
        # valid sparse event profile reaches SciPy IC bookkeeping inside the
        # model assembly, which emits this third-party RuntimeWarning under
        # our modern SciPy; it is not a tear-sheet failure condition.
        warnings.filterwarnings(
            "ignore", message="An input array is constant; the correlation coefficient is not defined"
        )
        if study:
            result = strict_tears.create_event_study_tear_sheet(
                model.factor_data,
                returns,
                avgretplot=cast("tuple[int, int]", options["avgretplot"]),
                n_bars=50,
                set_context=False,
            )
            expected_show = 3
        else:
            by_group = cast("bool", options["by_group"])
            result = strict_tears.create_event_returns_tear_sheet(
                model.factor_data,
                returns,
                avgretplot=cast("tuple[int, int]", options["avgretplot"]),
                long_short=model.config.long_short,
                group_neutral=model.config.group_neutral,
                by_group=by_group,
                set_context=False,
            )
            expected_show = 2 if by_group else 1
    assert result is None
    assert len(calls) == expected_show
    assert not pyplot.get_fignums()


def assert_event_source_calendar(model: object, options: dict[str, object]) -> None:
    """Keep every event invocation ID bound to the pinned input calendar."""

    dates = model.factor_data.index.levels[0]
    assert isinstance(dates, pd.DatetimeIndex)
    assert dates.freqstr == options["calendar"]
    assert len(dates) == 30
    assert set(model.factor_data.index.get_level_values("asset")) == {"A", "B", "C", "D", "E", "F"}


def assert_artifact_ownership(artifacts: object) -> None:
    """C4 helper: figures and tables have explicit workflow ownership."""

    assert artifacts.tables
    assert artifacts.model is not None


def assert_no_open_figures(artifacts: object) -> None:
    """C4 helper: cleanup releases only returned Figure artifacts."""

    del artifacts
    assert not _pyplot().get_fignums()


@pytest.fixture(autouse=True)
def _close_figures_after_each_workflow_test() -> None:
    """Keep C4 ownership assertions independent of a preceding failed test."""

    pyplot = _pyplot()
    pyplot.close("all")
    yield
    pyplot.close("all")


def test_enhanced_summary_tear_sheet_consumes_one_model_and_leaves_figures_to_the_caller(
    clean_factor_data: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The enhanced workflow is model-driven, artifact-returning, and non-showing by default."""

    from fincore.factor_analysis.analysis import analyze_factor
    from fincore.factor_analysis.tears import FactorTearSheetArtifacts, create_summary_tear_sheet

    model = analyze_factor(clean_factor_data, periods=("1D",), turnover_periods=(1,), include_portfolio_inputs=False)
    pyplot = _pyplot()
    shown: list[object] = []
    monkeypatch.setattr(pyplot, "show", lambda *args, **kwargs: shown.append((args, kwargs)))

    artifacts = create_summary_tear_sheet(model)

    assert isinstance(artifacts, FactorTearSheetArtifacts)
    assert "figures" in get_type_hints(FactorTearSheetArtifacts)
    assert artifacts.model.result_fingerprint == model.result_fingerprint
    assert artifacts.figures
    renderer = importlib.import_module("fincore.factor_analysis.render_matplotlib")
    pd.testing.assert_frame_equal(
        artifacts.tables["quantile_statistics"], renderer.build_quantile_statistics_table(model.factor_data)
    )
    assert not shown
    figure = artifacts.figures[0]
    assert figure.axes
    for owned_figure in artifacts.figures:
        pyplot.close(owned_figure)
    assert not pyplot.get_fignums()


def test_all_enhanced_tear_workflows_read_the_model_without_reentering_kernels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The renderer boundary consumes snapshots; it never recomputes analytical fields."""

    import fincore.factor_analysis.performance as performance
    from fincore.factor_analysis.tears import (
        close_owned_figures,
        create_event_returns_tear_sheet,
        create_event_study_tear_sheet,
        create_full_tear_sheet,
        create_information_tear_sheet,
        create_returns_tear_sheet,
        create_summary_tear_sheet,
        create_turnover_tear_sheet,
    )

    model = _workflow_model(event=True, by_group=True)

    def unexpected_kernel(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("tear workflow re-entered an analytical kernel after model assembly")

    for name in (
        "factor_returns",
        "mean_return_by_quantile",
        "factor_information_coefficient",
        "mean_information_coefficient",
        "quantile_turnover",
        "factor_rank_autocorrelation",
        "average_cumulative_return_by_quantile",
    ):
        monkeypatch.setattr(performance, name, unexpected_kernel)

    artifacts = (
        create_summary_tear_sheet(model),
        create_returns_tear_sheet(model, by_group=True),
        create_information_tear_sheet(model, by_group=True),
        create_turnover_tear_sheet(model),
        create_full_tear_sheet(model, by_group=True),
        create_event_returns_tear_sheet(model, by_group=True),
        create_event_study_tear_sheet(model, avgretplot=(1, 2)),
    )
    assert all(result.figures and result.tables for result in artifacts)
    for result in artifacts:
        close_owned_figures(result)
    assert not _pyplot().get_fignums()


@pytest.mark.parametrize(
    "workflow_name,event,by_group,kwargs,expected_figures,required_tables",
    (
        (
            "create_summary_tear_sheet",
            False,
            False,
            {},
            1,
            frozenset(("quantile_statistics", "returns", "information", "turnover", "autocorrelation")),
        ),
        (
            "create_returns_tear_sheet",
            False,
            True,
            {"by_group": True},
            2,
            frozenset(("quantile_statistics", "returns")),
        ),
        (
            "create_information_tear_sheet",
            False,
            True,
            {"by_group": True},
            1,
            frozenset(("information",)),
        ),
        (
            "create_turnover_tear_sheet",
            False,
            False,
            {},
            1,
            frozenset(("turnover", "autocorrelation")),
        ),
        (
            "create_full_tear_sheet",
            False,
            True,
            {"by_group": True},
            4,
            frozenset(("quantile_statistics", "returns.returns", "information.information", "turnover.turnover")),
        ),
        (
            "create_event_returns_tear_sheet",
            True,
            True,
            {"by_group": True},
            2,
            frozenset(("event_average", "event_windows")),
        ),
        (
            "create_event_study_tear_sheet",
            True,
            False,
            {"avgretplot": (1, 2)},
            3,
            frozenset(("event_returns.event_average", "event_returns.event_windows")),
        ),
    ),
)
def test_enhanced_workflow_matrix_exposes_expected_sections_without_showing(
    monkeypatch: pytest.MonkeyPatch,
    workflow_name: str,
    event: bool,
    by_group: bool,
    kwargs: dict[str, object],
    expected_figures: int,
    required_tables: frozenset[str],
) -> None:
    """All seven enhanced APIs expose actual sections and retain caller lifecycle control."""

    import fincore.factor_analysis.tears as tears

    pyplot = _pyplot()
    shown: list[object] = []
    monkeypatch.setattr(pyplot, "show", lambda *args, **keyword_args: shown.append((args, keyword_args)))
    workflow = cast("Any", getattr(tears, workflow_name))
    artifacts = workflow(_workflow_model(event=event, by_group=by_group), **kwargs)

    assert len(artifacts.figures) == expected_figures
    assert all(figure.axes for figure in artifacts.figures)
    assert required_tables.issubset(artifacts.tables)
    assert not shown
    tears.close_owned_figures(artifacts)
    assert not pyplot.get_fignums()


def test_event_study_rate_of_return_switch_changes_multi_period_return_artists() -> None:
    """The event-study flag retains the source's period-rate conversion semantics."""

    from fincore.factor_analysis.tears import close_owned_figures, create_event_study_tear_sheet

    model = _source_event_model(
        study=False,
        ordinal=0,
        input_ordinal=0,
        long_short=False,
        group_neutral=False,
        by_group=False,
    )
    simple = create_event_study_tear_sheet(model, avgretplot=(1, 2), rate_of_ret=False)
    rate = create_event_study_tear_sheet(model, avgretplot=(1, 2), rate_of_ret=True)
    simple_heights = [patch.get_height() for patch in simple.figures[-1].axes[0].patches]
    rate_heights = [patch.get_height() for patch in rate.figures[-1].axes[0].patches]

    assert len(simple_heights) == len(rate_heights)
    assert any(
        not pd.isna(simple_height) and not pd.isna(rate_height) and abs(simple_height - rate_height) > 1e-10
        for simple_height, rate_height in zip(simple_heights, rate_heights, strict=True)
    )
    close_owned_figures(simple)
    close_owned_figures(rate)
    assert not _pyplot().get_fignums()


def test_grid_figure_owns_one_lazy_matplotlib_grid_and_closes_only_itself() -> None:
    """The shared grid layout has frozen row/cell lifecycle primitives."""

    from fincore.factor_analysis.tears import GridFigure

    pyplot = _pyplot()
    supplied_figure, _ = pyplot.subplots()
    grid = GridFigure(2, 2)
    first = grid.next_row()
    second = grid.next_cell()

    assert first.figure is grid.fig
    assert second.figure is grid.fig
    grid.close()
    assert grid.fig is None
    assert supplied_figure.number in pyplot.get_fignums()
    pyplot.close(supplied_figure)
    assert not pyplot.get_fignums()


def test_strict_summary_tear_sheet_projects_show_then_closes_its_owned_figure(
    clean_factor_data: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The legacy facade uses the enhanced model workflow but returns ``None``."""

    from fincore.alphalens.tears import create_summary_tear_sheet

    pyplot = _pyplot()
    shown: list[object] = []
    monkeypatch.setattr(pyplot, "show", lambda *args, **kwargs: shown.append((args, kwargs)))

    assert create_summary_tear_sheet(clean_factor_data, long_short=False, group_neutral=False) is None
    assert len(shown) == 1
    assert not pyplot.get_fignums()


def test_strict_event_workflows_allocate_only_their_source_sections(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Strict event group/study workflows do not allocate hidden composite grids."""

    from fincore.alphalens import tears as strict_tears

    model = _workflow_model(event=True, by_group=True)
    returns = model.event_input_snapshot
    assert returns is not None
    pyplot = _pyplot()
    original_figure = pyplot.figure
    created: list[object] = []

    def record_figure(*args: object, **kwargs: object) -> object:
        figure = original_figure(*args, **kwargs)
        created.append(figure)
        return figure

    monkeypatch.setattr(pyplot, "figure", record_figure)
    monkeypatch.setattr(pyplot, "show", lambda *args, **kwargs: None)
    monkeypatch.setattr(strict_tears._plotting, "_display_table", lambda *args, **kwargs: None)
    assert (
        strict_tears.create_event_returns_tear_sheet(
            model.factor_data,
            returns,
            avgretplot=(1, 2),
            by_group=True,
            set_context=False,
        )
        is None
    )
    assert len(created) == 2
    assert not pyplot.get_fignums()

    created.clear()
    assert (
        strict_tears.create_event_study_tear_sheet(
            model.factor_data,
            returns,
            avgretplot=(1, 2),
            set_context=False,
        )
        is None
    )
    assert len(created) == 3
    assert not pyplot.get_fignums()


def test_strict_event_study_ignores_returns_when_average_section_is_disabled(
    clean_factor_data: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The source does not validate an unused event-return placeholder."""

    from fincore.alphalens import tears

    pyplot = _pyplot()
    calls: list[object] = []
    monkeypatch.setattr(pyplot, "show", lambda *args, **kwargs: calls.append((args, kwargs)))
    monkeypatch.setattr(tears._plotting, "_display_table", lambda *args, **kwargs: None)

    assert (
        tears.create_event_study_tear_sheet(
            clean_factor_data,
            "not-a-dataframe",
            avgretplot=None,
            set_context=False,
        )
        is None
    )
    assert len(calls) == 2
    assert not pyplot.get_fignums()


def test_strict_event_study_defers_malformed_returns_until_after_distribution(
    clean_factor_data: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bad optional return frame fails after the pinned first event section."""

    from fincore.alphalens import tears

    pyplot = _pyplot()
    shown: list[object] = []
    displayed: list[object] = []
    monkeypatch.setattr(pyplot, "show", lambda *args, **kwargs: shown.append((args, kwargs)))
    monkeypatch.setattr(
        tears._plotting,
        "_display_table",
        lambda heading, table, **kwargs: displayed.append(heading),
    )

    with pytest.raises(AttributeError, match="builtin_function_or_method.*get_loc"):
        tears.create_event_study_tear_sheet(
            clean_factor_data,
            "bad",
            avgretplot=(1, 2),
            set_context=False,
        )

    assert displayed == ["Quantiles Statistics"]
    assert len(shown) == 1
    assert not pyplot.get_fignums()


@pytest.mark.parametrize(
    "returns",
    (
        pd.Series([1.0, 2.0]),
        pd.DataFrame(),
        pd.DataFrame({"Z": [1.0, 2.0]}, index=pd.date_range("2040-01-01", periods=2)),
    ),
)
def test_strict_event_study_preserves_late_no_event_window_error(
    clean_factor_data: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
    returns: object,
) -> None:
    """No aligned event slices fail only after the source distribution section."""

    from fincore.alphalens import tears

    pyplot = _pyplot()
    shown: list[object] = []
    displayed: list[object] = []
    monkeypatch.setattr(pyplot, "show", lambda *args, **kwargs: shown.append((args, kwargs)))
    monkeypatch.setattr(
        tears._plotting,
        "_display_table",
        lambda heading, table, **kwargs: displayed.append(heading),
    )

    with pytest.raises(ValueError, match="No objects to concatenate"):
        tears.create_event_study_tear_sheet(
            clean_factor_data,
            returns,
            avgretplot=(1, 2),
            set_context=False,
        )

    assert displayed == ["Quantiles Statistics"]
    assert len(shown) == 1
    assert not pyplot.get_fignums()


def test_strict_event_study_defers_nonoverlapping_price_dates_until_after_distribution(
    clean_factor_data: pd.DataFrame,
    prices: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Matching assets without an event date still take the late source error."""

    from fincore.alphalens import tears

    returns = pd.DataFrame(
        0.01,
        index=pd.date_range("2040-01-01", periods=3, name="date"),
        columns=prices.columns,
    )
    pyplot = _pyplot()
    shown: list[object] = []
    displayed: list[object] = []
    monkeypatch.setattr(pyplot, "show", lambda *args, **kwargs: shown.append((args, kwargs)))
    monkeypatch.setattr(
        tears._plotting,
        "_display_table",
        lambda heading, table, **kwargs: displayed.append(heading),
    )

    with pytest.raises(ValueError, match="No objects to concatenate"):
        tears.create_event_study_tear_sheet(
            clean_factor_data,
            returns,
            avgretplot=(1, 2),
            set_context=False,
        )

    assert displayed == ["Quantiles Statistics"]
    assert len(shown) == 1
    assert not pyplot.get_fignums()


@pytest.mark.parametrize(
    ("case", "expected_exception", "message"),
    (
        ("none", AttributeError, "NoneType.*index"),
        ("string", AttributeError, "builtin_function_or_method.*get_loc"),
        ("series", ValueError, "No objects to concatenate"),
        ("empty", ValueError, "No objects to concatenate"),
        ("wrong_assets", ValueError, "No objects to concatenate"),
        ("no_dates", ValueError, "No objects to concatenate"),
        ("duplicate_dates", TypeError, "unsupported operand type.*slice.*int"),
        ("partial_assets", KeyError, None),
    ),
)
def test_strict_event_returns_preserves_source_invalid_return_errors(
    clean_factor_data: pd.DataFrame,
    prices: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    expected_exception: type[Exception],
    message: str | None,
) -> None:
    """Direct event returns retain the pinned event-window error surface."""

    from fincore.alphalens import tears

    event_date = clean_factor_data.index.get_level_values("date")[0]
    returns: object
    if case == "none":
        returns = None
    elif case == "string":
        returns = "bad"
    elif case == "series":
        returns = pd.Series([1.0, 2.0])
    elif case == "empty":
        returns = pd.DataFrame()
    elif case == "wrong_assets":
        returns = pd.DataFrame({"Z": [1.0, 2.0]}, index=pd.date_range("2040-01-01", periods=2))
    elif case == "no_dates":
        returns = pd.DataFrame(0.01, index=pd.date_range("2040-01-01", periods=3), columns=prices.columns)
    elif case == "duplicate_dates":
        returns = pd.concat([prices.loc[[event_date]], prices])
    else:
        assert case == "partial_assets"
        returns = prices.iloc[:, :1]

    pyplot = _pyplot()
    monkeypatch.setattr(pyplot, "show", lambda *args, **kwargs: None)
    monkeypatch.setattr(tears._plotting, "_display_table", lambda *args, **kwargs: None)
    with pytest.raises(expected_exception, match=message):
        tears.create_event_returns_tear_sheet(
            clean_factor_data,
            returns,
            avgretplot=(1, 2),
            set_context=False,
        )
    assert not pyplot.get_fignums()


@pytest.mark.parametrize(
    ("avgretplot", "expected_exception", "message"),
    (
        ("x", ValueError, "not enough values to unpack"),
        ((1,), ValueError, "not enough values to unpack"),
        (42, TypeError, "cannot unpack non-iterable int object"),
        ((1, 2, 3), ValueError, "too many values to unpack"),
    ),
)
def test_strict_event_study_defers_bad_avgretplot_unpacking_until_after_distribution(
    clean_factor_data: pd.DataFrame,
    prices: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
    avgretplot: object,
    expected_exception: type[Exception],
    message: str,
) -> None:
    """The optional event child owns the source's late tuple-unpacking error."""

    from fincore.alphalens import tears

    pyplot = _pyplot()
    shown: list[object] = []
    displayed: list[object] = []
    monkeypatch.setattr(pyplot, "show", lambda *args, **kwargs: shown.append((args, kwargs)))
    monkeypatch.setattr(
        tears._plotting,
        "_display_table",
        lambda heading, table, **kwargs: displayed.append(heading),
    )

    with pytest.raises(expected_exception, match=message):
        tears.create_event_study_tear_sheet(
            clean_factor_data,
            prices,
            avgretplot=avgretplot,
            set_context=False,
        )

    assert displayed == ["Quantiles Statistics"]
    assert len(shown) == 1
    assert not pyplot.get_fignums()


def test_strict_returns_preserves_the_source_empty_group_grid_error(
    clean_factor_data: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An all-NaN group column fails only after the aggregate return section.

    Pinned Alphalens constructs a zero-row ``GridFigure`` for its return
    group panel.  Matplotlib then raises from ``GridSpec`` after the aggregate
    Figure has been shown and closed.
    """

    from fincore.alphalens import tears

    factor_data = clean_factor_data.assign(group=np.nan)
    pyplot = _pyplot()
    shown: list[object] = []
    monkeypatch.setattr(pyplot, "show", lambda *args, **kwargs: shown.append((args, kwargs)))

    with pytest.raises(ValueError, match="Number of rows must be a positive integer, not 0"):
        tears.create_returns_tear_sheet(
            factor_data,
            by_group=True,
            set_context=False,
        )

    assert len(shown) == 1
    # The source-shaped error is retained, while our strict ownership wrapper
    # still releases the exceptional Figure before returning control.
    assert not pyplot.get_fignums()


@pytest.mark.parametrize("name", ("create_returns_tear_sheet", "create_full_tear_sheet"))
def test_strict_return_workflows_preserve_unused_categorical_group_error(
    clean_factor_data: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
) -> None:
    """Unused categorical groups retain the source all-NaN bar branch."""

    from fincore.alphalens import tears

    factor_data = clean_factor_data.copy(deep=True)
    factor_data["group"] = pd.Categorical([np.nan] * len(factor_data), categories=["x", "y"])
    pyplot = _pyplot()
    shown: list[object] = []
    monkeypatch.setattr(pyplot, "show", lambda *args, **kwargs: shown.append((args, kwargs)))
    monkeypatch.setattr(tears._plotting, "_display_table", lambda *args, **kwargs: None)

    workflow = cast("Any", getattr(tears, name))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        with pytest.raises(ValueError, match="Axis limits cannot be NaN or Inf"):
            workflow(factor_data, by_group=True, set_context=False)
    assert any("All-NaN slice encountered" in str(item.message) for item in caught)
    assert len(shown) == 1
    assert not pyplot.get_fignums()


@pytest.mark.parametrize("categorical", (False, True))
def test_strict_information_preserves_source_all_missing_group_branches(
    clean_factor_data: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
    categorical: bool,
) -> None:
    """Empty numeric and unused-categorical groups follow distinct source paths."""

    from fincore.alphalens import tears

    factor_data = clean_factor_data.copy(deep=True)
    if categorical:
        factor_data["group"] = pd.Categorical([np.nan] * len(factor_data), categories=["x", "y"])
    else:
        factor_data["group"] = np.nan
    pyplot = _pyplot()
    shown_axes: list[int] = []
    monkeypatch.setattr(
        pyplot,
        "show",
        lambda *args, **kwargs: shown_axes.extend(len(pyplot.figure(number).axes) for number in pyplot.get_fignums()),
    )
    monkeypatch.setattr(tears._plotting, "_display_table", lambda *args, **kwargs: None)

    if not categorical:
        with pytest.raises(IndexError, match="index 0 is out of bounds"):
            tears.create_information_tear_sheet(factor_data, by_group=True, set_context=False)
        assert not shown_axes
    else:
        assert tears.create_information_tear_sheet(factor_data, by_group=True, set_context=False) is None
        assert shown_axes == [10]
    assert not pyplot.get_fignums()


@pytest.mark.parametrize(
    "source_invocation_id",
    [
        pytest.param(
            "tests/test_tears.py::TearsTestCase::test_create_event_returns_tear_sheet#00/input-00/call-00",
            id="tests/test_tears.py::TearsTestCase::test_create_event_returns_tear_sheet#00/input-00/call-00",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_tears.py::TearsTestCase::test_create_event_returns_tear_sheet#00/input-00/call-00"
            ),
        ),
        pytest.param(
            "tests/test_tears.py::TearsTestCase::test_create_event_returns_tear_sheet#00/input-00/call-01",
            id="tests/test_tears.py::TearsTestCase::test_create_event_returns_tear_sheet#00/input-00/call-01",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_tears.py::TearsTestCase::test_create_event_returns_tear_sheet#00/input-00/call-01"
            ),
        ),
        pytest.param(
            "tests/test_tears.py::TearsTestCase::test_create_event_returns_tear_sheet#00/input-00/call-02",
            id="tests/test_tears.py::TearsTestCase::test_create_event_returns_tear_sheet#00/input-00/call-02",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_tears.py::TearsTestCase::test_create_event_returns_tear_sheet#00/input-00/call-02"
            ),
        ),
        pytest.param(
            "tests/test_tears.py::TearsTestCase::test_create_event_returns_tear_sheet#00/input-00/call-03",
            id="tests/test_tears.py::TearsTestCase::test_create_event_returns_tear_sheet#00/input-00/call-03",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_tears.py::TearsTestCase::test_create_event_returns_tear_sheet#00/input-00/call-03"
            ),
        ),
        pytest.param(
            "tests/test_tears.py::TearsTestCase::test_create_event_returns_tear_sheet#00/input-00/call-04",
            id="tests/test_tears.py::TearsTestCase::test_create_event_returns_tear_sheet#00/input-00/call-04",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_tears.py::TearsTestCase::test_create_event_returns_tear_sheet#00/input-00/call-04"
            ),
        ),
        pytest.param(
            "tests/test_tears.py::TearsTestCase::test_create_event_returns_tear_sheet#00/input-00/call-05",
            id="tests/test_tears.py::TearsTestCase::test_create_event_returns_tear_sheet#00/input-00/call-05",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_tears.py::TearsTestCase::test_create_event_returns_tear_sheet#00/input-00/call-05"
            ),
        ),
        pytest.param(
            "tests/test_tears.py::TearsTestCase::test_create_event_returns_tear_sheet#00/input-01/call-00",
            id="tests/test_tears.py::TearsTestCase::test_create_event_returns_tear_sheet#00/input-01/call-00",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_tears.py::TearsTestCase::test_create_event_returns_tear_sheet#00/input-01/call-00"
            ),
        ),
        pytest.param(
            "tests/test_tears.py::TearsTestCase::test_create_event_returns_tear_sheet#00/input-01/call-01",
            id="tests/test_tears.py::TearsTestCase::test_create_event_returns_tear_sheet#00/input-01/call-01",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_tears.py::TearsTestCase::test_create_event_returns_tear_sheet#00/input-01/call-01"
            ),
        ),
        pytest.param(
            "tests/test_tears.py::TearsTestCase::test_create_event_returns_tear_sheet#00/input-01/call-02",
            id="tests/test_tears.py::TearsTestCase::test_create_event_returns_tear_sheet#00/input-01/call-02",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_tears.py::TearsTestCase::test_create_event_returns_tear_sheet#00/input-01/call-02"
            ),
        ),
        pytest.param(
            "tests/test_tears.py::TearsTestCase::test_create_event_returns_tear_sheet#00/input-01/call-03",
            id="tests/test_tears.py::TearsTestCase::test_create_event_returns_tear_sheet#00/input-01/call-03",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_tears.py::TearsTestCase::test_create_event_returns_tear_sheet#00/input-01/call-03"
            ),
        ),
        pytest.param(
            "tests/test_tears.py::TearsTestCase::test_create_event_returns_tear_sheet#00/input-01/call-04",
            id="tests/test_tears.py::TearsTestCase::test_create_event_returns_tear_sheet#00/input-01/call-04",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_tears.py::TearsTestCase::test_create_event_returns_tear_sheet#00/input-01/call-04"
            ),
        ),
        pytest.param(
            "tests/test_tears.py::TearsTestCase::test_create_event_returns_tear_sheet#00/input-01/call-05",
            id="tests/test_tears.py::TearsTestCase::test_create_event_returns_tear_sheet#00/input-01/call-05",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_tears.py::TearsTestCase::test_create_event_returns_tear_sheet#00/input-01/call-05"
            ),
        ),
        pytest.param(
            "tests/test_tears.py::TearsTestCase::test_create_event_returns_tear_sheet#01/input-00/call-00",
            id="tests/test_tears.py::TearsTestCase::test_create_event_returns_tear_sheet#01/input-00/call-00",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_tears.py::TearsTestCase::test_create_event_returns_tear_sheet#01/input-00/call-00"
            ),
        ),
        pytest.param(
            "tests/test_tears.py::TearsTestCase::test_create_event_returns_tear_sheet#01/input-00/call-01",
            id="tests/test_tears.py::TearsTestCase::test_create_event_returns_tear_sheet#01/input-00/call-01",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_tears.py::TearsTestCase::test_create_event_returns_tear_sheet#01/input-00/call-01"
            ),
        ),
        pytest.param(
            "tests/test_tears.py::TearsTestCase::test_create_event_returns_tear_sheet#01/input-00/call-02",
            id="tests/test_tears.py::TearsTestCase::test_create_event_returns_tear_sheet#01/input-00/call-02",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_tears.py::TearsTestCase::test_create_event_returns_tear_sheet#01/input-00/call-02"
            ),
        ),
        pytest.param(
            "tests/test_tears.py::TearsTestCase::test_create_event_returns_tear_sheet#01/input-00/call-03",
            id="tests/test_tears.py::TearsTestCase::test_create_event_returns_tear_sheet#01/input-00/call-03",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_tears.py::TearsTestCase::test_create_event_returns_tear_sheet#01/input-00/call-03"
            ),
        ),
        pytest.param(
            "tests/test_tears.py::TearsTestCase::test_create_event_returns_tear_sheet#01/input-00/call-04",
            id="tests/test_tears.py::TearsTestCase::test_create_event_returns_tear_sheet#01/input-00/call-04",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_tears.py::TearsTestCase::test_create_event_returns_tear_sheet#01/input-00/call-04"
            ),
        ),
        pytest.param(
            "tests/test_tears.py::TearsTestCase::test_create_event_returns_tear_sheet#01/input-00/call-05",
            id="tests/test_tears.py::TearsTestCase::test_create_event_returns_tear_sheet#01/input-00/call-05",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_tears.py::TearsTestCase::test_create_event_returns_tear_sheet#01/input-00/call-05"
            ),
        ),
        pytest.param(
            "tests/test_tears.py::TearsTestCase::test_create_event_returns_tear_sheet#01/input-01/call-00",
            id="tests/test_tears.py::TearsTestCase::test_create_event_returns_tear_sheet#01/input-01/call-00",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_tears.py::TearsTestCase::test_create_event_returns_tear_sheet#01/input-01/call-00"
            ),
        ),
        pytest.param(
            "tests/test_tears.py::TearsTestCase::test_create_event_returns_tear_sheet#01/input-01/call-01",
            id="tests/test_tears.py::TearsTestCase::test_create_event_returns_tear_sheet#01/input-01/call-01",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_tears.py::TearsTestCase::test_create_event_returns_tear_sheet#01/input-01/call-01"
            ),
        ),
        pytest.param(
            "tests/test_tears.py::TearsTestCase::test_create_event_returns_tear_sheet#01/input-01/call-02",
            id="tests/test_tears.py::TearsTestCase::test_create_event_returns_tear_sheet#01/input-01/call-02",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_tears.py::TearsTestCase::test_create_event_returns_tear_sheet#01/input-01/call-02"
            ),
        ),
        pytest.param(
            "tests/test_tears.py::TearsTestCase::test_create_event_returns_tear_sheet#01/input-01/call-03",
            id="tests/test_tears.py::TearsTestCase::test_create_event_returns_tear_sheet#01/input-01/call-03",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_tears.py::TearsTestCase::test_create_event_returns_tear_sheet#01/input-01/call-03"
            ),
        ),
        pytest.param(
            "tests/test_tears.py::TearsTestCase::test_create_event_returns_tear_sheet#01/input-01/call-04",
            id="tests/test_tears.py::TearsTestCase::test_create_event_returns_tear_sheet#01/input-01/call-04",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_tears.py::TearsTestCase::test_create_event_returns_tear_sheet#01/input-01/call-04"
            ),
        ),
        pytest.param(
            "tests/test_tears.py::TearsTestCase::test_create_event_returns_tear_sheet#01/input-01/call-05",
            id="tests/test_tears.py::TearsTestCase::test_create_event_returns_tear_sheet#01/input-01/call-05",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_tears.py::TearsTestCase::test_create_event_returns_tear_sheet#01/input-01/call-05"
            ),
        ),
        pytest.param(
            "tests/test_tears.py::TearsTestCase::test_create_event_returns_tear_sheet#02/input-00/call-00",
            id="tests/test_tears.py::TearsTestCase::test_create_event_returns_tear_sheet#02/input-00/call-00",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_tears.py::TearsTestCase::test_create_event_returns_tear_sheet#02/input-00/call-00"
            ),
        ),
        pytest.param(
            "tests/test_tears.py::TearsTestCase::test_create_event_returns_tear_sheet#02/input-00/call-01",
            id="tests/test_tears.py::TearsTestCase::test_create_event_returns_tear_sheet#02/input-00/call-01",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_tears.py::TearsTestCase::test_create_event_returns_tear_sheet#02/input-00/call-01"
            ),
        ),
        pytest.param(
            "tests/test_tears.py::TearsTestCase::test_create_event_returns_tear_sheet#02/input-00/call-02",
            id="tests/test_tears.py::TearsTestCase::test_create_event_returns_tear_sheet#02/input-00/call-02",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_tears.py::TearsTestCase::test_create_event_returns_tear_sheet#02/input-00/call-02"
            ),
        ),
        pytest.param(
            "tests/test_tears.py::TearsTestCase::test_create_event_returns_tear_sheet#02/input-00/call-03",
            id="tests/test_tears.py::TearsTestCase::test_create_event_returns_tear_sheet#02/input-00/call-03",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_tears.py::TearsTestCase::test_create_event_returns_tear_sheet#02/input-00/call-03"
            ),
        ),
        pytest.param(
            "tests/test_tears.py::TearsTestCase::test_create_event_returns_tear_sheet#02/input-00/call-04",
            id="tests/test_tears.py::TearsTestCase::test_create_event_returns_tear_sheet#02/input-00/call-04",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_tears.py::TearsTestCase::test_create_event_returns_tear_sheet#02/input-00/call-04"
            ),
        ),
        pytest.param(
            "tests/test_tears.py::TearsTestCase::test_create_event_returns_tear_sheet#02/input-00/call-05",
            id="tests/test_tears.py::TearsTestCase::test_create_event_returns_tear_sheet#02/input-00/call-05",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_tears.py::TearsTestCase::test_create_event_returns_tear_sheet#02/input-00/call-05"
            ),
        ),
        pytest.param(
            "tests/test_tears.py::TearsTestCase::test_create_event_returns_tear_sheet#02/input-01/call-00",
            id="tests/test_tears.py::TearsTestCase::test_create_event_returns_tear_sheet#02/input-01/call-00",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_tears.py::TearsTestCase::test_create_event_returns_tear_sheet#02/input-01/call-00"
            ),
        ),
        pytest.param(
            "tests/test_tears.py::TearsTestCase::test_create_event_returns_tear_sheet#02/input-01/call-01",
            id="tests/test_tears.py::TearsTestCase::test_create_event_returns_tear_sheet#02/input-01/call-01",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_tears.py::TearsTestCase::test_create_event_returns_tear_sheet#02/input-01/call-01"
            ),
        ),
        pytest.param(
            "tests/test_tears.py::TearsTestCase::test_create_event_returns_tear_sheet#02/input-01/call-02",
            id="tests/test_tears.py::TearsTestCase::test_create_event_returns_tear_sheet#02/input-01/call-02",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_tears.py::TearsTestCase::test_create_event_returns_tear_sheet#02/input-01/call-02"
            ),
        ),
        pytest.param(
            "tests/test_tears.py::TearsTestCase::test_create_event_returns_tear_sheet#02/input-01/call-03",
            id="tests/test_tears.py::TearsTestCase::test_create_event_returns_tear_sheet#02/input-01/call-03",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_tears.py::TearsTestCase::test_create_event_returns_tear_sheet#02/input-01/call-03"
            ),
        ),
        pytest.param(
            "tests/test_tears.py::TearsTestCase::test_create_event_returns_tear_sheet#02/input-01/call-04",
            id="tests/test_tears.py::TearsTestCase::test_create_event_returns_tear_sheet#02/input-01/call-04",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_tears.py::TearsTestCase::test_create_event_returns_tear_sheet#02/input-01/call-04"
            ),
        ),
        pytest.param(
            "tests/test_tears.py::TearsTestCase::test_create_event_returns_tear_sheet#02/input-01/call-05",
            id="tests/test_tears.py::TearsTestCase::test_create_event_returns_tear_sheet#02/input-01/call-05",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_tears.py::TearsTestCase::test_create_event_returns_tear_sheet#02/input-01/call-05"
            ),
        ),
        pytest.param(
            "tests/test_tears.py::TearsTestCase::test_create_event_returns_tear_sheet#03/input-00/call-00",
            id="tests/test_tears.py::TearsTestCase::test_create_event_returns_tear_sheet#03/input-00/call-00",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_tears.py::TearsTestCase::test_create_event_returns_tear_sheet#03/input-00/call-00"
            ),
        ),
        pytest.param(
            "tests/test_tears.py::TearsTestCase::test_create_event_returns_tear_sheet#03/input-00/call-01",
            id="tests/test_tears.py::TearsTestCase::test_create_event_returns_tear_sheet#03/input-00/call-01",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_tears.py::TearsTestCase::test_create_event_returns_tear_sheet#03/input-00/call-01"
            ),
        ),
        pytest.param(
            "tests/test_tears.py::TearsTestCase::test_create_event_returns_tear_sheet#03/input-00/call-02",
            id="tests/test_tears.py::TearsTestCase::test_create_event_returns_tear_sheet#03/input-00/call-02",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_tears.py::TearsTestCase::test_create_event_returns_tear_sheet#03/input-00/call-02"
            ),
        ),
        pytest.param(
            "tests/test_tears.py::TearsTestCase::test_create_event_returns_tear_sheet#03/input-00/call-03",
            id="tests/test_tears.py::TearsTestCase::test_create_event_returns_tear_sheet#03/input-00/call-03",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_tears.py::TearsTestCase::test_create_event_returns_tear_sheet#03/input-00/call-03"
            ),
        ),
        pytest.param(
            "tests/test_tears.py::TearsTestCase::test_create_event_returns_tear_sheet#03/input-00/call-04",
            id="tests/test_tears.py::TearsTestCase::test_create_event_returns_tear_sheet#03/input-00/call-04",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_tears.py::TearsTestCase::test_create_event_returns_tear_sheet#03/input-00/call-04"
            ),
        ),
        pytest.param(
            "tests/test_tears.py::TearsTestCase::test_create_event_returns_tear_sheet#03/input-00/call-05",
            id="tests/test_tears.py::TearsTestCase::test_create_event_returns_tear_sheet#03/input-00/call-05",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_tears.py::TearsTestCase::test_create_event_returns_tear_sheet#03/input-00/call-05"
            ),
        ),
        pytest.param(
            "tests/test_tears.py::TearsTestCase::test_create_event_returns_tear_sheet#03/input-01/call-00",
            id="tests/test_tears.py::TearsTestCase::test_create_event_returns_tear_sheet#03/input-01/call-00",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_tears.py::TearsTestCase::test_create_event_returns_tear_sheet#03/input-01/call-00"
            ),
        ),
        pytest.param(
            "tests/test_tears.py::TearsTestCase::test_create_event_returns_tear_sheet#03/input-01/call-01",
            id="tests/test_tears.py::TearsTestCase::test_create_event_returns_tear_sheet#03/input-01/call-01",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_tears.py::TearsTestCase::test_create_event_returns_tear_sheet#03/input-01/call-01"
            ),
        ),
        pytest.param(
            "tests/test_tears.py::TearsTestCase::test_create_event_returns_tear_sheet#03/input-01/call-02",
            id="tests/test_tears.py::TearsTestCase::test_create_event_returns_tear_sheet#03/input-01/call-02",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_tears.py::TearsTestCase::test_create_event_returns_tear_sheet#03/input-01/call-02"
            ),
        ),
        pytest.param(
            "tests/test_tears.py::TearsTestCase::test_create_event_returns_tear_sheet#03/input-01/call-03",
            id="tests/test_tears.py::TearsTestCase::test_create_event_returns_tear_sheet#03/input-01/call-03",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_tears.py::TearsTestCase::test_create_event_returns_tear_sheet#03/input-01/call-03"
            ),
        ),
        pytest.param(
            "tests/test_tears.py::TearsTestCase::test_create_event_returns_tear_sheet#03/input-01/call-04",
            id="tests/test_tears.py::TearsTestCase::test_create_event_returns_tear_sheet#03/input-01/call-04",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_tears.py::TearsTestCase::test_create_event_returns_tear_sheet#03/input-01/call-04"
            ),
        ),
        pytest.param(
            "tests/test_tears.py::TearsTestCase::test_create_event_returns_tear_sheet#03/input-01/call-05",
            id="tests/test_tears.py::TearsTestCase::test_create_event_returns_tear_sheet#03/input-01/call-05",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_tears.py::TearsTestCase::test_create_event_returns_tear_sheet#03/input-01/call-05"
            ),
        ),
    ],
)
def test_create_event_returns_tear_sheet_upstream_invocation(
    source_invocation_id: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rebuild every frozen event-return input/call path as a C4 artifact workflow."""

    from fincore.factor_analysis.tears import close_owned_figures, create_event_returns_tear_sheet

    model, options = _event_model_for_invocation(source_invocation_id, study=False)
    assert_event_source_calendar(model, options)
    pyplot = _pyplot()
    calls: list[object] = []
    monkeypatch.setattr(pyplot, "show", lambda *args, **kwargs: calls.append((args, kwargs)))

    artifacts = create_event_returns_tear_sheet(
        model,
        std_bar=True,
        by_group=cast("bool", options["by_group"]),
        show=True,
    )
    expected_figures = 2 if cast("bool", options["by_group"]) else 1
    assert_figure_artifacts(
        artifacts,
        expected_figures=expected_figures,
        required_tables=frozenset(("event_average", "event_windows")),
    )
    assert_show_called(calls, artifacts, expected_show=expected_figures)
    assert_artifact_ownership(artifacts)
    assert_event_snapshot_values(artifacts, model)
    assert model.config.long_short is source_invocation_id.endswith(("call-01", "call-04"))
    assert model.config.group_neutral is source_invocation_id.endswith(("call-02", "call-05"))
    close_owned_figures(artifacts)
    assert_no_open_figures(artifacts)
    assert_strict_event_source_invocation(
        source_invocation_id,
        model,
        options,
        monkeypatch,
        study=False,
    )


@pytest.mark.parametrize(
    "source_invocation_id",
    [
        pytest.param(
            "tests/test_tears.py::TearsTestCase::test_create_event_study_tear_sheet#00/input-00/call-00",
            id="tests/test_tears.py::TearsTestCase::test_create_event_study_tear_sheet#00/input-00/call-00",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_tears.py::TearsTestCase::test_create_event_study_tear_sheet#00/input-00/call-00"
            ),
        ),
        pytest.param(
            "tests/test_tears.py::TearsTestCase::test_create_event_study_tear_sheet#00/input-01/call-00",
            id="tests/test_tears.py::TearsTestCase::test_create_event_study_tear_sheet#00/input-01/call-00",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_tears.py::TearsTestCase::test_create_event_study_tear_sheet#00/input-01/call-00"
            ),
        ),
        pytest.param(
            "tests/test_tears.py::TearsTestCase::test_create_event_study_tear_sheet#01/input-00/call-00",
            id="tests/test_tears.py::TearsTestCase::test_create_event_study_tear_sheet#01/input-00/call-00",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_tears.py::TearsTestCase::test_create_event_study_tear_sheet#01/input-00/call-00"
            ),
        ),
        pytest.param(
            "tests/test_tears.py::TearsTestCase::test_create_event_study_tear_sheet#01/input-01/call-00",
            id="tests/test_tears.py::TearsTestCase::test_create_event_study_tear_sheet#01/input-01/call-00",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_tears.py::TearsTestCase::test_create_event_study_tear_sheet#01/input-01/call-00"
            ),
        ),
        pytest.param(
            "tests/test_tears.py::TearsTestCase::test_create_event_study_tear_sheet#02/input-00/call-00",
            id="tests/test_tears.py::TearsTestCase::test_create_event_study_tear_sheet#02/input-00/call-00",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_tears.py::TearsTestCase::test_create_event_study_tear_sheet#02/input-00/call-00"
            ),
        ),
        pytest.param(
            "tests/test_tears.py::TearsTestCase::test_create_event_study_tear_sheet#02/input-01/call-00",
            id="tests/test_tears.py::TearsTestCase::test_create_event_study_tear_sheet#02/input-01/call-00",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_tears.py::TearsTestCase::test_create_event_study_tear_sheet#02/input-01/call-00"
            ),
        ),
        pytest.param(
            "tests/test_tears.py::TearsTestCase::test_create_event_study_tear_sheet#03/input-00/call-00",
            id="tests/test_tears.py::TearsTestCase::test_create_event_study_tear_sheet#03/input-00/call-00",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_tears.py::TearsTestCase::test_create_event_study_tear_sheet#03/input-00/call-00"
            ),
        ),
        pytest.param(
            "tests/test_tears.py::TearsTestCase::test_create_event_study_tear_sheet#03/input-01/call-00",
            id="tests/test_tears.py::TearsTestCase::test_create_event_study_tear_sheet#03/input-01/call-00",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_tears.py::TearsTestCase::test_create_event_study_tear_sheet#03/input-01/call-00"
            ),
        ),
        pytest.param(
            "tests/test_tears.py::TearsTestCase::test_create_event_study_tear_sheet#04/input-00/call-00",
            id="tests/test_tears.py::TearsTestCase::test_create_event_study_tear_sheet#04/input-00/call-00",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_tears.py::TearsTestCase::test_create_event_study_tear_sheet#04/input-00/call-00"
            ),
        ),
        pytest.param(
            "tests/test_tears.py::TearsTestCase::test_create_event_study_tear_sheet#04/input-01/call-00",
            id="tests/test_tears.py::TearsTestCase::test_create_event_study_tear_sheet#04/input-01/call-00",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_tears.py::TearsTestCase::test_create_event_study_tear_sheet#04/input-01/call-00"
            ),
        ),
        pytest.param(
            "tests/test_tears.py::TearsTestCase::test_create_event_study_tear_sheet#05/input-00/call-00",
            id="tests/test_tears.py::TearsTestCase::test_create_event_study_tear_sheet#05/input-00/call-00",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_tears.py::TearsTestCase::test_create_event_study_tear_sheet#05/input-00/call-00"
            ),
        ),
        pytest.param(
            "tests/test_tears.py::TearsTestCase::test_create_event_study_tear_sheet#05/input-01/call-00",
            id="tests/test_tears.py::TearsTestCase::test_create_event_study_tear_sheet#05/input-01/call-00",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_tears.py::TearsTestCase::test_create_event_study_tear_sheet#05/input-01/call-00"
            ),
        ),
    ],
)
def test_create_event_study_tear_sheet_upstream_invocation(
    source_invocation_id: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rebuild every frozen event-study input path as a C4 artifact workflow."""

    from fincore.factor_analysis.tears import close_owned_figures, create_event_study_tear_sheet

    model, options = _event_model_for_invocation(source_invocation_id, study=True)
    assert_event_source_calendar(model, options)
    pyplot = _pyplot()
    calls: list[object] = []
    monkeypatch.setattr(pyplot, "show", lambda *args, **kwargs: calls.append((args, kwargs)))

    artifacts = create_event_study_tear_sheet(
        model,
        avgretplot=cast("tuple[int, int]", options["avgretplot"]),
        n_bars=50,
        show=True,
    )
    assert_figure_artifacts(
        artifacts,
        expected_figures=3,
        required_tables=frozenset(("event_returns.event_average", "event_returns.event_windows")),
    )
    assert_show_called(calls, artifacts, expected_show=3)
    assert_artifact_ownership(artifacts)
    assert_event_snapshot_values(artifacts, model)
    assert model.event_returns is not None
    close_owned_figures(artifacts)
    assert_no_open_figures(artifacts)
    assert_strict_event_source_invocation(
        source_invocation_id,
        model,
        options,
        monkeypatch,
        study=True,
    )


@pytest.mark.parametrize(
    "avgretplot",
    (
        (np.int64(1), np.int64(2)),
        (-1, 2),
        (1, -1),
    ),
)
def test_strict_event_workflows_preserve_source_integral_window_grammar(
    avgretplot: tuple[object, object],
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Strict event sheets accept the signed/NumPy offsets used by source."""

    from fincore.alphalens import tears as strict_tears
    from tests.compat.alphalens.conftest import _shared_clean_factor_data, _shared_inputs

    pyplot = _pyplot()
    monkeypatch.setattr(pyplot, "show", lambda *args, **kwargs: None)
    factor_data = _shared_clean_factor_data()
    returns = _shared_inputs()[1]

    assert (
        strict_tears.create_event_returns_tear_sheet(
            factor_data,
            returns,
            avgretplot=avgretplot,
        )
        is None
    )
    assert (
        strict_tears.create_event_study_tear_sheet(
            factor_data,
            returns,
            avgretplot=avgretplot,
        )
        is None
    )
    assert "series = " in capsys.readouterr().out


@pytest.mark.parametrize("window", ((np.int64(1), np.int64(2)), (-1, 2), (1, -1)))
def test_strict_event_performance_accepts_source_integral_window_grammar(
    window: tuple[object, object],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The strict numerical projection shares the tear-sheet window grammar."""

    from fincore.alphalens.performance import average_cumulative_return_by_quantile
    from tests.compat.alphalens.conftest import _shared_clean_factor_data, _shared_inputs

    result = average_cumulative_return_by_quantile(
        _shared_clean_factor_data(),
        _shared_inputs()[1],
        periods_before=cast("int", window[0]),
        periods_after=cast("int", window[1]),
    )
    assert isinstance(result, pd.DataFrame)
    assert "series = " in capsys.readouterr().out
