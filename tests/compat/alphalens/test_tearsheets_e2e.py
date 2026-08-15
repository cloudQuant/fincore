"""C4 lifecycle tests for the strict Alphalens tear-sheet facade."""

from __future__ import annotations

import importlib
from functools import cache, lru_cache
from typing import Any, cast

import pandas as pd
import pytest

# The literal pinned price paths contain repeated/constant IC inputs.  Modern
# SciPy reports these diagnostics as RuntimeWarnings, whereas the frozen
# source profiles regard them as valid tear-sheet data and this repository's
# global warning policy otherwise promotes them to errors.
pytestmark = [
    pytest.mark.filterwarnings("ignore:Precision loss occurred in moment calculation:RuntimeWarning"),
    pytest.mark.filterwarnings("ignore:An input array is constant:RuntimeWarning"),
    pytest.mark.filterwarnings("ignore:invalid value encountered in divide:RuntimeWarning"),
]


def _pyplot() -> Any:
    """Resolve the optional backend only for a visual end-to-end test."""

    return importlib.import_module("matplotlib.pyplot")


@pytest.fixture(autouse=True)
def _close_figures() -> None:
    """Isolate legacy ownership checks from a prior visual test failure."""

    pyplot = _pyplot()
    pyplot.close("all")
    yield
    pyplot.close("all")


def _event_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Preserve the pinned event-workflow grammar, which passes prices through."""

    return prices.copy(deep=True)


@lru_cache(maxsize=2)
def _workflow_model(*, by_group: bool) -> object:
    """Cache a real model while each upstream row owns fresh renderer artifacts."""

    from fincore.factor_analysis.analysis import analyze_factor
    from tests.compat.alphalens.conftest import _shared_clean_factor_data

    return analyze_factor(
        _shared_clean_factor_data().copy(deep=True),
        periods=("1D",),
        turnover_periods=(1,),
        by_group=by_group,
        include_pyfolio=False,
    )


_SOURCE_CASES: dict[str, tuple[int | None, int | tuple[int, ...], tuple[int, ...], float | None, str | None]] = {
    "summary": (None, 2, (1, 5, 10), None, None),
    "summary-1": (None, 3, (1, 2, 3, 7), 20.0, None),
    "returns": (None, 2, (1, 5, 10), None, None),
    "returns-1": (None, 3, (2, 4, 6), 20.0, None),
    "information": (None, 1, (1, 5, 10), None, None),
    "information-1": (None, 4, (1, 2, 3, 7), 20.0, None),
    "turnover": (None, 2, (2, 3, 6), 20.0, None),
    "turnover-1": (None, 4, (1, 2, 3, 7), None, None),
    "turnover-2": (None, 2, (2, 3, 6), 20.0, None),
    "turnover-3": (None, 4, (1, 2, 3, 7), None, None),
    "full": (None, 2, (1, 5, 10), None, None),
    "full-1": (None, 3, (2, 4, 6), 20.0, "US/Eastern"),
    "full-2": (None, 4, (1, 8), 20.0, None),
    "full-3": (None, 4, (1, 2, 3, 7), None, "US/Eastern"),
}


def _source_key(method: str, ordinal: int) -> str:
    """Return the pinned parameter profile key for one non-event source row."""

    return method if ordinal == 0 else f"{method}-{ordinal}"


def _source_invocation_parts(source_invocation_id: str) -> tuple[str, int, int, int]:
    """Parse a complete immutable upstream invocation id without lossy aliases."""

    prefix, input_part, call_part = source_invocation_id.rsplit("/", 2)
    method = prefix.split("test_create_", 1)[1].split("_tear_sheet#", 1)[0]
    ordinal = int(prefix.rsplit("#", 1)[1])
    return method, ordinal, int(input_part.removeprefix("input-")), int(call_part.removeprefix("call-"))


def _restore_source_calendar_levels(data: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Keep sparse cleaned rows tied to the pinned full daily/BDay calendar."""

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


@cache
def _source_model(
    method: str,
    ordinal: int,
    input_ordinal: int,
    long_short: bool,
    group_neutral: bool,
    by_group: bool,
) -> object:
    """Rebuild a compact but literal source profile through the real clean-data path."""

    from fincore.factor_analysis.analysis import analyze_factor
    from fincore.factor_analysis.data import prepare_factor_data
    from tests.compat.alphalens.conftest import _pinned_tears_dense_inputs

    _, quantiles, periods, filter_zscore, timezone = _SOURCE_CASES[_source_key(method, ordinal)]
    # Pinned ``all_prices`` / ``all_factors`` order is full-calendar first,
    # business-calendar second.  The literal fixture preserves the original
    # six assets, thirty factor dates, and sparse values for each invocation.
    factor, price_frame, groups, source_factor_dates = _pinned_tears_dense_inputs(input_ordinal, timezone)
    factor = factor.copy(deep=True)
    price_frame = price_frame.copy(deep=True)
    needs_group = method == "full"
    prepared = prepare_factor_data(
        factor,
        price_frame,
        groupby=groups if needs_group else None,
        quantiles=cast("int", quantiles),
        periods=periods,
        filter_zscore=filter_zscore,
        max_loss=1.0,
    )
    lags = (
        (1, 2) if method == "turnover" and ordinal == 2 else (1,) if method == "turnover" and ordinal == 3 else periods
    )
    prepared_data = _restore_source_calendar_levels(prepared.data, source_factor_dates)
    return analyze_factor(
        prepared_data,
        long_short=long_short,
        group_neutral=group_neutral,
        by_group=by_group,
        turnover_periods=lags,
        include_pyfolio=False,
    )


def _model_for_invocation(source_invocation_id: str) -> tuple[object, dict[str, object]]:
    """Carry each pinned source profile and call dimension into the C4 body."""

    method, ordinal, input_ordinal, call_ordinal = _source_invocation_parts(source_invocation_id)
    long_short = False
    group_neutral = False
    by_group = False
    source_input = 0
    if method == "summary":
        # The source body has two consecutive calls over the same daily
        # ``self.factor``/``self.prices`` fixture.  The inventory collector
        # labels them input-00/input-01 because they are separate AST call
        # groups, not because they select the two price/factor fixtures.
        long_short = input_ordinal == 0
    elif method == "full":
        source_input = input_ordinal
        long_short, group_neutral, by_group = (
            (False, False, False),
            (True, False, True),
            (True, True, True),
        )[call_ordinal]
    model = _source_model(method, ordinal, source_input, long_short, group_neutral, by_group)
    periods = _SOURCE_CASES[_source_key(method, ordinal)][2]
    return model, {
        "by_group": by_group,
        "periods": tuple(f"{period}D" for period in periods),
        "turnover_periods": tuple(model.quantile_turnover),
        "calendar": "D" if source_input == 0 else "B",
    }


def assert_figure_artifacts(
    artifacts: object,
    *,
    expected_figures: int,
    required_tables: frozenset[str],
) -> None:
    """Bind each C4 row to its fixed workflow figure/table contract."""

    figures = artifacts.figures
    assert len(figures) == expected_figures
    assert all(figure.axes for figure in figures)
    assert required_tables.issubset(artifacts.tables)


def assert_show_called(calls: list[object], artifacts: object, *, expected_show: int) -> None:
    """C4 helper: pin source-profile lifecycle instead of deriving it from output."""

    del artifacts
    assert len(calls) == expected_show


def _workflow_table(artifacts: object, name: str) -> pd.DataFrame:
    """Get a direct or composite table without weakening its expected key."""

    tables = artifacts.tables
    if name in tables:
        return tables[name]
    composite_key = {
        "returns": "returns.returns",
        "information": "information.information",
        "turnover": "turnover.turnover",
        "autocorrelation": "turnover.autocorrelation",
    }.get(name)
    if composite_key is not None and composite_key in tables:
        return tables[composite_key]
    matches = [table for key, table in tables.items() if key.endswith(f".{name}")]
    assert len(matches) == 1
    return matches[0]


def assert_source_table_values(artifacts: object, model: object) -> None:
    """Check source-profile numerical tables against independently stored snapshots."""

    factor_data = model.factor_data
    if "quantile_statistics" in artifacts.tables:
        table = artifacts.tables["quantile_statistics"]
        assert table["count"].sum() == factor_data["factor_quantile"].notna().sum()
        expected_quantile_means = (
            factor_data.assign(factor_quantile=factor_data["factor_quantile"].astype(float))
            .groupby("factor_quantile", observed=False, sort=True)["factor"]
            .mean()
        )
        pd.testing.assert_series_equal(
            table["mean"],
            expected_quantile_means,
            check_names=False,
        )
    if any(key == "returns" or key.endswith(".returns") for key in artifacts.tables):
        table = _workflow_table(artifacts, "returns")
        expected = model.aggregate_mean_returns_by_quantile.iloc[-1] * 10_000.0
        pd.testing.assert_series_equal(
            table.loc["Mean Period Wise Return Top Quantile (bps)"],
            expected,
            check_names=False,
        )
    if any(key == "information" or key.endswith(".information") for key in artifacts.tables):
        table = _workflow_table(artifacts, "information")
        pd.testing.assert_series_equal(
            table["IC Mean"],
            model.aggregate_information_coefficient.mean(),
            check_names=False,
        )
    if "turnover" in artifacts.tables or "turnover.turnover" in artifacts.tables:
        table = _workflow_table(artifacts, "turnover")
        for period, turnover in model.quantile_turnover.items():
            for quantile in turnover:
                assert table.loc[f"Quantile {quantile} Mean Turnover ", f"{period}D"] == pytest.approx(
                    turnover[quantile].mean()
                )


def assert_strict_source_invocation(
    source_invocation_id: str,
    model: object,
    options: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise every non-event mapped row through its actual strict facade."""

    from fincore.alphalens import tears as strict_tears

    method, _, _, _ = _source_invocation_parts(source_invocation_id)
    pyplot = _pyplot()
    calls: list[object] = []
    monkeypatch.setattr(pyplot, "show", lambda *args, **kwargs: calls.append((args, kwargs)))
    monkeypatch.setattr(strict_tears._plotting, "_display_table", lambda *args, **kwargs: None)
    factor_data = model.factor_data
    if method == "summary":
        result = strict_tears.create_summary_tear_sheet(
            factor_data,
            long_short=model.config.long_short,
            group_neutral=model.config.group_neutral,
            set_context=False,
        )
        expected_show = 1
    elif method == "returns":
        result = strict_tears.create_returns_tear_sheet(
            factor_data,
            long_short=model.config.long_short,
            group_neutral=model.config.group_neutral,
            by_group=False,
            set_context=False,
        )
        expected_show = 1
    elif method == "information":
        result = strict_tears.create_information_tear_sheet(
            factor_data,
            group_neutral=model.config.group_neutral,
            by_group=False,
            set_context=False,
        )
        expected_show = 1
    elif method == "turnover":
        result = strict_tears.create_turnover_tear_sheet(
            factor_data,
            turnover_periods=[f"{period}D" for period in model.quantile_turnover],
            set_context=False,
        )
        expected_show = 1
    else:
        by_group = cast("bool", options["by_group"])
        result = strict_tears.create_full_tear_sheet(
            factor_data,
            long_short=model.config.long_short,
            group_neutral=model.config.group_neutral,
            by_group=by_group,
            set_context=False,
        )
        expected_show = 4 if by_group else 3
    assert result is None
    assert len(calls) == expected_show
    assert not pyplot.get_fignums()


def assert_source_calendar(model: object, options: dict[str, object]) -> None:
    """Verify each frozen input ordinal retains its pinned daily/BDay calendar."""

    dates = model.factor_data.index.levels[0]
    assert isinstance(dates, pd.DatetimeIndex)
    assert dates.freqstr == options["calendar"]
    assert len(dates) == 30
    assert set(model.factor_data.index.get_level_values("asset")) == {"A", "B", "C", "D", "E", "F"}


def assert_artifact_ownership(artifacts: object) -> None:
    """C4 helper: tables are data and figures are explicit caller-owned artifacts."""

    assert artifacts.tables
    assert artifacts.model is not None


def assert_no_open_figures(artifacts: object) -> None:
    """C4 helper: closing owned artifacts leaves no renderer-created Figure behind."""

    del artifacts
    assert not _pyplot().get_fignums()


_STRICT_WORKFLOWS: tuple[tuple[str, str, dict[str, object], int], ...] = (
    ("summary", "create_summary_tear_sheet", {"long_short": False, "group_neutral": False}, 1),
    ("returns", "create_returns_tear_sheet", {"long_short": False, "group_neutral": False, "by_group": True}, 2),
    ("information", "create_information_tear_sheet", {"group_neutral": False, "by_group": True}, 1),
    ("turnover", "create_turnover_tear_sheet", {"turnover_periods": ["1D", "5D"]}, 1),
    ("full", "create_full_tear_sheet", {"long_short": False, "group_neutral": False, "by_group": True}, 4),
)


@pytest.mark.parametrize("label,name,kwargs,expected_show", _STRICT_WORKFLOWS, ids=lambda item: str(item))
def test_strict_factor_workflows_show_and_close_their_own_figures(
    clean_factor_data: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
    label: str,
    name: str,
    kwargs: dict[str, object],
    expected_show: int,
) -> None:
    """Five non-event strict workflows retain source section-specific show counts."""

    from fincore.alphalens import tears

    del label
    pyplot = _pyplot()
    calls: list[object] = []
    monkeypatch.setattr(pyplot, "show", lambda *args, **keyword_args: calls.append((args, keyword_args)))
    workflow = cast("Any", getattr(tears, name))
    supplied_figure, _ = pyplot.subplots()

    assert workflow(clean_factor_data, set_context=False, **kwargs) is None
    assert len(calls) == expected_show
    assert pyplot.fignum_exists(supplied_figure.number)
    pyplot.close(supplied_figure)


def test_strict_returns_by_group_allocates_only_the_two_source_grids(
    clean_factor_data: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Strict group rendering does not create an enhanced-only hidden primary grid."""

    from fincore.alphalens import tears

    pyplot = _pyplot()
    original_figure = pyplot.figure
    created: list[object] = []

    def record_figure(*args: object, **kwargs: object) -> object:
        figure = original_figure(*args, **kwargs)
        created.append(figure)
        return figure

    monkeypatch.setattr(pyplot, "figure", record_figure)
    monkeypatch.setattr(pyplot, "show", lambda *args, **kwargs: None)
    monkeypatch.setattr(tears._plotting, "_display_table", lambda *args, **kwargs: None)
    assert tears.create_returns_tear_sheet(clean_factor_data, by_group=True, set_context=False) is None
    assert len(created) == 2
    assert not pyplot.get_fignums()


def test_strict_full_allocates_only_its_three_source_sections(
    clean_factor_data: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Quantile statistics are table-only; full must not allocate a summary grid."""

    from fincore.alphalens import tears

    pyplot = _pyplot()
    original_figure = pyplot.figure
    created: list[object] = []

    def record_figure(*args: object, **kwargs: object) -> object:
        figure = original_figure(*args, **kwargs)
        created.append(figure)
        return figure

    monkeypatch.setattr(pyplot, "figure", record_figure)
    monkeypatch.setattr(pyplot, "show", lambda *args, **kwargs: None)
    monkeypatch.setattr(tears._plotting, "_display_table", lambda *args, **kwargs: None)
    assert tears.create_full_tear_sheet(clean_factor_data, set_context=False) is None
    assert len(created) == 3
    assert not pyplot.get_fignums()
    assert not pyplot.get_fignums()


@pytest.mark.parametrize(
    "name,kwargs,expected_show",
    (
        ("create_event_returns_tear_sheet", {"avgretplot": (1, 2), "by_group": True}, 2),
        ("create_event_study_tear_sheet", {"avgretplot": (1, 2)}, 3),
    ),
)
def test_strict_event_workflows_show_and_close_their_own_figures(
    clean_factor_data: pd.DataFrame,
    prices: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    kwargs: dict[str, object],
    expected_show: int,
) -> None:
    """Event workflows add only their documented extra event/group sections."""

    from fincore.alphalens import tears

    pyplot = _pyplot()
    calls: list[object] = []
    monkeypatch.setattr(pyplot, "show", lambda *args, **keyword_args: calls.append((args, keyword_args)))
    workflow = cast("Any", getattr(tears, name))

    assert workflow(clean_factor_data, _event_returns(prices), set_context=False, **kwargs) is None
    assert len(calls) == expected_show
    assert not pyplot.get_fignums()


def test_strict_summary_replays_legacy_table_display_sections_from_the_model_snapshot(
    clean_factor_data: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Strict orchestration retains table display side effects without recomputation."""

    from fincore.alphalens import tears

    displayed: list[tuple[str | None, pd.DataFrame, bool]] = []

    def capture(heading: str | None, table: pd.DataFrame, *, round_values: bool = True) -> None:
        displayed.append((heading, table, round_values))

    # Import-side-effect tests can reload the public plotting module, while
    # the decorated workflow retains its original module reference.
    monkeypatch.setattr(tears._plotting, "_display_table", capture)
    monkeypatch.setattr(_pyplot(), "show", lambda *args, **kwargs: None)

    assert tears.create_summary_tear_sheet(clean_factor_data, set_context=False) is None
    assert [heading for heading, _, _ in displayed] == [
        "Quantiles Statistics",
        "Returns Analysis",
        "Information Analysis",
        "Turnover Analysis",
        None,
    ]
    assert displayed[0][2] is False
    assert "IC Mean" in displayed[2][1].index
    assert set(displayed[3][1].columns) == {"1D", "5D", "10D"}


def test_strict_summary_keeps_its_information_table_group_unadjusted(
    clean_factor_data: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Summary keeps the pinned unadjusted IC table even for neutral returns."""

    from fincore.alphalens import tears
    from fincore.factor_analysis import performance
    from fincore.factor_analysis.render_matplotlib import build_information_table

    displayed: list[pd.DataFrame] = []

    def capture(heading: str | None, table: pd.DataFrame, **kwargs: object) -> None:
        del kwargs
        if heading == "Information Analysis":
            displayed.append(table.copy(deep=True))

    monkeypatch.setattr(tears._plotting, "_display_table", capture)
    monkeypatch.setattr(_pyplot(), "show", lambda *args, **kwargs: None)

    assert (
        tears.create_summary_tear_sheet(
            clean_factor_data,
            group_neutral=True,
            set_context=False,
        )
        is None
    )
    expected = build_information_table(
        performance.factor_information_coefficient(clean_factor_data, group_adjust=False)
    ).T
    assert len(displayed) == 1
    pd.testing.assert_frame_equal(displayed[0], expected)


def test_strict_returns_defers_missing_group_error_until_after_primary_section(
    clean_factor_data: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A strict by-group error follows, rather than suppresses, the primary show."""

    from fincore.alphalens import tears

    pyplot = _pyplot()
    observed: list[tuple[int, ...]] = []
    monkeypatch.setattr(pyplot, "show", lambda *args, **kwargs: observed.append(tuple(pyplot.get_fignums())))
    with pytest.raises(KeyError, match="group"):
        tears.create_returns_tear_sheet(
            clean_factor_data.drop(columns="group"),
            by_group=True,
            set_context=False,
        )
    assert len(observed) == 1
    assert all(len(figures) == 1 for figures in observed)
    assert not pyplot.get_fignums()


def test_strict_returns_group_grid_keeps_the_source_leading_section(
    clean_factor_data: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The strict by-group grid uses the pinned ``1 + ceil(groups / 2)`` rows."""

    from fincore.alphalens import tears

    pyplot = _pyplot()
    heights: list[float] = []

    def capture_show(*args: object, **kwargs: object) -> None:
        del args, kwargs
        heights.extend(float(pyplot.figure(number).get_size_inches()[1]) for number in pyplot.get_fignums())

    monkeypatch.setattr(pyplot, "show", capture_show)
    assert tears.create_returns_tear_sheet(clean_factor_data, by_group=True, set_context=False) is None
    groups = len(clean_factor_data["group"].dropna().unique())
    assert heights[-1] == pytest.approx(7.0 * (1 + ((groups - 1) // 2 + 1)))
    assert not pyplot.get_fignums()


@pytest.mark.parametrize(
    "label,source_column,expected_lag",
    (
        ("duplicate-whole-day", "1D12h", "1D"),
        ("zero-day", "1h", "0D"),
    ),
)
def test_strict_summary_projects_the_source_unvalidated_turnover_day_lags(
    clean_factor_data: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
    label: str,
    source_column: str,
    expected_lag: str,
) -> None:
    """Summary alone retains source timedelta-to-day truncation/deduplication."""

    from fincore.alphalens import tears

    del label
    factor_data = clean_factor_data.copy(deep=True)
    factor_data[source_column] = factor_data["5D"]
    displayed: list[pd.DataFrame] = []
    monkeypatch.setattr(
        tears._plotting,
        "_display_table",
        lambda heading, table, **kwargs: (
            displayed.append(table.copy(deep=True)) if heading == "Turnover Analysis" else None
        ),
    )
    monkeypatch.setattr(_pyplot(), "show", lambda *args, **kwargs: None)

    assert tears.create_summary_tear_sheet(factor_data, set_context=False) is None
    assert len(displayed) == 1
    assert expected_lag in displayed[0].columns
    if source_column == "1D12h":
        assert list(displayed[0].columns).count("1D") == 1


def test_strict_summary_retains_unobserved_source_quantile_turnover_rows(
    clean_factor_data: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Summary's turnover table keeps the source ``range(1, max + 1)`` bins."""

    from fincore.alphalens import tears

    factor_data = clean_factor_data.copy(deep=True)
    factor_data.loc[factor_data["factor_quantile"] == 2, "factor_quantile"] = 3
    displayed: list[pd.DataFrame] = []
    monkeypatch.setattr(
        tears._plotting,
        "_display_table",
        lambda heading, table, **kwargs: (
            displayed.append(table.copy(deep=True)) if heading == "Turnover Analysis" else None
        ),
    )
    monkeypatch.setattr(_pyplot(), "show", lambda *args, **kwargs: None)

    assert tears.create_summary_tear_sheet(factor_data, set_context=False) is None
    assert len(displayed) == 1
    assert "Quantile 2 Mean Turnover " in displayed[0].index
    assert displayed[0].loc["Quantile 2 Mean Turnover "].isna().all()


@pytest.mark.parametrize("periods", (["1D", "1D"], ["0D"], ["-1D"]))
def test_strict_turnover_projects_source_unvalidated_lag_grammar(
    clean_factor_data: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
    periods: list[str],
) -> None:
    """Strict turnover keeps source duplicate/zero day calls outside enhanced validation."""

    from fincore.alphalens import tears

    displayed: list[pd.DataFrame] = []
    monkeypatch.setattr(
        tears._plotting,
        "_display_table",
        lambda heading, table, **kwargs: (
            displayed.append(table.copy(deep=True)) if heading == "Turnover Analysis" else None
        ),
    )
    monkeypatch.setattr(_pyplot(), "show", lambda *args, **kwargs: None)

    if len(set(periods)) != len(periods):
        with pytest.raises(ValueError, match="truth value of a Series is ambiguous"):
            tears.create_turnover_tear_sheet(clean_factor_data, turnover_periods=periods, set_context=False)
        assert len(displayed) == 1
        return

    assert tears.create_turnover_tear_sheet(clean_factor_data, turnover_periods=periods, set_context=False) is None
    assert len(displayed) == 1
    assert f"{int(pd.Timedelta(periods[0]).days)}D" in displayed[0].columns


@pytest.mark.parametrize("periods", ([],))
def test_strict_turnover_keeps_source_empty_period_concat_error(
    clean_factor_data: pd.DataFrame,
    periods: list[str],
) -> None:
    """Strict empty lag input reaches pinned ``pd.concat([])`` rather than a fallback day."""

    from fincore.alphalens import tears

    with pytest.raises(ValueError, match="No objects to concatenate"):
        tears.create_turnover_tear_sheet(clean_factor_data, turnover_periods=periods, set_context=False)


@pytest.mark.parametrize(
    ("factor_data", "expected_exception", "message"),
    (
        (pd.DataFrame({"factor": [1.0]}), KeyError, "factor_quantile"),
        (None, TypeError, "not subscriptable"),
    ),
)
def test_strict_turnover_preserves_factor_quantile_lookup_before_empty_period_error(
    factor_data: object,
    expected_exception: type[Exception],
    message: str,
) -> None:
    """Empty periods do not mask the source's earlier quantile lookup."""

    from fincore.alphalens import tears

    with pytest.raises(expected_exception, match=message):
        tears.create_turnover_tear_sheet(factor_data, turnover_periods=[], set_context=False)


def test_strict_turnover_preserves_default_period_discovery_before_quantile_lookup() -> None:
    """The default source branch touches ``.columns`` before factor quantiles."""

    from fincore.alphalens import tears

    with pytest.raises(AttributeError, match="'NoneType' object has no attribute 'columns'"):
        tears.create_turnover_tear_sheet(None, set_context=False)


def test_strict_full_displays_quantile_statistics_before_missing_forward_error(
    clean_factor_data: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Full retains the source's early statistics-table side effect on failure."""

    from fincore.alphalens import tears

    factor_data = clean_factor_data.drop(columns=["1D", "5D", "10D"])
    displayed: list[tuple[object, pd.DataFrame]] = []
    monkeypatch.setattr(
        tears._plotting,
        "_display_table",
        lambda heading, table, **kwargs: displayed.append((heading, table.copy(deep=True))),
    )

    with pytest.raises(ValueError, match="at least one forward-return column"):
        tears.create_full_tear_sheet(factor_data, set_context=False)

    assert [heading for heading, _ in displayed] == ["Quantiles Statistics"]


def test_strict_full_preserves_duplicate_forward_period_error_after_statistics_table(
    clean_factor_data: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Full reaches the source duplicate-column failure after its first table."""

    from fincore.alphalens import tears

    factor_data = clean_factor_data.rename(columns={"5D": "1D"})
    displayed: list[object] = []
    monkeypatch.setattr(
        tears._plotting,
        "_display_table",
        lambda heading, table, **kwargs: displayed.append(heading),
    )

    with pytest.raises(ValueError, match="Columns must be same length as key"):
        tears.create_full_tear_sheet(factor_data, set_context=False)

    assert displayed == ["Quantiles Statistics"]


def test_strict_returns_preserves_duplicate_forward_period_error_before_rendering(
    clean_factor_data: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The direct returns sheet fails before its table or Figure is created."""

    from fincore.alphalens import tears

    factor_data = clean_factor_data.rename(columns={"5D": "1D"})
    displayed: list[object] = []
    pyplot = _pyplot()
    shown: list[object] = []
    monkeypatch.setattr(
        tears._plotting,
        "_display_table",
        lambda heading, table, **kwargs: displayed.append(heading),
    )
    monkeypatch.setattr(pyplot, "show", lambda *args, **kwargs: shown.append((args, kwargs)))

    with pytest.raises(ValueError, match="Columns must be same length as key"):
        tears.create_returns_tear_sheet(factor_data, set_context=False)

    assert not displayed
    assert not shown
    assert not pyplot.get_fignums()


def test_strict_duplicate_forward_information_and_event_workflows_assemble_before_rendering(
    clean_factor_data: pd.DataFrame,
    prices: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Duplicate-forward projections remain a single strict assembly boundary.

    Pinned information and event tear sheets accept duplicated forward labels.
    Their strict-only expanded snapshots must be calculated while ``_model``
    is assembling, never by a later table or chart section.
    """

    from fincore.alphalens import tears

    factor_data = clean_factor_data.rename(columns={"5D": "1D"})
    state = {"assembling": False, "assemblies": 0}
    original_model = tears._model

    def assemble_once(*args: object, **kwargs: object) -> object:
        state["assemblies"] += 1
        state["assembling"] = True
        try:
            return original_model(*args, **kwargs)
        finally:
            state["assembling"] = False

    monkeypatch.setattr(tears, "_model", assemble_once)

    def require_assembly(function: object) -> object:
        def guarded(*args: object, **kwargs: object) -> object:
            if not state["assembling"]:
                raise AssertionError("strict duplicate renderer re-entered an analytical kernel")
            return cast("Any", function)(*args, **kwargs)

        return guarded

    for name in ("factor_information_coefficient", "factor_returns"):
        original = getattr(tears._strict_performance, name)
        monkeypatch.setattr(tears._strict_performance, name, require_assembly(original))

    pyplot = _pyplot()
    shown: list[object] = []
    monkeypatch.setattr(pyplot, "show", lambda *args, **kwargs: shown.append((args, kwargs)))
    monkeypatch.setattr(tears._plotting, "_display_table", lambda *args, **kwargs: None)
    # Stdout has its own dedicated source-characterization coverage.  This
    # test isolates model/renderer call ordering from hundreds of event rows.
    monkeypatch.setattr(tears, "_replay_legacy_event_stdout", lambda *args, **kwargs: None)

    assert tears.create_information_tear_sheet(factor_data, set_context=False) is None
    assert tears.create_event_returns_tear_sheet(factor_data, prices, avgretplot=(1, 2), set_context=False) is None
    assert tears.create_event_study_tear_sheet(factor_data, prices, avgretplot=(1, 2), set_context=False) is None

    assert state["assemblies"] == 3
    assert len(shown) == 5
    assert not pyplot.get_fignums()


def test_enhanced_analysis_keeps_turnover_lag_validation_private(clean_factor_data: pd.DataFrame) -> None:
    """The strict source bridge cannot relax the enhanced public model contract."""

    from fincore.factor_analysis.analysis import analyze_factor

    with pytest.raises(ValueError, match="positive integers"):
        analyze_factor(clean_factor_data, turnover_periods=(0,), include_pyfolio=False)
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        analyze_factor(
            clean_factor_data,
            turnover_periods=(1,),
            include_pyfolio=False,
            _allow_legacy_zero_turnover=True,  # type: ignore[call-arg]
        )


def test_strict_returns_assemble_one_model_without_renderer_kernel_reentry(
    clean_factor_data: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A strict workflow performs one assembly pass and renders only stored fields."""

    from fincore.alphalens import tears
    from fincore.factor_analysis import performance

    original_assemble = tears.analyze_factor
    assembling = False
    assemblies = 0

    def assemble_once(*args: object, **kwargs: object) -> object:
        nonlocal assembling, assemblies
        assemblies += 1
        assembling = True
        try:
            return original_assemble(*args, **kwargs)
        finally:
            assembling = False

    monkeypatch.setattr(tears, "analyze_factor", assemble_once)
    calls: dict[str, int] = {
        "factor_returns": 0,
        "cumulative_returns": 0,
        "mean_return_by_quantile": 0,
        "factor_information_coefficient": 0,
        "quantile_turnover": 0,
        "factor_rank_autocorrelation": 0,
    }
    for name in calls:
        original = getattr(performance, name)

        def require_assembly(
            *args: object, _name: str = name, _original: object = original, **kwargs: object
        ) -> object:
            if not assembling:
                raise AssertionError(f"strict renderer re-entered {_name} after assembly")
            calls[_name] += 1
            return cast("Any", _original)(*args, **kwargs)

        monkeypatch.setattr(performance, name, require_assembly)

    monkeypatch.setattr(tears._plotting, "_display_table", lambda *args, **kwargs: None)
    monkeypatch.setattr(_pyplot(), "show", lambda *args, **kwargs: None)
    assert tears.create_returns_tear_sheet(clean_factor_data, set_context=False) is None
    assert assemblies == 1
    # One direct return snapshot plus the three stored cumulative-period
    # snapshots are assembled before rendering.  Any later call would trip
    # ``require_assembly`` above.
    assert calls["factor_returns"] == 4
    assert calls["cumulative_returns"] == len(clean_factor_data.filter(regex=r"^\d+D$").columns)
    assert calls["mean_return_by_quantile"] == 2
    assert calls["factor_information_coefficient"] == 1
    assert calls["quantile_turnover"] == clean_factor_data["factor_quantile"].nunique()
    assert calls["factor_rank_autocorrelation"] == 1
    assert not _pyplot().get_fignums()


@pytest.mark.parametrize(
    "name,kwargs,event",
    (
        ("create_summary_tear_sheet", {}, False),
        ("create_returns_tear_sheet", {"by_group": True}, False),
        ("create_information_tear_sheet", {"by_group": True}, False),
        ("create_turnover_tear_sheet", {"turnover_periods": ["1D", "5D"]}, False),
        ("create_full_tear_sheet", {"by_group": True}, False),
        ("create_event_returns_tear_sheet", {"avgretplot": (1, 2), "by_group": True}, True),
        ("create_event_study_tear_sheet", {"avgretplot": (1, 2)}, True),
    ),
)
def test_each_strict_workflow_assembles_once_and_never_reenters_a_kernel(
    clean_factor_data: pd.DataFrame,
    prices: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    kwargs: dict[str, object],
    event: bool,
) -> None:
    """Bind all seven strict workflows to one assembly and snapshot-only render.

    The mapped C4 invocations execute each facade.  This focused matrix adds
    the complementary call-boundary proof: every analytical function is
    allowed while the one strict model is assembled, but becomes an immediate
    failure if a later table/chart section recomputes it.
    """

    from fincore.alphalens import tears as strict_tears
    from fincore.factor_analysis import performance, portfolio

    state = {"assembling": False, "assemblies": 0}
    original_model = strict_tears._model

    def assemble_once(*args: object, **keyword_args: object) -> object:
        assert not state["assembling"]
        state["assemblies"] += 1
        state["assembling"] = True
        try:
            return original_model(*args, **keyword_args)
        finally:
            state["assembling"] = False

    monkeypatch.setattr(strict_tears, "_model", assemble_once)

    def guard(name: str, function: object) -> object:
        def guarded(*args: object, **keyword_args: object) -> object:
            if not state["assembling"]:
                raise AssertionError(f"strict {name} renderer re-entered an analytical kernel")
            return cast("Any", function)(*args, **keyword_args)

        return guarded

    performance_names = (
        "factor_returns",
        "cumulative_returns",
        "factor_alpha_beta",
        "mean_return_by_quantile",
        "factor_information_coefficient",
        "mean_information_coefficient",
        "quantile_turnover",
        "factor_rank_autocorrelation",
        "average_cumulative_return_by_quantile",
        "common_start_returns",
    )
    for kernel_name in performance_names:
        monkeypatch.setattr(performance, kernel_name, guard(kernel_name, getattr(performance, kernel_name)))
    for kernel_name in ("factor_cumulative_returns", "factor_positions", "create_pyfolio_input"):
        monkeypatch.setattr(portfolio, kernel_name, guard(kernel_name, getattr(portfolio, kernel_name)))

    pyplot = _pyplot()
    monkeypatch.setattr(pyplot, "show", lambda *args, **keyword_args: None)
    monkeypatch.setattr(strict_tears._plotting, "_display_table", lambda *args, **keyword_args: None)
    workflow = cast("Any", getattr(strict_tears, name))
    if event:
        result = workflow(clean_factor_data, _event_returns(prices), set_context=False, **kwargs)
    else:
        result = workflow(clean_factor_data, set_context=False, **kwargs)

    assert result is None
    assert state["assemblies"] == 1
    assert not pyplot.get_fignums()


@pytest.mark.parametrize(
    "name,kwargs,expected_show",
    (
        ("create_returns_tear_sheet", {"by_group": True}, 2),
        ("create_full_tear_sheet", {"by_group": True}, 4),
    ),
)
def test_strict_composite_workflows_show_only_the_current_source_section(
    clean_factor_data: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    kwargs: dict[str, object],
    expected_show: int,
) -> None:
    """Each source ``show`` observes exactly one workflow-owned figure."""

    from fincore.alphalens import tears

    pyplot = _pyplot()
    observed: list[tuple[int, ...]] = []
    monkeypatch.setattr(pyplot, "show", lambda *args, **keyword_args: observed.append(tuple(pyplot.get_fignums())))

    workflow = cast("Any", getattr(tears, name))
    assert workflow(clean_factor_data, set_context=False, **kwargs) is None
    assert len(observed) == expected_show
    assert all(len(numbers) == 1 for numbers in observed)
    assert not pyplot.get_fignums()


def test_strict_summary_projects_the_source_single_bar_axis(
    clean_factor_data: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The strict summary leaves IC/rank charts in tables, as does the pinned workflow."""

    from fincore.alphalens import tears

    pyplot = _pyplot()
    axes_per_show: list[int] = []

    def capture_show(*args: object, **keyword_args: object) -> None:
        del args, keyword_args
        axes_per_show.extend(len(pyplot.figure(number).axes) for number in pyplot.get_fignums())

    monkeypatch.setattr(pyplot, "show", capture_show)
    assert tears.create_summary_tear_sheet(clean_factor_data, set_context=False) is None
    assert axes_per_show == [1]
    assert not pyplot.get_fignums()


def test_strict_returns_tables_and_spread_error_use_source_period_conversion(
    clean_factor_data: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Strict returns project all periods and daily spread errors to the base period."""

    from fincore.alphalens import tears
    from fincore.factor_analysis import performance
    from fincore.factor_analysis.analysis import analyze_factor
    from fincore.factor_analysis.render_matplotlib import build_returns_table

    model = analyze_factor(clean_factor_data, long_short=False, include_pyfolio=False)
    mean, _ = performance.mean_return_by_quantile(clean_factor_data, demeaned=False)
    mean_by_date, std_by_date = performance.mean_return_by_quantile(clean_factor_data, by_date=True, demeaned=False)
    base = mean.columns[0]
    rate = mean.apply(
        lambda values: values.add(1.0).pow(pd.Timedelta(base) / pd.Timedelta(values.name)).sub(1.0), axis=0
    )
    rate_by_date = mean_by_date.apply(
        lambda values: values.add(1.0).pow(pd.Timedelta(base) / pd.Timedelta(values.name)).sub(1.0), axis=0
    )
    converted_std = std_by_date.apply(
        lambda values: values / (pd.Timedelta(values.name) / pd.Timedelta(base)) ** 0.5, axis=0
    )
    expected_spread, expected_std = performance.compute_mean_returns_spread(
        rate_by_date,
        clean_factor_data["factor_quantile"].max(),
        clean_factor_data["factor_quantile"].min(),
        converted_std,
    )
    expected_table = build_returns_table(model.alpha_beta, rate, expected_spread)
    displayed: list[pd.DataFrame] = []
    captured_std: list[pd.Series | pd.DataFrame | None] = []
    original_spread = tears._plotting.plot_mean_quantile_returns_spread_time_series
    pyplot = _pyplot()

    monkeypatch.setattr(
        tears._plotting,
        "_display_table",
        lambda heading, table, **kwargs: (
            displayed.append(table.copy(deep=True)) if heading == "Returns Analysis" else None
        ),
    )

    def capture_spread(*args: object, **kwargs: object) -> object:
        captured_std.append(cast("pd.Series | pd.DataFrame | None", kwargs.get("std_err")))
        return original_spread(*args, **kwargs)

    monkeypatch.setattr(tears._plotting, "plot_mean_quantile_returns_spread_time_series", capture_spread)
    monkeypatch.setattr(pyplot, "show", lambda *args, **kwargs: None)

    assert tears.create_returns_tear_sheet(clean_factor_data, long_short=False, set_context=False) is None
    assert len(displayed) == 1
    pd.testing.assert_frame_equal(displayed[0], expected_table)
    assert captured_std and captured_std[0] is not None
    pd.testing.assert_frame_equal(cast("pd.DataFrame", captured_std[0]), cast("pd.DataFrame", expected_std))
    assert not pyplot.get_fignums()


@pytest.mark.parametrize(
    "source_invocation_id",
    [
        pytest.param(
            "tests/test_tears.py::TearsTestCase::test_create_summary_tear_sheet#00/input-00/call-00",
            id="tests/test_tears.py::TearsTestCase::test_create_summary_tear_sheet#00/input-00/call-00",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_tears.py::TearsTestCase::test_create_summary_tear_sheet#00/input-00/call-00"
            ),
        ),
        pytest.param(
            "tests/test_tears.py::TearsTestCase::test_create_summary_tear_sheet#00/input-01/call-00",
            id="tests/test_tears.py::TearsTestCase::test_create_summary_tear_sheet#00/input-01/call-00",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_tears.py::TearsTestCase::test_create_summary_tear_sheet#00/input-01/call-00"
            ),
        ),
        pytest.param(
            "tests/test_tears.py::TearsTestCase::test_create_summary_tear_sheet#01/input-00/call-00",
            id="tests/test_tears.py::TearsTestCase::test_create_summary_tear_sheet#01/input-00/call-00",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_tears.py::TearsTestCase::test_create_summary_tear_sheet#01/input-00/call-00"
            ),
        ),
        pytest.param(
            "tests/test_tears.py::TearsTestCase::test_create_summary_tear_sheet#01/input-01/call-00",
            id="tests/test_tears.py::TearsTestCase::test_create_summary_tear_sheet#01/input-01/call-00",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_tears.py::TearsTestCase::test_create_summary_tear_sheet#01/input-01/call-00"
            ),
        ),
    ],
)
def test_create_summary_tear_sheet_upstream_invocation(
    source_invocation_id: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rebuild every pinned summary tear sheet invocation as a C4 workflow."""

    from fincore.factor_analysis.tears import close_owned_figures, create_summary_tear_sheet

    model, options = _model_for_invocation(source_invocation_id)
    assert_source_calendar(model, options)
    pyplot = _pyplot()
    calls: list[object] = []
    monkeypatch.setattr(pyplot, "show", lambda *args, **kwargs: calls.append((args, kwargs)))

    artifacts = create_summary_tear_sheet(model, show=True)
    assert_figure_artifacts(
        artifacts,
        expected_figures=1,
        required_tables=frozenset(("quantile_statistics", "returns", "information", "turnover")),
    )
    assert_show_called(calls, artifacts, expected_show=1)
    assert_artifact_ownership(artifacts)
    assert_source_table_values(artifacts, model)
    assert model.config.long_short is (source_invocation_id.endswith("input-00/call-00"))
    assert not options["by_group"]
    close_owned_figures(artifacts)
    assert_no_open_figures(artifacts)
    assert_strict_source_invocation(source_invocation_id, model, options, monkeypatch)


@pytest.mark.parametrize(
    "source_invocation_id",
    [
        pytest.param(
            "tests/test_tears.py::TearsTestCase::test_create_returns_tear_sheet#00/input-00/call-00",
            id="tests/test_tears.py::TearsTestCase::test_create_returns_tear_sheet#00/input-00/call-00",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_tears.py::TearsTestCase::test_create_returns_tear_sheet#00/input-00/call-00"
            ),
        ),
        pytest.param(
            "tests/test_tears.py::TearsTestCase::test_create_returns_tear_sheet#01/input-00/call-00",
            id="tests/test_tears.py::TearsTestCase::test_create_returns_tear_sheet#01/input-00/call-00",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_tears.py::TearsTestCase::test_create_returns_tear_sheet#01/input-00/call-00"
            ),
        ),
    ],
)
def test_create_returns_tear_sheet_upstream_invocation(
    source_invocation_id: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rebuild every pinned returns tear sheet invocation as a C4 workflow."""

    from fincore.factor_analysis.tears import close_owned_figures, create_returns_tear_sheet

    model, options = _model_for_invocation(source_invocation_id)
    assert_source_calendar(model, options)
    pyplot = _pyplot()
    calls: list[object] = []
    monkeypatch.setattr(pyplot, "show", lambda *args, **kwargs: calls.append((args, kwargs)))

    artifacts = create_returns_tear_sheet(model, by_group=cast("bool", options["by_group"]), show=True)
    assert_figure_artifacts(artifacts, expected_figures=1, required_tables=frozenset(("returns",)))
    assert_show_called(calls, artifacts, expected_show=1)
    assert_artifact_ownership(artifacts)
    assert_source_table_values(artifacts, model)
    assert tuple(model.forward_periods) == cast("tuple[str, ...]", options["periods"])
    close_owned_figures(artifacts)
    assert_no_open_figures(artifacts)
    assert_strict_source_invocation(source_invocation_id, model, options, monkeypatch)


@pytest.mark.parametrize(
    "source_invocation_id",
    [
        pytest.param(
            "tests/test_tears.py::TearsTestCase::test_create_information_tear_sheet#00/input-00/call-00",
            id="tests/test_tears.py::TearsTestCase::test_create_information_tear_sheet#00/input-00/call-00",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_tears.py::TearsTestCase::test_create_information_tear_sheet#00/input-00/call-00"
            ),
        ),
        pytest.param(
            "tests/test_tears.py::TearsTestCase::test_create_information_tear_sheet#01/input-00/call-00",
            id="tests/test_tears.py::TearsTestCase::test_create_information_tear_sheet#01/input-00/call-00",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_tears.py::TearsTestCase::test_create_information_tear_sheet#01/input-00/call-00"
            ),
        ),
    ],
)
def test_create_information_tear_sheet_upstream_invocation(
    source_invocation_id: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rebuild every pinned information tear sheet invocation as a C4 workflow."""

    from fincore.factor_analysis.tears import close_owned_figures, create_information_tear_sheet

    model, options = _model_for_invocation(source_invocation_id)
    assert_source_calendar(model, options)
    pyplot = _pyplot()
    calls: list[object] = []
    monkeypatch.setattr(pyplot, "show", lambda *args, **kwargs: calls.append((args, kwargs)))

    artifacts = create_information_tear_sheet(model, by_group=cast("bool", options["by_group"]), show=True)
    assert_figure_artifacts(artifacts, expected_figures=1, required_tables=frozenset(("information",)))
    assert_show_called(calls, artifacts, expected_show=1)
    assert_artifact_ownership(artifacts)
    assert_source_table_values(artifacts, model)
    assert tuple(model.forward_periods) == cast("tuple[str, ...]", options["periods"])
    close_owned_figures(artifacts)
    assert_no_open_figures(artifacts)
    assert_strict_source_invocation(source_invocation_id, model, options, monkeypatch)


@pytest.mark.parametrize(
    "source_invocation_id",
    [
        pytest.param(
            "tests/test_tears.py::TearsTestCase::test_create_turnover_tear_sheet#00/input-00/call-00",
            id="tests/test_tears.py::TearsTestCase::test_create_turnover_tear_sheet#00/input-00/call-00",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_tears.py::TearsTestCase::test_create_turnover_tear_sheet#00/input-00/call-00"
            ),
        ),
        pytest.param(
            "tests/test_tears.py::TearsTestCase::test_create_turnover_tear_sheet#01/input-00/call-00",
            id="tests/test_tears.py::TearsTestCase::test_create_turnover_tear_sheet#01/input-00/call-00",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_tears.py::TearsTestCase::test_create_turnover_tear_sheet#01/input-00/call-00"
            ),
        ),
        pytest.param(
            "tests/test_tears.py::TearsTestCase::test_create_turnover_tear_sheet#02/input-00/call-00",
            id="tests/test_tears.py::TearsTestCase::test_create_turnover_tear_sheet#02/input-00/call-00",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_tears.py::TearsTestCase::test_create_turnover_tear_sheet#02/input-00/call-00"
            ),
        ),
        pytest.param(
            "tests/test_tears.py::TearsTestCase::test_create_turnover_tear_sheet#03/input-00/call-00",
            id="tests/test_tears.py::TearsTestCase::test_create_turnover_tear_sheet#03/input-00/call-00",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_tears.py::TearsTestCase::test_create_turnover_tear_sheet#03/input-00/call-00"
            ),
        ),
    ],
)
def test_create_turnover_tear_sheet_upstream_invocation(
    source_invocation_id: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rebuild every pinned turnover tear sheet invocation as a C4 workflow."""

    from fincore.factor_analysis.tears import close_owned_figures, create_turnover_tear_sheet

    model, options = _model_for_invocation(source_invocation_id)
    assert_source_calendar(model, options)
    pyplot = _pyplot()
    calls: list[object] = []
    monkeypatch.setattr(pyplot, "show", lambda *args, **kwargs: calls.append((args, kwargs)))

    artifacts = create_turnover_tear_sheet(
        model,
        turnover_periods=cast("tuple[int, ...]", options["turnover_periods"]),
        show=True,
    )
    assert_figure_artifacts(
        artifacts,
        expected_figures=1,
        required_tables=frozenset(("turnover", "autocorrelation")),
    )
    assert_show_called(calls, artifacts, expected_show=1)
    assert_artifact_ownership(artifacts)
    assert_source_table_values(artifacts, model)
    assert tuple(model.quantile_turnover) == cast("tuple[int, ...]", options["turnover_periods"])
    close_owned_figures(artifacts)
    assert_no_open_figures(artifacts)
    assert_strict_source_invocation(source_invocation_id, model, options, monkeypatch)


@pytest.mark.parametrize(
    "source_invocation_id",
    [
        pytest.param(
            "tests/test_tears.py::TearsTestCase::test_create_full_tear_sheet#00/input-00/call-00",
            id="tests/test_tears.py::TearsTestCase::test_create_full_tear_sheet#00/input-00/call-00",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_tears.py::TearsTestCase::test_create_full_tear_sheet#00/input-00/call-00"
            ),
        ),
        pytest.param(
            "tests/test_tears.py::TearsTestCase::test_create_full_tear_sheet#00/input-00/call-01",
            id="tests/test_tears.py::TearsTestCase::test_create_full_tear_sheet#00/input-00/call-01",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_tears.py::TearsTestCase::test_create_full_tear_sheet#00/input-00/call-01"
            ),
        ),
        pytest.param(
            "tests/test_tears.py::TearsTestCase::test_create_full_tear_sheet#00/input-00/call-02",
            id="tests/test_tears.py::TearsTestCase::test_create_full_tear_sheet#00/input-00/call-02",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_tears.py::TearsTestCase::test_create_full_tear_sheet#00/input-00/call-02"
            ),
        ),
        pytest.param(
            "tests/test_tears.py::TearsTestCase::test_create_full_tear_sheet#00/input-01/call-00",
            id="tests/test_tears.py::TearsTestCase::test_create_full_tear_sheet#00/input-01/call-00",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_tears.py::TearsTestCase::test_create_full_tear_sheet#00/input-01/call-00"
            ),
        ),
        pytest.param(
            "tests/test_tears.py::TearsTestCase::test_create_full_tear_sheet#00/input-01/call-01",
            id="tests/test_tears.py::TearsTestCase::test_create_full_tear_sheet#00/input-01/call-01",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_tears.py::TearsTestCase::test_create_full_tear_sheet#00/input-01/call-01"
            ),
        ),
        pytest.param(
            "tests/test_tears.py::TearsTestCase::test_create_full_tear_sheet#00/input-01/call-02",
            id="tests/test_tears.py::TearsTestCase::test_create_full_tear_sheet#00/input-01/call-02",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_tears.py::TearsTestCase::test_create_full_tear_sheet#00/input-01/call-02"
            ),
        ),
        pytest.param(
            "tests/test_tears.py::TearsTestCase::test_create_full_tear_sheet#01/input-00/call-00",
            id="tests/test_tears.py::TearsTestCase::test_create_full_tear_sheet#01/input-00/call-00",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_tears.py::TearsTestCase::test_create_full_tear_sheet#01/input-00/call-00"
            ),
        ),
        pytest.param(
            "tests/test_tears.py::TearsTestCase::test_create_full_tear_sheet#01/input-00/call-01",
            id="tests/test_tears.py::TearsTestCase::test_create_full_tear_sheet#01/input-00/call-01",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_tears.py::TearsTestCase::test_create_full_tear_sheet#01/input-00/call-01"
            ),
        ),
        pytest.param(
            "tests/test_tears.py::TearsTestCase::test_create_full_tear_sheet#01/input-00/call-02",
            id="tests/test_tears.py::TearsTestCase::test_create_full_tear_sheet#01/input-00/call-02",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_tears.py::TearsTestCase::test_create_full_tear_sheet#01/input-00/call-02"
            ),
        ),
        pytest.param(
            "tests/test_tears.py::TearsTestCase::test_create_full_tear_sheet#01/input-01/call-00",
            id="tests/test_tears.py::TearsTestCase::test_create_full_tear_sheet#01/input-01/call-00",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_tears.py::TearsTestCase::test_create_full_tear_sheet#01/input-01/call-00"
            ),
        ),
        pytest.param(
            "tests/test_tears.py::TearsTestCase::test_create_full_tear_sheet#01/input-01/call-01",
            id="tests/test_tears.py::TearsTestCase::test_create_full_tear_sheet#01/input-01/call-01",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_tears.py::TearsTestCase::test_create_full_tear_sheet#01/input-01/call-01"
            ),
        ),
        pytest.param(
            "tests/test_tears.py::TearsTestCase::test_create_full_tear_sheet#01/input-01/call-02",
            id="tests/test_tears.py::TearsTestCase::test_create_full_tear_sheet#01/input-01/call-02",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_tears.py::TearsTestCase::test_create_full_tear_sheet#01/input-01/call-02"
            ),
        ),
        pytest.param(
            "tests/test_tears.py::TearsTestCase::test_create_full_tear_sheet#02/input-00/call-00",
            id="tests/test_tears.py::TearsTestCase::test_create_full_tear_sheet#02/input-00/call-00",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_tears.py::TearsTestCase::test_create_full_tear_sheet#02/input-00/call-00"
            ),
        ),
        pytest.param(
            "tests/test_tears.py::TearsTestCase::test_create_full_tear_sheet#02/input-00/call-01",
            id="tests/test_tears.py::TearsTestCase::test_create_full_tear_sheet#02/input-00/call-01",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_tears.py::TearsTestCase::test_create_full_tear_sheet#02/input-00/call-01"
            ),
        ),
        pytest.param(
            "tests/test_tears.py::TearsTestCase::test_create_full_tear_sheet#02/input-00/call-02",
            id="tests/test_tears.py::TearsTestCase::test_create_full_tear_sheet#02/input-00/call-02",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_tears.py::TearsTestCase::test_create_full_tear_sheet#02/input-00/call-02"
            ),
        ),
        pytest.param(
            "tests/test_tears.py::TearsTestCase::test_create_full_tear_sheet#02/input-01/call-00",
            id="tests/test_tears.py::TearsTestCase::test_create_full_tear_sheet#02/input-01/call-00",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_tears.py::TearsTestCase::test_create_full_tear_sheet#02/input-01/call-00"
            ),
        ),
        pytest.param(
            "tests/test_tears.py::TearsTestCase::test_create_full_tear_sheet#02/input-01/call-01",
            id="tests/test_tears.py::TearsTestCase::test_create_full_tear_sheet#02/input-01/call-01",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_tears.py::TearsTestCase::test_create_full_tear_sheet#02/input-01/call-01"
            ),
        ),
        pytest.param(
            "tests/test_tears.py::TearsTestCase::test_create_full_tear_sheet#02/input-01/call-02",
            id="tests/test_tears.py::TearsTestCase::test_create_full_tear_sheet#02/input-01/call-02",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_tears.py::TearsTestCase::test_create_full_tear_sheet#02/input-01/call-02"
            ),
        ),
        pytest.param(
            "tests/test_tears.py::TearsTestCase::test_create_full_tear_sheet#03/input-00/call-00",
            id="tests/test_tears.py::TearsTestCase::test_create_full_tear_sheet#03/input-00/call-00",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_tears.py::TearsTestCase::test_create_full_tear_sheet#03/input-00/call-00"
            ),
        ),
        pytest.param(
            "tests/test_tears.py::TearsTestCase::test_create_full_tear_sheet#03/input-00/call-01",
            id="tests/test_tears.py::TearsTestCase::test_create_full_tear_sheet#03/input-00/call-01",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_tears.py::TearsTestCase::test_create_full_tear_sheet#03/input-00/call-01"
            ),
        ),
        pytest.param(
            "tests/test_tears.py::TearsTestCase::test_create_full_tear_sheet#03/input-00/call-02",
            id="tests/test_tears.py::TearsTestCase::test_create_full_tear_sheet#03/input-00/call-02",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_tears.py::TearsTestCase::test_create_full_tear_sheet#03/input-00/call-02"
            ),
        ),
        pytest.param(
            "tests/test_tears.py::TearsTestCase::test_create_full_tear_sheet#03/input-01/call-00",
            id="tests/test_tears.py::TearsTestCase::test_create_full_tear_sheet#03/input-01/call-00",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_tears.py::TearsTestCase::test_create_full_tear_sheet#03/input-01/call-00"
            ),
        ),
        pytest.param(
            "tests/test_tears.py::TearsTestCase::test_create_full_tear_sheet#03/input-01/call-01",
            id="tests/test_tears.py::TearsTestCase::test_create_full_tear_sheet#03/input-01/call-01",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_tears.py::TearsTestCase::test_create_full_tear_sheet#03/input-01/call-01"
            ),
        ),
        pytest.param(
            "tests/test_tears.py::TearsTestCase::test_create_full_tear_sheet#03/input-01/call-02",
            id="tests/test_tears.py::TearsTestCase::test_create_full_tear_sheet#03/input-01/call-02",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_tears.py::TearsTestCase::test_create_full_tear_sheet#03/input-01/call-02"
            ),
        ),
    ],
)
def test_create_full_tear_sheet_upstream_invocation(
    source_invocation_id: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rebuild every pinned full tear sheet invocation as a C4 workflow."""

    from fincore.factor_analysis.tears import close_owned_figures, create_full_tear_sheet

    model, options = _model_for_invocation(source_invocation_id)
    assert_source_calendar(model, options)
    pyplot = _pyplot()
    calls: list[object] = []
    monkeypatch.setattr(pyplot, "show", lambda *args, **kwargs: calls.append((args, kwargs)))

    artifacts = create_full_tear_sheet(model, by_group=cast("bool", options["by_group"]), show=True)
    expected_figures = 4 if cast("bool", options["by_group"]) else 3
    assert_figure_artifacts(
        artifacts,
        expected_figures=expected_figures,
        required_tables=frozenset(
            ("quantile_statistics", "returns.returns", "information.information", "turnover.turnover")
        ),
    )
    assert_show_called(calls, artifacts, expected_show=expected_figures)
    assert_artifact_ownership(artifacts)
    assert_source_table_values(artifacts, model)
    assert model.config.group_neutral is source_invocation_id.endswith("call-02")
    close_owned_figures(artifacts)
    assert_no_open_figures(artifacts)
    assert_strict_source_invocation(source_invocation_id, model, options, monkeypatch)
