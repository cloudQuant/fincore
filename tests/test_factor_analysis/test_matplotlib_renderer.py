"""Artist-level contracts for the lazy enhanced Matplotlib renderer."""

from __future__ import annotations

import importlib
import sys
from contextlib import contextmanager
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture
def plot_inputs() -> dict[str, object]:
    """Small deterministic tables spanning every renderer input shape."""

    dates = pd.date_range("2024-01-01", periods=6, freq="B", name="date")
    assets = pd.Index(("A", "B"), name="asset")
    factor_index = pd.MultiIndex.from_product((dates[:3], assets), names=("date", "asset"))
    factor_data = pd.DataFrame(
        {
            "factor": np.linspace(-1.0, 1.0, len(factor_index)),
            "factor_quantile": [1, 2, 3, 1, 2, 3],
            "group": ["g1", "g1", "g2", "g2", "g1", "g2"],
        },
        index=factor_index,
    )
    ic = pd.DataFrame(
        {"1D": [0.10, 0.20, np.nan, 0.15, 0.05, 0.11], "5D": [-0.10, 0.00, 0.05, 0.10, 0.20, 0.17]},
        index=dates,
    )
    quantiles = pd.Index((1, 2, 3), name="factor_quantile")
    mean_quantile = pd.DataFrame({"1D": [-0.01, 0.00, 0.02], "5D": [-0.02, 0.01, 0.03]}, index=quantiles)
    return_by_quantile = pd.DataFrame(
        {"1D": np.linspace(-0.03, 0.04, len(dates) * len(quantiles)), "5D": np.linspace(-0.02, 0.05, 18)},
        index=pd.MultiIndex.from_product((dates, quantiles), names=("date", "factor_quantile")),
    )
    spread = pd.DataFrame(
        {"1D": np.linspace(-0.01, 0.02, len(dates)), "5D": np.linspace(-0.02, 0.03, len(dates))}, index=dates
    )
    average_cumulative = pd.DataFrame(
        {"1D": np.linspace(-0.01, 0.02, 12)},
        index=pd.MultiIndex.from_product(
            ((1, 2), ("mean", "std"), (-1, 0, 1)), names=("factor_quantile", "statistic", "period")
        ),
    )
    events = pd.Series(
        np.linspace(-0.02, 0.03, len(factor_index)),
        index=factor_index,
        name="event_return",
    )
    return {
        "dates": dates,
        "factor_data": factor_data,
        "ic": ic,
        "mean_quantile": mean_quantile,
        "return_by_quantile": return_by_quantile,
        "spread": spread,
        "ic_group": pd.DataFrame({"1D": [0.10, -0.05], "5D": [0.05, 0.10]}, index=pd.Index(("g1", "g2"), name="group")),
        "rank": pd.Series(np.linspace(-0.2, 0.6, len(dates)), index=dates, name="rank"),
        "turnover": pd.DataFrame({1: [0.2, 0.3, 0.1, 0.2, 0.25, 0.15], 3: [0.4, 0.2, 0.3, 0.4, 0.3, 0.2]}, index=dates),
        "monthly": ic.resample("ME").mean(),
        "returns": pd.Series([0.01, -0.02, np.nan, 0.03, 0.01, -0.01], index=dates, name="factor_return"),
        "quantile_returns": return_by_quantile.loc[:, ["1D"]],
        "average_cumulative": average_cumulative,
        "events": events,
        "alpha_beta": pd.DataFrame({"1D": [0.12, 0.8], "5D": [0.08, 0.7]}, index=("Ann. alpha", "beta")),
    }


@contextmanager
def _figures_closed() -> Iterator[object]:
    """Close only figures created by a rendering assertion."""

    pyplot = importlib.import_module("matplotlib.pyplot")
    before = set(pyplot.get_fignums())
    try:
        yield pyplot
    finally:
        for number in set(pyplot.get_fignums()) - before:
            pyplot.close(number)


def _axes(value: object) -> tuple[object, ...]:
    """Flatten one Axes or a renderer's multi-axes return into artists."""

    if isinstance(value, np.ndarray):
        return tuple(value.flat)
    if isinstance(value, (list, tuple)):
        return tuple(value)
    return (value,)


_CHART_NAMES = (
    "ic_ts",
    "ic_hist",
    "ic_qq",
    "quantile_bar",
    "quantile_violin",
    "spread",
    "ic_group",
    "rank_autocorrelation",
    "top_bottom_turnover",
    "monthly_heatmap",
    "cumulative_returns",
    "cumulative_returns_by_quantile",
    "quantile_average_cumulative_returns",
    "events_distribution",
)


def _invoke_chart(renderer: object, name: str, inputs: dict[str, object], ax: object = None) -> object:
    """Call one chart through its real public renderer entry point."""

    if name == "ic_ts":
        return renderer.plot_ic_ts(inputs["ic"], ax=ax)
    if name == "ic_hist":
        return renderer.plot_ic_hist(inputs["ic"], ax=ax)
    if name == "ic_qq":
        return renderer.plot_ic_qq(inputs["ic"], ax=ax)
    if name == "quantile_bar":
        return renderer.plot_quantile_returns_bar(inputs["mean_quantile"], ax=ax)
    if name == "quantile_violin":
        return renderer.plot_quantile_returns_violin(inputs["return_by_quantile"], ax=ax)
    if name == "spread":
        return renderer.plot_mean_quantile_returns_spread_time_series(inputs["spread"], ax=ax)
    if name == "ic_group":
        return renderer.plot_ic_by_group(inputs["ic_group"], ax=ax)
    if name == "rank_autocorrelation":
        return renderer.plot_factor_rank_auto_correlation(inputs["rank"], period=3, ax=ax)
    if name == "top_bottom_turnover":
        return renderer.plot_top_bottom_quantile_turnover(inputs["turnover"], period=3, ax=ax)
    if name == "monthly_heatmap":
        return renderer.plot_monthly_ic_heatmap(inputs["monthly"], ax=ax)
    if name == "cumulative_returns":
        return renderer.plot_cumulative_returns(inputs["returns"], "1D", ax=ax)
    if name == "cumulative_returns_by_quantile":
        return renderer.plot_cumulative_returns_by_quantile(inputs["quantile_returns"], "1D", ax=ax)
    if name == "quantile_average_cumulative_returns":
        return renderer.plot_quantile_average_cumulative_return(inputs["average_cumulative"], ax=ax)
    if name == "events_distribution":
        return renderer.plot_events_distribution(inputs["events"], ax=ax)
    raise AssertionError(name)


def _assert_chart_values(name: str, axis: object, inputs: dict[str, object]) -> None:
    """Assert one source-relevant artist/data contract for each chart family."""

    if name == "ic_ts":
        np.testing.assert_allclose(axis.lines[0].get_ydata(), inputs["ic"].iloc[:, 0].to_numpy(), equal_nan=True)
    elif name == "ic_hist":
        assert axis.lines[0].get_xdata()[0] == pytest.approx(inputs["ic"].iloc[:, 0].mean())
    elif name == "ic_qq":
        assert "Normal Dist. Q-Q" in axis.get_title()
    elif name == "quantile_bar":
        assert axis.patches[0].get_height() == pytest.approx(inputs["mean_quantile"].iloc[0, 0] * 10_000)
    elif name == "quantile_violin":
        assert axis.get_ylabel() == "Return (bps)"
    elif name == "spread":
        np.testing.assert_allclose(axis.lines[0].get_ydata(), inputs["spread"].iloc[:, 0].to_numpy() * 10_000)
    elif name == "ic_group":
        assert axis.patches[0].get_height() == pytest.approx(inputs["ic_group"].iloc[0, 0])
    elif name == "rank_autocorrelation":
        np.testing.assert_allclose(axis.lines[0].get_ydata(), inputs["rank"].to_numpy())
    elif name == "top_bottom_turnover":
        np.testing.assert_allclose(axis.lines[0].get_ydata(), inputs["turnover"].iloc[:, -1].to_numpy())
    elif name == "monthly_heatmap":
        assert axis.collections
    elif name == "cumulative_returns":
        np.testing.assert_allclose(axis.lines[0].get_ydata(), inputs["returns"].fillna(0.0).add(1.0).cumprod())
    elif name == "cumulative_returns_by_quantile":
        assert axis.lines and axis.get_yscale() == "symlog"
    elif name == "quantile_average_cumulative_returns":
        expected = inputs["average_cumulative"].loc[(1, "mean"), "1D"].to_numpy() * 10_000
        np.testing.assert_allclose(axis.lines[0].get_ydata(), expected)
    elif name == "events_distribution":
        assert sum(patch.get_height() for patch in axis.patches) == inputs["events"].count()


@pytest.mark.parametrize("name", _CHART_NAMES)
def test_every_chart_has_auto_and_caller_axes_artist_contracts(name: str, plot_inputs: dict[str, object]) -> None:
    """Exercise all 14 charts for auto ownership, caller reuse, labels, and data artists."""

    renderer = importlib.import_module("fincore.factor_analysis.render_matplotlib")
    multi_panel = {"ic_ts", "ic_hist", "ic_qq", "spread", "monthly_heatmap"}
    expected_auto_axes = {
        "ic_ts": 2,
        "ic_hist": 3,
        "ic_qq": 3,
        "quantile_bar": 1,
        "quantile_violin": 1,
        "spread": 2,
        "ic_group": 1,
        "rank_autocorrelation": 1,
        "top_bottom_turnover": 1,
        "monthly_heatmap": 3,
        "cumulative_returns": 1,
        "cumulative_returns_by_quantile": 1,
        "quantile_average_cumulative_returns": 1,
        "events_distribution": 1,
    }
    with _figures_closed() as pyplot:
        before = set(pyplot.get_fignums())
        automatic = _invoke_chart(renderer, name, plot_inputs)
        automatic_axes = _axes(automatic)
        assert len(automatic_axes) == expected_auto_axes[name]
        assert len(set(pyplot.get_fignums()) - before) == 1
        _assert_chart_values(name, automatic_axes[0], plot_inputs)

        if name in multi_panel:
            figure, supplied = pyplot.subplots(1, 2)
            caller_axes: object = np.asarray(supplied)
            result = _invoke_chart(renderer, name, plot_inputs, ax=caller_axes)
            assert result is caller_axes
            _assert_chart_values(name, next(iter(supplied)), plot_inputs)
        else:
            figure, caller_axes = pyplot.subplots()
            assert _invoke_chart(renderer, name, plot_inputs, ax=caller_axes) is caller_axes
            _assert_chart_values(name, caller_axes, plot_inputs)
        pyplot.close(figure)


def test_renderer_import_is_headless_and_contexts_restore_rcparams() -> None:
    """Importing the enhanced layer must not load plotting dependencies or mutate rcParams."""

    sys.modules.pop("fincore.factor_analysis.render_matplotlib", None)
    before = {name for name in sys.modules if name.startswith(("matplotlib", "seaborn"))}
    renderer = importlib.import_module("fincore.factor_analysis.render_matplotlib")
    after = {name for name in sys.modules if name.startswith(("matplotlib", "seaborn"))}
    assert after == before

    matplotlib = importlib.import_module("matplotlib")
    line_width = matplotlib.rcParams["lines.linewidth"]
    with renderer.plotting_context(rc={"lines.linewidth": line_width + 3.0}):
        assert matplotlib.rcParams["lines.linewidth"] == line_width + 3.0
    assert matplotlib.rcParams["lines.linewidth"] == line_width


def test_plotting_contract_records_the_implemented_renderer_projection() -> None:
    """The static C0/C1 registry no longer labels delivered charts as deferred."""

    from fincore.contracts.factor_analysis import function_specs_for_module

    specs = function_specs_for_module("plotting")
    assert len(specs) == 21
    assert {spec.implementation for spec in specs} == {"factor_analysis_task_7_renderer"}
    assert {spec.result_projection for spec in specs} == {"strict_alphalens_plotting_projection"}


def test_renderer_table_builders_return_dataframes_without_display(
    plot_inputs: dict[str, object], capsys: pytest.CaptureFixture[str]
) -> None:
    """Enhanced table helpers are data-only and retain their analytical values."""

    renderer = importlib.import_module("fincore.factor_analysis.render_matplotlib")
    returns = renderer.build_returns_table(
        plot_inputs["alpha_beta"], plot_inputs["mean_quantile"], plot_inputs["spread"]
    )
    turnover, autocorrelation = renderer.build_turnover_tables({1: plot_inputs["rank"]}, {1: plot_inputs["turnover"]})
    information = renderer.build_information_table(plot_inputs["ic"])
    statistics = renderer.build_quantile_statistics_table(plot_inputs["factor_data"])

    assert isinstance(returns, pd.DataFrame)
    assert returns.loc["Mean Period Wise Return Top Quantile (bps)", "1D"] == pytest.approx(200.0)
    assert isinstance(turnover, pd.DataFrame)
    assert isinstance(autocorrelation, pd.DataFrame)
    assert isinstance(information, pd.DataFrame)
    assert "IC Mean" in information.columns
    assert isinstance(statistics, pd.DataFrame)
    assert tuple(statistics.columns[-2:]) == ("count", "count %")
    assert capsys.readouterr().out == ""


def test_renderer_tables_preserve_pinned_broadcast_order_and_quantile_dtype() -> None:
    """Table builders retain the legacy period alignment and display metadata."""

    renderer = importlib.import_module("fincore.factor_analysis.render_matplotlib")
    alpha_beta = pd.DataFrame({"1D": [0.1, 0.8], "5D": [0.2, 0.7]}, index=("Ann. alpha", "beta"))
    quantile_returns = pd.DataFrame({"1D": [-0.01, 0.02], "5D": [-0.02, 0.04]}, index=(1, 2))
    spread = pd.Series([0.01, 0.03], name="spread")
    returns = renderer.build_returns_table(alpha_beta, quantile_returns, spread)
    pd.testing.assert_series_equal(
        returns.loc["Mean Period Wise Spread (bps)"],
        pd.Series({"1D": 200.0, "5D": 200.0}, name="Mean Period Wise Spread (bps)"),
    )

    turnover = {
        10: pd.DataFrame({1: [0.10, 0.20]}),
        2: pd.DataFrame({1: [0.20, 0.30]}),
        1: pd.DataFrame({1: [0.30, 0.40]}),
    }
    autocorrelation = {
        2: pd.Series([0.1, 0.2]),
        10: pd.Series([0.2, 0.3]),
        1: pd.Series([0.3, 0.4]),
    }
    turnover_table, autocorrelation_table = renderer.build_turnover_tables(autocorrelation, turnover)
    assert list(turnover_table.columns) == ["1D", "2D", "10D"]
    assert turnover_table.index.tolist() == ["Quantile 1 Mean Turnover "]
    assert list(autocorrelation_table.columns) == ["2D", "10D", "1D"]

    factor_data = pd.DataFrame({"factor": [1.0, 2.0], "factor_quantile": [1, 2], "group": ["a", "b"]})
    statistics = renderer.build_quantile_statistics_table(factor_data)
    assert statistics.index.dtype == np.dtype("float64")


def test_renderer_empty_panels_return_caller_axes_without_orphan_figures() -> None:
    """Empty multi-panel shapes are no-ops rather than hidden, unreachable figures."""

    renderer = importlib.import_module("fincore.factor_analysis.render_matplotlib")
    dates = pd.date_range("2024-01-01", periods=2, freq="B", name="date")
    empty_periods = pd.DataFrame(index=dates)
    empty_groups = pd.DataFrame(
        {"1D": pd.Series(dtype=float)},
        index=pd.MultiIndex(levels=([], []), codes=([], []), names=("factor_quantile", "group")),
    )
    empty_event = pd.DataFrame(
        {"1D": pd.Series(dtype=float)},
        index=pd.MultiIndex(levels=([], [], []), codes=([], [], []), names=("factor_quantile", "statistic", "period")),
    )
    empty_quantiles = pd.DataFrame(
        {"1D": pd.Series(dtype=float)},
        index=pd.MultiIndex(levels=([], []), codes=([], []), names=("date", "factor_quantile")),
    )
    with _figures_closed() as pyplot:
        before = set(pyplot.get_fignums())
        assert renderer.plot_ic_ts(empty_periods) is None
        assert renderer.plot_ic_hist(empty_periods) is None
        assert renderer.plot_ic_qq(empty_periods) is None
        assert renderer.plot_monthly_ic_heatmap(empty_periods) is None
        assert renderer.plot_quantile_returns_bar(empty_groups, by_group=True) is None
        assert renderer.plot_quantile_average_cumulative_return(empty_event, by_quantile=True) is None
        assert renderer.plot_ic_by_group(empty_periods) is None
        assert renderer.plot_cumulative_returns_by_quantile(empty_quantiles, "1D") is None
        assert set(pyplot.get_fignums()) == before


def test_renderer_nan_percentiles_and_source_shaped_event_rows_are_renderable() -> None:
    """Enhanced charts stay robust for NaNs and consume actual event-kernel row layouts."""

    renderer = importlib.import_module("fincore.factor_analysis.render_matplotlib")
    dates = pd.date_range("2024-01-01", periods=2, freq="B", name="date")
    nan_bars = pd.DataFrame({"1D": [np.nan, np.nan]}, index=pd.Index((1, 2), name="factor_quantile"))
    nan_violin = pd.DataFrame(
        {"1D": [np.nan, np.nan]},
        index=pd.MultiIndex.from_product((dates, (1,)), names=("date", "factor_quantile")),
    )
    source_event_shape = pd.DataFrame(
        [[-0.01, 0.00, 0.02], [0.002, 0.003, 0.004], [-0.02, 0.01, 0.03], [0.004, 0.005, 0.006]],
        index=pd.MultiIndex.from_product(((1, 2), ("mean", "std")), names=("factor_quantile", None)),
        columns=pd.Index((-1, 0, 1), name="period"),
    )
    with _figures_closed() as pyplot:
        figure, axes = pyplot.subplots(1, 3)
        assert renderer.plot_quantile_returns_bar(nan_bars, ylim_percentiles=(5, 95), ax=axes[0]) is axes[0]
        assert renderer.plot_quantile_returns_violin(nan_violin, ylim_percentiles=(5, 95), ax=axes[1]) is axes[1]
        assert renderer.plot_quantile_average_cumulative_return(source_event_shape, std_bar=True, ax=axes[2]) is axes[2]
        np.testing.assert_array_equal(axes[2].lines[0].get_xdata(), np.asarray((-1, 0, 1)))
        np.testing.assert_allclose(axes[2].lines[0].get_ydata(), np.asarray((-100.0, 0.0, 200.0)))
        pyplot.close(figure)


def test_renderer_events_count_non_null_values_and_accept_numpy_integral_bins() -> None:
    """Event bars use source ``count`` semantics instead of counting all index rows."""

    renderer = importlib.import_module("fincore.factor_analysis.render_matplotlib")
    dates = pd.date_range("2024-01-01", periods=51, freq="D", name="date")
    events = pd.Series(1.0, index=dates)
    events.iloc[10] = np.nan
    with _figures_closed() as pyplot:
        figure, axis = pyplot.subplots()
        assert renderer.plot_events_distribution(events, num_bars=np.int64(50), ax=axis) is axis
        heights = np.asarray([patch.get_height() for patch in axis.patches])
        assert 0.0 in heights
        pyplot.close(figure)


def test_events_distribution_uses_categorical_nonoverlapping_intraday_bars() -> None:
    """Pinned event plots bucket time labels as evenly spaced bar categories."""

    renderer = importlib.import_module("fincore.factor_analysis.render_matplotlib")
    events = pd.Series(1.0, index=pd.date_range("2024-01-01", periods=51, freq="min", name="date"))
    with _figures_closed() as pyplot:
        figure, axis = pyplot.subplots()
        assert renderer.plot_events_distribution(events, num_bars=50, ax=axis) is axis
        centers = np.asarray([patch.get_x() + patch.get_width() / 2 for patch in axis.patches])
        np.testing.assert_allclose(np.diff(centers), np.ones(len(centers) - 1))
        np.testing.assert_allclose([patch.get_width() for patch in axis.patches], np.full(len(centers), 0.5))
        pyplot.close(figure)


def test_quantile_bar_skips_unused_categorical_group_levels() -> None:
    """Source-style groupby panels include observed categorical groups only."""

    renderer = importlib.import_module("fincore.factor_analysis.render_matplotlib")
    grouped = pd.DataFrame(
        {"1D": [0.01, 0.02]},
        index=pd.MultiIndex.from_arrays(
            (
                (1, 2),
                pd.Categorical(("g1", "g1"), categories=("g1", "g2")),
            ),
            names=("factor_quantile", "group"),
        ),
    )
    with _figures_closed():
        axes = _axes(renderer.plot_quantile_returns_bar(grouped, by_group=True))
        assert len(axes) == 2
        assert axes[0].get_title() == "g1"
        assert not axes[1].get_visible()


def test_monthly_heatmap_keeps_source_year_rows_and_month_columns() -> None:
    """The heatmap matrix follows the pinned ``set_index(...).unstack`` orientation."""

    renderer = importlib.import_module("fincore.factor_analysis.render_matplotlib")
    monthly = pd.DataFrame(
        {"1D": [1.0, 2.0, 3.0, 4.0]},
        index=pd.DatetimeIndex(("2023-01-31", "2023-02-28", "2024-01-31", "2024-02-29")),
    )
    with _figures_closed() as pyplot:
        figure, axis = pyplot.subplots()
        assert renderer.plot_monthly_ic_heatmap(monthly, ax=axis) is axis
        mesh = axis.collections[0]
        np.testing.assert_allclose(np.asarray(mesh.get_array()).ravel(), np.asarray((1.0, 2.0, 3.0, 4.0)))
        assert [label.get_text() for label in axis.get_xticklabels()] == ["1", "2"]
        assert [label.get_text() for label in axis.get_yticklabels()] == ["2023", "2024"]
        pyplot.close(figure)


def test_quantile_average_chart_adds_one_shared_zero_line() -> None:
    """The all-quantile panel gets one source-style shared event marker."""

    renderer = importlib.import_module("fincore.factor_analysis.render_matplotlib")
    events = pd.DataFrame(
        [[0.01, 0.02], [0.001, 0.002], [-0.01, 0.00], [0.003, 0.004]],
        index=pd.MultiIndex.from_tuples(
            ((1, "mean"), (1, "std"), (2, "mean"), (2, "std")), names=("factor_quantile", None)
        ),
        columns=pd.Index((-1, 0)),
    )
    with _figures_closed() as pyplot:
        figure, axis = pyplot.subplots()
        assert renderer.plot_quantile_average_cumulative_return(events, ax=axis) is axis
        assert len(axis.lines) == 3
        assert [line.get_label() for line in axis.lines[:2]] == ["Quantile 1", "Quantile 2"]
        pyplot.close(figure)


def test_renderer_charts_reuse_passed_axes_and_plot_expected_values(plot_inputs: dict[str, object]) -> None:
    """Every chart builds artists on caller-owned axes without a hidden Figure."""

    renderer = importlib.import_module("fincore.factor_analysis.render_matplotlib")
    with _figures_closed() as pyplot:
        figures = set(pyplot.get_fignums())
        figure, axes = pyplot.subplots(2, 2, figsize=(10, 8))
        flat = axes.flat
        assert _axes(renderer.plot_ic_ts(plot_inputs["ic"], ax=np.asarray(list(flat)[:2]))) == tuple(axes.flat)[:2]
        pyplot.close(figure)

        figure, axes = pyplot.subplots(1, 2)
        assert _axes(renderer.plot_ic_hist(plot_inputs["ic"], ax=axes)) == tuple(axes)
        pyplot.close(figure)

        figure, axes = pyplot.subplots(1, 2)
        assert _axes(renderer.plot_ic_qq(plot_inputs["ic"], ax=axes)) == tuple(axes)
        pyplot.close(figure)

        chart_calls = (
            (renderer.plot_quantile_returns_bar, (plot_inputs["mean_quantile"],)),
            (renderer.plot_quantile_returns_violin, (plot_inputs["return_by_quantile"],)),
            (renderer.plot_mean_quantile_returns_spread_time_series, (plot_inputs["spread"].loc[:, ["1D"]],)),
            (renderer.plot_ic_by_group, (plot_inputs["ic_group"],)),
            (renderer.plot_factor_rank_auto_correlation, (plot_inputs["rank"],)),
            (renderer.plot_top_bottom_quantile_turnover, (plot_inputs["turnover"],)),
            (renderer.plot_cumulative_returns, (plot_inputs["returns"], "1D")),
            (renderer.plot_cumulative_returns_by_quantile, (plot_inputs["quantile_returns"], "1D")),
            (renderer.plot_quantile_average_cumulative_return, (plot_inputs["average_cumulative"],)),
            (renderer.plot_events_distribution, (plot_inputs["events"],)),
        )
        for function, arguments in chart_calls:
            figure, axis = pyplot.subplots()
            result = function(*arguments, ax=axis)
            assert result is axis
            assert axis.figure is figure
            pyplot.close(figure)

        figure, axes = pyplot.subplots(1, 2)
        assert _axes(renderer.plot_monthly_ic_heatmap(plot_inputs["monthly"], ax=axes)) == tuple(axes)
        pyplot.close(figure)

        figure, axis = pyplot.subplots()
        cumulative = renderer.plot_cumulative_returns(plot_inputs["returns"], "1D", ax=axis)
        expected = (plot_inputs["returns"].fillna(0.0).add(1.0).cumprod()).to_numpy()
        np.testing.assert_allclose(cumulative.lines[0].get_ydata(), expected)
        assert cumulative.get_ylabel() == "Cumulative Returns"
        assert "1D" in cumulative.get_title()
        pyplot.close(figure)

        figure, axis = pyplot.subplots()
        rank = renderer.plot_factor_rank_auto_correlation(plot_inputs["rank"], period=3, ax=axis)
        np.testing.assert_allclose(rank.lines[0].get_ydata(), plot_inputs["rank"].to_numpy())
        assert "3D" in rank.get_title()
        pyplot.close(figure)
        assert set(pyplot.get_fignums()) == figures


def test_renderer_auto_axes_empty_nan_and_decorator_ownership(plot_inputs: dict[str, object]) -> None:
    """Automatic charts own one Figure, and empty/NaN inputs remain renderable."""

    renderer = importlib.import_module("fincore.factor_analysis.render_matplotlib")
    with _figures_closed() as pyplot:
        before = set(pyplot.get_fignums())
        result = renderer.plot_ic_ts(plot_inputs["ic"])
        created = set(pyplot.get_fignums()) - before
        assert len(created) == 1
        assert len(_axes(result)) == 2

        empty_axis = renderer.plot_mean_quantile_returns_spread_time_series(
            pd.Series(np.nan, index=plot_inputs["dates"], name="1D")
        )
        assert empty_axis is not None

        date_indexed_events = pd.Series(1.0, index=plot_inputs["dates"], name="event")
        assert renderer.plot_events_distribution(date_indexed_events) is not None

        calls = {"value": 0}

        @renderer.customize
        def decorated() -> int:
            calls["value"] += 1
            return calls["value"]

        assert decorated(set_context=False) == 1
        assert decorated() == 2


def test_renderer_consumes_a_compute_once_model_without_reentering_kernels(
    clean_factor_data: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Renderer functions only construct artists from a finished model snapshot."""

    import fincore.factor_analysis.performance as performance
    from fincore.factor_analysis.analysis import analyze_factor

    renderer = importlib.import_module("fincore.factor_analysis.render_matplotlib")
    model = analyze_factor(clean_factor_data, periods=("1D",), turnover_periods=(1,), include_portfolio_inputs=False)

    def kernel_reentry(*args: object, **kwargs: object) -> None:
        raise AssertionError("a renderer must not re-enter factor-analysis kernels")

    for name in (
        "factor_information_coefficient",
        "factor_rank_autocorrelation",
        "factor_returns",
        "mean_return_by_quantile",
        "quantile_turnover",
    ):
        monkeypatch.setattr(performance, name, kernel_reentry)

    with _figures_closed() as pyplot:
        figure, axis = pyplot.subplots()
        assert renderer.plot_ic_ts(model.information_coefficient, ax=axis) is axis
        pyplot.close(figure)
        figure, axis = pyplot.subplots()
        assert renderer.plot_quantile_returns_bar(model.mean_returns_by_quantile, ax=axis) is axis
        pyplot.close(figure)
        figure, axis = pyplot.subplots()
        assert renderer.plot_top_bottom_quantile_turnover(model.quantile_turnover[1], ax=axis) is axis
        pyplot.close(figure)
        figure, axis = pyplot.subplots()
        assert renderer.plot_cumulative_returns(model.factor_returns["1D"], "1D", ax=axis) is axis
        pyplot.close(figure)
        assert isinstance(renderer.build_quantile_statistics_table(model.factor_data), pd.DataFrame)
