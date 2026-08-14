"""Strict Alphalens plotting projections backed by the lazy renderer."""

from __future__ import annotations

import importlib
import sys
from typing import get_type_hints

import numpy as np
import pandas as pd
import pytest

from fincore.exceptions import DependencyError


def _ic() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=4, freq="B", name="date")
    return pd.DataFrame({"1D": [0.1, 0.2, np.nan, 0.1]}, index=dates)


def _factor_data() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=2, freq="B", name="date")
    index = pd.MultiIndex.from_product((dates, ("A", "B")), names=("date", "asset"))
    return pd.DataFrame({"factor": [1.0, 2.0, 3.0, 4.0], "factor_quantile": [1, 2, 1, 2]}, index=index)


def test_strict_charts_return_iterable_caller_axes_without_showing() -> None:
    """The legacy chart surface renders but never calls ``plt.show`` itself."""

    pyplot = importlib.import_module("matplotlib.pyplot")
    plotting = importlib.import_module("fincore.alphalens.plotting")
    figure, axis = pyplot.subplots()
    try:
        supplied = np.asarray((axis,))
        result = plotting.plot_ic_ts(_ic(), ax=supplied)
        assert result is supplied
        assert axis.get_ylabel() == "IC"
    finally:
        pyplot.close(figure)


def test_strict_table_display_keeps_none_return_and_legacy_heading(capsys: pytest.CaptureFixture[str]) -> None:
    """Strict table calls display their legacy heading while enhanced builders stay data-only."""

    plotting = importlib.import_module("fincore.alphalens.plotting")
    mean_returns = pd.DataFrame({"1D": [-0.01, 0.02]}, index=(1, 2))
    alpha_beta = pd.DataFrame({"1D": [0.1, 0.7]}, index=("Ann. alpha", "beta"))
    assert plotting.plot_returns_table(alpha_beta, mean_returns, mean_returns) is None
    assert "Returns Analysis" in capsys.readouterr().out

    assert plotting.plot_quantile_statistics_table(_factor_data()) is None
    assert "Quantiles Statistics" in capsys.readouterr().out


def test_strict_contexts_restore_style_and_customize_accepts_hidden_context() -> None:
    """The decorator retains its hidden ``set_context`` grammar without global leakage."""

    matplotlib = importlib.import_module("matplotlib")
    plotting = importlib.import_module("fincore.alphalens.plotting")
    original = matplotlib.rcParams["lines.linewidth"]
    with plotting.plotting_context(rc={"lines.linewidth": original + 2.0}):
        assert matplotlib.rcParams["lines.linewidth"] == original + 2.0
    assert matplotlib.rcParams["lines.linewidth"] == original

    @plotting.customize
    def decorated(value: int) -> int:
        return value + 1

    assert decorated(1, set_context=False) == 2
    assert decorated(2) == 3


def test_missing_plot_dependencies_name_the_alphalens_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    """Optional plotting imports fail at the call boundary with installation guidance."""

    plotting = importlib.import_module("fincore.alphalens.plotting")
    renderer = importlib.import_module("fincore.factor_analysis.render_matplotlib")
    original = renderer.importlib.import_module

    def blocked(name: str, package: str | None = None) -> object:
        if name.startswith(("matplotlib", "seaborn")):
            raise ModuleNotFoundError(name)
        return original(name, package)

    monkeypatch.setattr(renderer.importlib, "import_module", blocked)
    with pytest.raises(DependencyError, match=r"fincore\[alphalens\]"):
        plotting.plot_ic_ts(_ic())


@pytest.mark.parametrize(
    ("name", "arguments"),
    (
        ("plot_quantile_returns_bar", (pd.DataFrame({"1D": []}, index=pd.Index([], name="factor_quantile")),)),
        (
            "plot_quantile_returns_violin",
            (
                pd.DataFrame(
                    {"1D": []},
                    index=pd.MultiIndex(levels=([], []), codes=([], []), names=("date", "factor_quantile")),
                ),
            ),
        ),
        ("plot_ic_by_group", (pd.DataFrame(index=pd.Index([], name="group")),)),
        ("plot_monthly_ic_heatmap", (pd.DataFrame({"1D": []}, index=pd.DatetimeIndex([])),)),
        (
            "plot_cumulative_returns_by_quantile",
            (
                pd.DataFrame(
                    {"1D": []},
                    index=pd.MultiIndex(levels=([], []), codes=([], []), names=("date", "factor_quantile")),
                ),
                "1D",
            ),
        ),
        (
            "plot_quantile_average_cumulative_return",
            (
                pd.DataFrame(
                    {"1D": []},
                    index=pd.MultiIndex(levels=([], []), codes=([], []), names=("factor_quantile", "statistic")),
                ),
            ),
        ),
    ),
)
def test_empty_strict_charts_still_require_the_alphalens_extra(
    monkeypatch: pytest.MonkeyPatch, name: str, arguments: tuple[object, ...]
) -> None:
    """Empty data avoids Figures, not the advertised runtime dependency check."""

    plotting = importlib.import_module("fincore.alphalens.plotting")
    renderer = importlib.import_module("fincore.factor_analysis.render_matplotlib")
    original = renderer.importlib.import_module

    def blocked(module_name: str, package: str | None = None) -> object:
        if module_name.startswith(("matplotlib", "seaborn")):
            raise ModuleNotFoundError(module_name)
        return original(module_name, package)

    monkeypatch.setattr(renderer.importlib, "import_module", blocked)
    with pytest.raises(DependencyError, match=r"fincore\[alphalens\]"):
        getattr(plotting, name)(*arguments)


def test_plotting_module_import_does_not_import_or_change_matplotlib_backend() -> None:
    """Strict facade import stays inert even when Matplotlib is already available."""

    matplotlib = importlib.import_module("matplotlib")
    before = matplotlib.get_backend()
    sys.modules.pop("fincore.alphalens.plotting", None)
    importlib.import_module("fincore.alphalens.plotting")
    assert matplotlib.get_backend() == before


def test_strict_plotting_runtime_type_hints_resolve_all_public_annotations() -> None:
    """Core pandas annotations remain consumable without importing visual extras."""

    plotting = importlib.import_module("fincore.alphalens.plotting")
    for name in plotting.__all__:
        hints = get_type_hints(getattr(plotting, name))
        assert "return" in hints, name


def test_strict_ic_time_series_keeps_the_legacy_full_window_rolling_mean() -> None:
    """A short IC history leaves the source's 22-observation mean undefined."""

    pyplot = importlib.import_module("matplotlib.pyplot")
    plotting = importlib.import_module("fincore.alphalens.plotting")
    figure, axis = pyplot.subplots()
    try:
        supplied = np.asarray((axis,))
        assert plotting.plot_ic_ts(_ic(), ax=supplied) is supplied
        assert len(axis.lines) == 3  # IC, 22-row rolling mean, and zero line.
        np.testing.assert_array_equal(axis.lines[1].get_ydata(), np.full(4, np.nan))
    finally:
        pyplot.close(figure)


def test_strict_ic_charts_keep_source_iterable_axes_errors() -> None:
    """The strict surface preserves the pinned scalar-axes grammar failure."""

    pyplot = importlib.import_module("matplotlib.pyplot")
    plotting = importlib.import_module("fincore.alphalens.plotting")
    figure, axis = pyplot.subplots()
    try:
        for name in ("plot_ic_ts", "plot_ic_hist", "plot_ic_qq"):
            with pytest.raises(TypeError, match="Axes.*not iterable"):
                getattr(plotting, name)(_ic(), ax=axis)
    finally:
        pyplot.close(figure)


def test_strict_all_nan_spread_returns_the_original_axis_without_creating_a_figure() -> None:
    """The pinned all-NaN branch is a no-op, including resource ownership."""

    pyplot = importlib.import_module("matplotlib.pyplot")
    plotting = importlib.import_module("fincore.alphalens.plotting")
    spread = pd.Series(np.nan, index=_ic().index, name="1D")
    before = set(pyplot.get_fignums())
    assert plotting.plot_mean_quantile_returns_spread_time_series(spread) is None
    assert set(pyplot.get_fignums()) == before

    figure, axis = pyplot.subplots()
    try:
        assert plotting.plot_mean_quantile_returns_spread_time_series(spread, ax=axis) is axis
        assert not axis.lines
    finally:
        pyplot.close(figure)


def test_strict_charts_keep_legacy_auto_axes_shapes_and_rolling_values() -> None:
    """Strict auto-created charts retain the pinned scalar/grid return conventions."""

    pyplot = importlib.import_module("matplotlib.pyplot")
    plotting = importlib.import_module("fincore.alphalens.plotting")
    spread = pd.Series([0.01, 0.02, 0.03, 0.04], index=_ic().index, name="1D")
    returns = pd.Series([0.01, -0.01, 0.02, 0.00], index=_ic().index, name="factor_returns")
    before = set(pyplot.get_fignums())
    try:
        ic_axes = plotting.plot_ic_ts(_ic())
        assert isinstance(ic_axes, np.ndarray)
        assert ic_axes.shape == (1,)

        histogram_axes = plotting.plot_ic_hist(_ic())
        qq_axes = plotting.plot_ic_qq(_ic())
        assert isinstance(histogram_axes, np.ndarray) and histogram_axes.shape == (3,)
        assert isinstance(qq_axes, np.ndarray) and qq_axes.shape == (3,)

        spread_axis = plotting.plot_mean_quantile_returns_spread_time_series(spread)
        np.testing.assert_array_equal(spread_axis.lines[1].get_ydata(), np.full(4, np.nan))

        cumulative_axis = plotting.plot_cumulative_returns(returns, "1D")
        assert cumulative_axis.figure is not None

        with pytest.raises(AttributeError, match="'NoneType' object has no attribute 'fit'"):
            plotting.plot_ic_qq(_ic(), theoretical_dist=None)
    finally:
        for number in set(pyplot.get_fignums()) - before:
            pyplot.close(number)


def test_strict_special_plotting_projections_preserve_source_side_effects_and_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Strict tables display through IPython, and legacy NaN limits retain their error."""

    plotting = importlib.import_module("fincore.alphalens.plotting")
    displayed: list[pd.DataFrame] = []
    original_import = plotting.importlib.import_module

    class _DisplayModule:
        @staticmethod
        def display(value: pd.DataFrame) -> None:
            displayed.append(value)

    def resolve(name: str, package: str | None = None) -> object:
        if name == "IPython.display":
            return _DisplayModule
        return original_import(name, package)

    monkeypatch.setattr(plotting.importlib, "import_module", resolve)
    assert plotting._display_table("Table", pd.DataFrame({"value": [1.23456]})) is None
    assert displayed and displayed[0].iloc[0, 0] == pytest.approx(1.235)

    precise_factor_data = _factor_data().copy()
    precise_factor_data.loc[:, "factor"] = (1.23456, 2.34567, 3.45678, 4.56789)
    assert plotting.plot_quantile_statistics_table(precise_factor_data) is None
    assert displayed[-1].loc[1.0, "min"] == pytest.approx(1.23456)

    nan_bars = pd.DataFrame({"1D": [np.nan, np.nan]}, index=pd.Index((1, 2), name="factor_quantile"))
    nan_violin = pd.DataFrame(
        {"1D": [np.nan, np.nan]},
        index=pd.MultiIndex.from_product((_ic().index[:2], (1,)), names=("date", "factor_quantile")),
    )
    with (
        pytest.warns(RuntimeWarning, match="All-NaN slice encountered"),
        pytest.raises(ValueError, match="Axis limits cannot be NaN or Inf"),
    ):
        plotting.plot_quantile_returns_bar(nan_bars, ylim_percentiles=(5, 95))
    with (
        pytest.warns(RuntimeWarning, match="All-NaN slice encountered"),
        pytest.raises(ValueError, match="Axis limits cannot be NaN or Inf"),
    ):
        plotting.plot_quantile_returns_violin(nan_violin, ylim_percentiles=(5, 95))

    mixed_inf_bars = pd.DataFrame({"1D": [0.01, np.inf]}, index=pd.Index((1, 2), name="factor_quantile"))
    mixed_inf_violin = pd.DataFrame(
        {"1D": [0.01, -np.inf]},
        index=pd.MultiIndex.from_product((_ic().index[:2], (1,)), names=("date", "factor_quantile")),
    )
    with pytest.warns(RuntimeWarning), pytest.raises(ValueError, match="Axis limits cannot be NaN or Inf"):
        plotting.plot_quantile_returns_bar(mixed_inf_bars, ylim_percentiles=(5, 95))
    with pytest.warns(RuntimeWarning), pytest.raises(ValueError, match="Axis limits cannot be NaN or Inf"):
        plotting.plot_quantile_returns_violin(mixed_inf_violin, ylim_percentiles=(5, 95))
    overflow_bars = pd.DataFrame({"1D": [1e305, 1e305]}, index=pd.Index((1, 2), name="factor_quantile"))
    overflow_violin = pd.DataFrame(
        {"1D": [1e305, 1e305]},
        index=pd.MultiIndex.from_product((_ic().index[:2], (1,)), names=("date", "factor_quantile")),
    )
    with (
        pytest.warns(RuntimeWarning, match="overflow encountered in scalar multiply"),
        pytest.raises(ValueError, match="Axis limits cannot be NaN or Inf"),
    ):
        plotting.plot_quantile_returns_bar(overflow_bars, ylim_percentiles=(5, 95))
    with (
        pytest.warns(RuntimeWarning, match="overflow encountered in scalar multiply"),
        pytest.raises(ValueError, match="Axis limits cannot be NaN or Inf"),
    ):
        plotting.plot_quantile_returns_violin(overflow_violin, ylim_percentiles=(5, 95))
    with (
        pytest.warns(RuntimeWarning, match="invalid value encountered in subtract"),
        pytest.raises(ValueError, match="Axis limits cannot be NaN or Inf"),
    ):
        plotting.plot_mean_quantile_returns_spread_time_series(
            pd.Series((0.01, np.inf), index=_ic().index[:2], name="1D")
        )

    mixed_spread = pd.DataFrame(
        {"1D": [0.01, 0.02, 0.03], "5D": [np.nan, np.nan, np.nan]},
        index=_ic().index[:3],
    )
    with pytest.raises(AttributeError, match="'NoneType' object has no attribute 'get_ylim'"):
        plotting.plot_mean_quantile_returns_spread_time_series(mixed_spread)


def test_strict_group_bar_order_and_date_like_monthly_index_match_source() -> None:
    """Group panels sort like source groupby and monthly plots accept PeriodIndex."""

    pyplot = importlib.import_module("matplotlib.pyplot")
    plotting = importlib.import_module("fincore.alphalens.plotting")
    grouped = pd.DataFrame(
        {"1D": [0.5, 0.7, -0.02, 0.01]},
        index=pd.MultiIndex.from_tuples(
            ((1, "g2"), (2, "g2"), (1, "g1"), (2, "g1")), names=("factor_quantile", "group")
        ),
    )
    before = set(pyplot.get_fignums())
    try:
        axes = plotting.plot_quantile_returns_bar(grouped, by_group=True)
        assert isinstance(axes, np.ndarray) and axes.shape == (2,)
        assert [axis.get_title() for axis in axes] == ["g1", "g2"]
        assert axes[0].get_shared_y_axes().joined(axes[0], axes[1])

        monthly = pd.DataFrame({"1D": [0.1, 0.2]}, index=pd.period_range("2024-01", periods=2, freq="M"))
        monthly_axes = plotting.plot_monthly_ic_heatmap(monthly)
        assert isinstance(monthly_axes, np.ndarray) and monthly_axes.shape == (3,)

        categorical_events = pd.DataFrame(
            {"1D": [0.01, 0.001]},
            index=pd.MultiIndex.from_arrays(
                (
                    pd.Categorical((1, 1), categories=(1, 2)),
                    ("mean", "std"),
                ),
                names=("factor_quantile", None),
            ),
        )
        event_axis = plotting.plot_quantile_average_cumulative_return(categorical_events)
        cm = importlib.import_module("matplotlib.cm")
        np.testing.assert_allclose(event_axis.lines[0].get_color(), cm.coolwarm(1.0))
    finally:
        for number in set(pyplot.get_fignums()) - before:
            pyplot.close(number)


def test_strict_input_level_errors_remain_legacy_key_errors() -> None:
    """Enhanced validation stays friendly without rewriting strict pandas failures."""

    plotting = importlib.import_module("fincore.alphalens.plotting")
    with pytest.raises(KeyError, match=r"Requested level \(group\) does not match index name \(None\)"):
        plotting.plot_quantile_returns_bar(pd.DataFrame({"1D": [0.1]}), by_group=True)
    with pytest.raises(KeyError, match=r"Requested level \(factor_quantile\) does not match index name \(None\)"):
        plotting.plot_cumulative_returns_by_quantile(pd.DataFrame({"1D": [0.1]}), "1D")
    with pytest.raises(KeyError, match=r"Requested level \(date\) does not match index name \(None\)"):
        plotting.plot_events_distribution(pd.Series([1.0], index=pd.Index([0])))


def test_strict_spread_dataframe_uses_the_pinned_default_bandwidth_and_sorted_quantiles() -> None:
    """Legacy recursive spread and groupby paths retain their source ordering quirks."""

    pyplot = importlib.import_module("matplotlib.pyplot")
    plotting = importlib.import_module("fincore.alphalens.plotting")
    dates = _ic().index
    spread = pd.DataFrame({"5D": 0.0, "1D": 0.0}, index=dates)
    standard_error = pd.DataFrame({"5D": 0.001, "1D": 0.001}, index=dates)
    event_rows = pd.DataFrame(
        [[0.01, 0.02], [0.0, 0.0], [-0.01, 0.00], [0.0, 0.0]],
        index=pd.MultiIndex.from_tuples(
            ((2, "mean"), (2, "std"), (1, "mean"), (1, "std")), names=("factor_quantile", None)
        ),
        columns=pd.Index((-1, 0)),
    )
    before = set(pyplot.get_fignums())
    try:
        _, axes = pyplot.subplots(1, 2)
        assert (
            plotting.plot_mean_quantile_returns_spread_time_series(spread, std_err=standard_error, bandwidth=2, ax=axes)
            is axes
        )
        vertices = axes[0].collections[0].get_paths()[0].vertices[:, 1]
        assert vertices.max() - vertices.min() == pytest.approx(20.0)

        event_axis = plotting.plot_quantile_average_cumulative_return(event_rows)
        assert event_axis.lines[0].get_label() == "Quantile 1"
    finally:
        for number in set(pyplot.get_fignums()) - before:
            pyplot.close(number)


def test_strict_spread_error_bands_use_source_positional_order_not_index_alignment() -> None:
    """Pinned bands pair mean ``.values`` with standard-error input order."""

    pyplot = importlib.import_module("matplotlib.pyplot")
    plotting = importlib.import_module("fincore.alphalens.plotting")
    dates = pd.date_range("2024-01-01", periods=3, freq="D")
    mean = pd.Series((0.1, 0.2, 0.3), index=dates, name="1D")
    reversed_error = pd.Series((0.001, 0.01, 0.1), index=dates[::-1], name="1D")
    before = set(pyplot.get_fignums())
    try:
        axis = plotting.plot_mean_quantile_returns_spread_time_series(mean, std_err=reversed_error)
        vertices = axis.collections[0].get_paths()[0].vertices[:, 1]
        np.testing.assert_allclose(
            np.sort(np.unique(vertices)),
            np.asarray((990.0, 1010.0, 1900.0, 2000.0, 2100.0, 4000.0)),
        )
    finally:
        for number in set(pyplot.get_fignums()) - before:
            pyplot.close(number)


def test_strict_spread_dataframe_preserves_all_nan_children_on_caller_axes() -> None:
    """Legacy DataFrame recursion leaves supplied all-NaN panels artist-free."""

    pyplot = importlib.import_module("matplotlib.pyplot")
    plotting = importlib.import_module("fincore.alphalens.plotting")
    dates = pd.date_range("2024-01-01", periods=3, freq="D")
    all_nan = pd.DataFrame({"1D": [np.nan, np.nan, np.nan], "5D": [np.nan, np.nan, np.nan]}, index=dates)
    mixed = pd.DataFrame({"1D": [0.01, 0.02, 0.03], "5D": [np.nan, np.nan, np.nan]}, index=dates)
    before = set(pyplot.get_fignums())
    try:
        _, axes = pyplot.subplots(1, 2)
        assert plotting.plot_mean_quantile_returns_spread_time_series(all_nan, ax=axes) is axes
        assert all(not axis.lines and not axis.collections for axis in axes)
        assert [axis.get_ylim() for axis in axes] == [(0.0, 1.0), (0.0, 1.0)]

        _, mixed_axes = pyplot.subplots(1, 2)
        assert plotting.plot_mean_quantile_returns_spread_time_series(mixed, ax=mixed_axes) is mixed_axes
        assert len(mixed_axes[0].lines) == 3
        assert not mixed_axes[1].lines and not mixed_axes[1].collections
        assert mixed_axes[0].get_ylim() == mixed_axes[1].get_ylim()
    finally:
        for number in set(pyplot.get_fignums()) - before:
            pyplot.close(number)


def test_strict_empty_charts_keep_pinned_figures_and_exception_surfaces() -> None:
    """Strict empty calls retain source ownership/errors while enhanced calls stay safe."""

    pyplot = importlib.import_module("matplotlib.pyplot")
    plotting = importlib.import_module("fincore.alphalens.plotting")
    empty_quantile_index = pd.MultiIndex(levels=([], []), codes=([], []), names=("date", "factor_quantile"))
    empty_average_index = pd.MultiIndex(levels=([], []), codes=([], []), names=("factor_quantile", "statistic"))
    cases = (
        (
            lambda: plotting.plot_quantile_returns_bar(
                pd.DataFrame({"1D": []}, index=pd.Index([], name="factor_quantile"))
            ),
            IndexError,
        ),
        (
            lambda: plotting.plot_ic_by_group(pd.DataFrame({"1D": []}, index=pd.Index([], name="group"))),
            IndexError,
        ),
        (
            lambda: plotting.plot_monthly_ic_heatmap(pd.DataFrame({"1D": []}, index=pd.DatetimeIndex([]))),
            ValueError,
        ),
        (
            lambda: plotting.plot_cumulative_returns_by_quantile(
                pd.DataFrame({"1D": []}, index=empty_quantile_index), "1D"
            ),
            TypeError,
        ),
        (
            lambda: plotting.plot_events_distribution(pd.Series([], dtype=float, index=empty_quantile_index)),
            TypeError,
        ),
    )
    for invoke, error in cases:
        before = set(pyplot.get_fignums())
        try:
            with pytest.raises(error):
                invoke()
            assert len(set(pyplot.get_fignums()) - before) == 1
        finally:
            for number in set(pyplot.get_fignums()) - before:
                pyplot.close(number)

    before = set(pyplot.get_fignums())
    try:
        violin = plotting.plot_quantile_returns_violin(pd.DataFrame({"1D": []}, index=empty_quantile_index))
        average = plotting.plot_quantile_average_cumulative_return(pd.DataFrame({"1D": []}, index=empty_average_index))
        assert violin.figure is not average.figure
        assert len(set(pyplot.get_fignums()) - before) == 2
    finally:
        for number in set(pyplot.get_fignums()) - before:
            pyplot.close(number)


def test_strict_events_keep_permissive_pinned_window_grammar() -> None:
    """The strict chart accepts source-supported float and NumPy bar counts."""

    pyplot = importlib.import_module("matplotlib.pyplot")
    plotting = importlib.import_module("fincore.alphalens.plotting")
    index = pd.MultiIndex.from_product(
        (pd.date_range("2024-01-01", periods=2, freq="D"), ("A",)), names=("date", "asset")
    )
    events = pd.Series((0.01, 0.02), index=index)
    for num_bars, expected in ((1.0, 2), (np.float64(1), 2), (2.5, 3)):
        before = set(pyplot.get_fignums())
        try:
            axis = plotting.plot_events_distribution(events, num_bars=num_bars)
            assert len(axis.patches) == expected
        finally:
            for number in set(pyplot.get_fignums()) - before:
                pyplot.close(number)


def test_strict_quantile_cumulative_returns_preserve_source_limits_and_empty_columns() -> None:
    """Strict limits retain source warnings rather than the enhanced safe expansion."""

    pyplot = importlib.import_module("matplotlib.pyplot")
    plotting = importlib.import_module("fincore.alphalens.plotting")
    index = pd.MultiIndex.from_product(
        (pd.date_range("2024-01-01", periods=3, freq="D"), (1, 2)), names=("date", "factor_quantile")
    )
    zero_returns = pd.DataFrame({"1D": 0.0}, index=index)
    before = set(pyplot.get_fignums())
    try:
        with pytest.warns(UserWarning, match="identical low and high"):
            axis = plotting.plot_cumulative_returns_by_quantile(zero_returns, "1D")
        assert axis.get_ylim() == pytest.approx((0.95, 1.05))
        np.testing.assert_allclose(axis.get_yticks(), np.ones(5))
    finally:
        for number in set(pyplot.get_fignums()) - before:
            pyplot.close(number)

    infinite_returns = pd.DataFrame({"1D": np.inf}, index=index)
    before = set(pyplot.get_fignums())
    try:
        with pytest.warns(RuntimeWarning), pytest.raises(ValueError, match="Axis limits cannot be NaN or Inf"):
            plotting.plot_cumulative_returns_by_quantile(infinite_returns, "1D")
        assert len(set(pyplot.get_fignums()) - before) == 1
    finally:
        for number in set(pyplot.get_fignums()) - before:
            pyplot.close(number)

    empty_turnover = pd.DataFrame(index=pd.date_range("2024-01-01", periods=2, freq="D"))
    before = set(pyplot.get_fignums())
    try:
        with pytest.raises(KeyError, match="nan"):
            plotting.plot_top_bottom_quantile_turnover(empty_turnover)
        assert len(set(pyplot.get_fignums()) - before) == 1
    finally:
        for number in set(pyplot.get_fignums()) - before:
            pyplot.close(number)


def test_strict_malformed_plot_inputs_keep_source_error_classes() -> None:
    """Friendly enhanced schema validation cannot replace strict source exceptions."""

    plotting = importlib.import_module("fincore.alphalens.plotting")
    with pytest.raises(ValueError, match=r"Could not interpret value `factor_quantile` for `x`"):
        plotting.plot_quantile_returns_violin(pd.DataFrame({"1D": [0.01]}, index=pd.DatetimeIndex(["2024-01-01"])))
    with pytest.raises(AttributeError, match="DatetimeIndex.*levels"):
        plotting.plot_quantile_average_cumulative_return(
            pd.DataFrame({"1D": [0.01]}, index=pd.DatetimeIndex(["2024-01-01"]))
        )
    with pytest.raises(AttributeError, match="int.*year"):
        plotting.plot_monthly_ic_heatmap(pd.DataFrame({"1D": [0.01]}, index=pd.Index([1])))
