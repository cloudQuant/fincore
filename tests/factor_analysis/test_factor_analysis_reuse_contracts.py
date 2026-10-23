"""Reuse contracts for factor-analysis weights and research labels."""

from __future__ import annotations

from collections import defaultdict
from functools import wraps
from inspect import signature

import numpy as np
import pandas as pd
import pytest

from fincore.factor_analysis import analysis, data, performance, portfolio, research


def _clean_factor_data(
    *,
    n_dates: int = 8,
    n_assets: int = 12,
    factor_values: np.ndarray | None = None,
) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=n_dates, tz="UTC")
    assets = tuple(f"A{index:02d}" for index in range(n_assets))
    index = pd.MultiIndex.from_product((dates, assets), names=("date", "asset"))
    if factor_values is None:
        cross_section = np.linspace(-1.0, 1.0, n_assets)
        factor_values = np.tile(cross_section, n_dates) + np.repeat(np.arange(n_dates) * 0.03, n_assets)
    factor = pd.Series(np.asarray(factor_values, dtype=float), index=index, name="factor")
    row = np.tile(np.arange(n_assets, dtype=float), n_dates)
    session = np.repeat(np.arange(n_dates, dtype=float), n_assets)
    returns = pd.DataFrame(
        {
            "1D": 0.002 * row + 0.001 * session,
            "2D": 0.003 * row - 0.0005 * session,
        },
        index=index,
    )
    groups = {asset: "left" if position < n_assets // 2 else "right" for position, asset in enumerate(assets)}
    prepared = data.prepare_factor_data_from_forward_returns(
        factor,
        returns,
        groupby=groups,
        quantiles=4,
        max_loss=1.0,
    )
    return prepared.data


def _analyze(factor_data: pd.DataFrame) -> analysis.FactorAnalysisModel:
    return analysis.analyze_factor(
        factor_data,
        long_short=True,
        group_neutral=True,
        equal_weight=False,
        by_group=False,
        periods=("1D",),
        turnover_periods=(1,),
        time_aggregation=(),
        include_portfolio_inputs=True,
        portfolio_benchmark_period="9D",
    )


def test_cached_portfolio_projection_matches_builder_and_isolates_inputs() -> None:
    factor_data = _clean_factor_data()
    weights = performance.factor_weights(
        factor_data,
        demeaned=True,
        group_adjust=True,
        equal_weight=False,
    )
    cumulative = portfolio.factor_cumulative_returns(
        factor_data,
        "1D",
        long_short=True,
        group_neutral=True,
        equal_weight=False,
        weights=weights,
    )
    daily_cumulative, daily_returns = portfolio._daily_cumulative_returns(cumulative)
    raw_positions = portfolio.factor_positions(
        factor_data,
        "1D",
        long_short=True,
        group_neutral=True,
        equal_weight=False,
        weights=weights,
    )
    cumulative_before = daily_cumulative.copy(deep=True)
    returns_before = daily_returns.copy(deep=True)
    positions_before = raw_positions.copy(deep=True)
    factor_data_before = factor_data.copy(deep=True)

    # This is the pre-reuse builder projection, kept independent of the new
    # helper so exact values, names, indexes, and dtypes remain pinned.
    expected_positions = raw_positions.resample("1D").sum().ffill()
    asset_gross = expected_positions.abs().sum(axis=1)
    expected_positions = expected_positions.div(asset_gross, axis=0).fillna(0.0)
    expected_positions["cash"] = 1.0 - expected_positions.sum(axis=1)
    capital_curve = daily_cumulative.reindex(expected_positions.index).ffill()
    expected_positions = expected_positions.mul(capital_curve * 100_000.0, axis=0)
    benchmark_data = performance._copy_factor_data(factor_data)
    benchmark_data["factor"] = benchmark_data["factor"].abs()
    benchmark_cumulative = portfolio.factor_cumulative_returns(
        benchmark_data,
        "1D",
        long_short=False,
        group_neutral=False,
        equal_weight=True,
    )
    _, expected_benchmark = portfolio._daily_cumulative_returns(benchmark_cumulative)
    expected_benchmark.name = "benchmark"

    expected = portfolio.FactorPortfolioInputs(
        returns=daily_returns.copy(deep=True),
        positions=expected_positions.copy(deep=True),
        benchmark_returns=expected_benchmark.copy(deep=True),
    )
    projected = portfolio._build_factor_portfolio_inputs_from_outputs(
        factor_data,
        daily_cumulative,
        daily_returns,
        raw_positions,
        capital=100_000.0,
        benchmark_period="1D",
    )
    public = portfolio.build_factor_portfolio_inputs(
        factor_data,
        "1D",
        capital=100_000.0,
        long_short=True,
        group_neutral=True,
        equal_weight=False,
        benchmark_period="1D",
        weights=weights,
    )

    expected_parameters = (
        "factor_data",
        "period",
        "capital",
        "long_short",
        "group_neutral",
        "equal_weight",
        "quantiles",
        "groups",
        "benchmark_period",
        "weights",
    )
    assert tuple(signature(portfolio.build_factor_portfolio_inputs).parameters) == (
        expected_parameters
    )
    for actual in (projected, public):
        pd.testing.assert_series_equal(actual.returns, expected.returns, check_exact=True)
        pd.testing.assert_frame_equal(actual.positions, expected.positions, check_exact=True)
        assert actual.benchmark_returns is not None
        pd.testing.assert_series_equal(
            actual.benchmark_returns,
            expected.benchmark_returns,
            check_exact=True,
        )

    assert projected.returns is not daily_returns
    assert projected.positions is not raw_positions
    projected.returns.iloc[0] = 123.0
    projected.positions.iloc[0, 0] = 456.0
    assert projected.benchmark_returns is not None
    projected.benchmark_returns.iloc[0] = 789.0
    pd.testing.assert_series_equal(daily_cumulative, cumulative_before, check_exact=True)
    pd.testing.assert_series_equal(daily_returns, returns_before, check_exact=True)
    pd.testing.assert_frame_equal(raw_positions, positions_before, check_exact=True)
    pd.testing.assert_frame_equal(factor_data, factor_data_before, check_exact=True)


def test_analyze_factor_builds_one_weight_vector_and_reuses_it(monkeypatch: pytest.MonkeyPatch) -> None:
    clean = _clean_factor_data()
    analysis_data = clean.drop(columns=["2D"]).copy(deep=True)
    expected_weights = performance.factor_weights(
        analysis_data,
        demeaned=True,
        group_adjust=True,
        equal_weight=False,
    )
    expected_cumulative = portfolio.factor_cumulative_returns(
        analysis_data,
        "1D",
        long_short=True,
        group_neutral=True,
        equal_weight=False,
        weights=None,
    )
    expected_positions = portfolio.factor_positions(
        analysis_data,
        "1D",
        long_short=True,
        group_neutral=True,
        equal_weight=False,
        weights=None,
    )
    expected_inputs = portfolio.build_factor_portfolio_inputs(
        analysis_data,
        "1D",
        long_short=True,
        group_neutral=True,
        equal_weight=False,
        benchmark_period="9D",
        weights=None,
    )

    original_weights = performance.factor_weights
    original_returns = performance.factor_returns
    original_cumulative = portfolio.factor_cumulative_returns
    original_positions = portfolio.factor_positions
    original_projection = portfolio._build_factor_portfolio_inputs_from_outputs
    weight_calls: list[dict[str, object]] = []
    public_calls: dict[str, list[dict[str, object]]] = defaultdict(list)
    projection_calls: list[dict[str, object]] = []

    @wraps(original_weights)
    def factor_weights_spy(*args: object, **kwargs: object) -> pd.Series:
        result = original_weights(*args, **kwargs)
        weight_calls.append(
            {
                "data": args[0] if args else kwargs["factor_data"],
                "demeaned": kwargs.get("demeaned", args[1] if len(args) > 1 else True),
                "group_adjust": kwargs.get("group_adjust", args[2] if len(args) > 2 else False),
                "equal_weight": kwargs.get("equal_weight", args[3] if len(args) > 3 else False),
                "result": result,
            }
        )
        return result

    def record_call(name: str, original: object):
        @wraps(original)
        def spy(*args: object, **kwargs: object) -> object:
            public_calls[name].append(
                {
                    "long_short": kwargs.get("long_short", kwargs.get("demeaned")),
                    "group_neutral": kwargs.get("group_neutral", kwargs.get("group_adjust")),
                    "equal_weight": kwargs.get("equal_weight"),
                    "weights": kwargs.get("weights"),
                }
            )
            return original(*args, **kwargs)  # type: ignore[operator]

        return spy

    monkeypatch.setattr(performance, "factor_weights", factor_weights_spy)
    monkeypatch.setattr(performance, "factor_returns", record_call("factor_returns", original_returns))
    monkeypatch.setattr(portfolio, "factor_cumulative_returns", record_call("cumulative", original_cumulative))
    monkeypatch.setattr(portfolio, "factor_positions", record_call("positions", original_positions))

    @wraps(original_projection)
    def projection_spy(*args: object, **kwargs: object) -> portfolio.FactorPortfolioInputs:
        projection_calls.append(dict(kwargs))
        return original_projection(*args, **kwargs)  # type: ignore[arg-type]

    def unexpected_public_builder(
        *args: object,
        **kwargs: object,
    ) -> portfolio.FactorPortfolioInputs:
        pytest.fail("analysis should project its already-computed first-period outputs")

    monkeypatch.setattr(portfolio, "_build_factor_portfolio_inputs_from_outputs", projection_spy)
    monkeypatch.setattr(portfolio, "build_factor_portfolio_inputs", unexpected_public_builder)

    model = _analyze(clean)

    # These comparisons use the public uncached functions above as the oracle.
    pd.testing.assert_frame_equal(model.factor_weights, expected_weights.to_frame("factor"))
    pd.testing.assert_series_equal(model.factor_cumulative_returns["1D"], expected_cumulative)
    pd.testing.assert_frame_equal(model.factor_positions["1D"], expected_positions)
    assert model.portfolio_inputs is not None
    pd.testing.assert_series_equal(model.portfolio_inputs.returns, expected_inputs.returns)
    pd.testing.assert_frame_equal(model.portfolio_inputs.positions, expected_inputs.positions)

    assert weight_calls, "analysis must construct weights"
    assert all(
        (call["demeaned"], call["group_adjust"], call["equal_weight"]) == (True, True, False)
        for call in weight_calls
    )
    assert all(public_calls.values()), "all exposed portfolio calculations should be exercised"
    assert len(public_calls["cumulative"]) == 1
    assert len(public_calls["positions"]) == 1
    assert len(projection_calls) == 1
    projected_daily_curve, projected_daily_returns = portfolio._daily_cumulative_returns(
        expected_cumulative
    )
    pd.testing.assert_series_equal(
        projection_calls[0]["daily_cumulative"],
        projected_daily_curve,
    )
    pd.testing.assert_series_equal(projection_calls[0]["returns"], projected_daily_returns)
    pd.testing.assert_frame_equal(projection_calls[0]["raw_positions"], expected_positions)
    for calls in public_calls.values():
        for call in calls:
            assert (call["long_short"], call["group_neutral"], call["equal_weight"]) == (True, True, False)
            assert call["weights"] is weight_calls[0]["result"]

    # A single analysis owns one weight build; factor_returns and portfolio
    # projections must consume that same vector instead of rebuilding it.
    assert len(weight_calls) == 1


def test_weight_and_model_snapshots_are_isolated_across_runs() -> None:
    clean = _clean_factor_data()
    reversed_clean = clean.copy(deep=True)
    reversed_clean["factor"] = -reversed_clean["factor"]

    first = _analyze(clean)
    first_repeat = _analyze(clean)
    different_parameters = analysis.analyze_factor(
        clean,
        long_short=False,
        group_neutral=False,
        equal_weight=True,
        by_group=False,
        periods=("1D",),
        turnover_periods=(1,),
        time_aggregation=(),
        include_portfolio_inputs=True,
        portfolio_benchmark_period="9D",
    )
    different_factor = _analyze(reversed_clean)
    selected_clean = clean.drop(columns=["2D"])
    selected_reversed = reversed_clean.drop(columns=["2D"])
    expected_first = performance.factor_weights(
        selected_clean,
        demeaned=True,
        group_adjust=True,
        equal_weight=False,
    )
    expected_parameters = performance.factor_weights(
        selected_clean,
        demeaned=False,
        group_adjust=False,
        equal_weight=True,
    )
    expected_factor = performance.factor_weights(
        selected_reversed,
        demeaned=True,
        group_adjust=True,
        equal_weight=False,
    )

    pd.testing.assert_frame_equal(first.factor_weights, expected_first.to_frame("factor"))
    pd.testing.assert_frame_equal(
        different_parameters.factor_weights,
        expected_parameters.to_frame("factor"),
    )
    pd.testing.assert_frame_equal(different_factor.factor_weights, expected_factor.to_frame("factor"))
    assert first.result_fingerprint == first_repeat.result_fingerprint
    assert first.result_fingerprint != different_parameters.result_fingerprint
    assert first.result_fingerprint != different_factor.result_fingerprint

    # Both the model-to-model boundary and the public getter return independent
    # mutable snapshots; modifying a read cannot poison a later model read.
    first_view = first.factor_weights
    repeat_view = first_repeat.factor_weights
    parameters_view = different_parameters.factor_weights
    factor_view = different_factor.factor_weights
    assert first_view is not repeat_view
    assert first_view is not parameters_view
    assert first_view is not factor_view
    original_first = first_view.copy(deep=True)
    original_repeat = repeat_view.copy(deep=True)
    original_parameters = parameters_view.copy(deep=True)
    original_factor = factor_view.copy(deep=True)
    first_fingerprint = first.result_fingerprint
    first_view.iloc[0, 0] = 12345.0
    pd.testing.assert_frame_equal(first.factor_weights, original_first)
    pd.testing.assert_frame_equal(first_repeat.factor_weights, original_repeat)
    pd.testing.assert_frame_equal(different_parameters.factor_weights, original_parameters)
    pd.testing.assert_frame_equal(different_factor.factor_weights, original_factor)
    assert first.result_fingerprint == first_fingerprint

    position_view = first.factor_positions["1D"]
    repeat_position_view = first_repeat.factor_positions["1D"]
    assert position_view is not repeat_position_view
    original_repeat_positions = repeat_position_view.copy(deep=True)
    position_view.iloc[0, 0] = 9876.0
    pd.testing.assert_frame_equal(first_repeat.factor_positions["1D"], original_repeat_positions)

    # The reviewed group-neutral path balances gross exposure between groups.
    weight_series = first.factor_weights["factor"]
    grouped_gross = weight_series.groupby(
        [weight_series.index.get_level_values("date"), clean["group"]],
        observed=False,
        sort=True,
    ).apply(lambda values: values.abs().sum())
    per_date_gross = grouped_gross.groupby(level=0, sort=True)
    for _, gross_by_group in per_date_gross:
        assert gross_by_group.to_numpy(dtype=float) == pytest.approx(
            np.full(len(gross_by_group), 1.0 / len(gross_by_group))
        )


def test_constant_group_factor_produces_nan_zero_gross_weights() -> None:
    index = pd.MultiIndex.from_product(
        (pd.bdate_range("2024-02-01", periods=2, tz="UTC"), ("A", "B", "C", "D")),
        names=("date", "asset"),
    )
    factor_data = pd.DataFrame(
        {
            "factor": [1.0, 1.0, 3.0, 5.0] * 2,
            "group": ["flat", "flat", "moving", "moving"] * 2,
        },
        index=index,
    )

    weights = performance.factor_weights(
        factor_data,
        demeaned=True,
        group_adjust=True,
        equal_weight=False,
    )

    assert weights.index.equals(index)
    assert weights.loc[(slice(None), ["A", "B"])].isna().all()
    assert weights.loc[(slice(None), ["C", "D"])].notna().all()
    assert weights.loc[(slice(None), ["C", "D"])].abs().groupby(level="date").sum().to_numpy() == pytest.approx(
        [1.0, 1.0]
    )


def _research_panel() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DatetimeIndex, tuple[str, ...]]:
    dates = pd.bdate_range("2024-03-04", periods=8, tz="UTC")
    assets = tuple(f"S{index:02d}" for index in range(12))
    grid = pd.bdate_range(dates[0], periods=len(dates) + 3, tz="UTC")
    index = pd.MultiIndex.from_product((dates, assets), names=("time", "asset"))
    cross_section = np.linspace(-0.05, 0.05, len(assets))
    session = np.linspace(-0.01, 0.01, len(dates))
    first_values = cross_section[None, :] + session[:, None]
    second_values = cross_section[::-1][None, :] - session[:, None]
    factors = pd.DataFrame(
        {
            "first": first_values.reshape(-1),
            "second": second_values.reshape(-1),
        },
        index=index,
    )
    factors.iloc[0, factors.columns.get_loc("first")] = np.nan
    factors.iloc[1, factors.columns.get_loc("first")] = np.nan
    factors.iloc[2, factors.columns.get_loc("second")] = np.nan
    opens = pd.DataFrame(index=grid, columns=list(assets), dtype=float)
    for position, asset in enumerate(assets):
        path = np.empty(len(grid), dtype=float)
        path[0] = 100.0 + position
        for step in range(1, len(grid)):
            path[step] = path[step - 1] * (1.0 + 0.001 * (position + 1) + 0.0003 * step)
        opens[asset] = path
    prices = {"open": opens, "close": opens * 0.99}
    return factors, pd.DataFrame({"open_time": grid}, index=grid), prices, dates, assets


def test_research_builds_each_raw_label_once_and_masks_each_factor_separately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factors, grid, prices, dates, assets = _research_panel()
    original_evaluate_period = research._evaluate_period
    evaluate_calls: list[dict[str, object]] = []

    @wraps(original_evaluate_period)
    def evaluate_period_spy(**kwargs: object):
        evaluate_calls.append(
            {
                "factor": kwargs["factor"],
                "label": kwargs["label"],
                "values": kwargs["values"],
                "label_values": kwargs["label_values"],
            }
        )
        return original_evaluate_period(**kwargs)

    monkeypatch.setattr(research, "_evaluate_period", evaluate_period_spy)
    result = research.evaluate_factor_research(
        factors,
        grid=grid,
        prices=prices,
        universe=None,
        groups=None,
        precomputed_returns=None,
        spec={
            "targets": assets,
            "start": dates[0],
            "end": dates[-1],
            "periods": (1, 2),
            "label": "next_open_to_open",
            "preprocess": "none",
            "quantiles": 4,
            "min_assets": 10,
            "min_quantile_assets": 1,
            "stability_lags": (),
        },
    )

    assert result.status == "OK"
    assert set(result.ic_summary["factor"]) == {"first", "second"}
    calls_by_label: dict[str, list[dict[str, object]]] = defaultdict(list)
    for call in evaluate_calls:
        calls_by_label[str(call["label"])].append(call)
    assert set(calls_by_label) == {"1D", "2D"}
    for calls in calls_by_label.values():
        assert {call["factor"] for call in calls} == {"first", "second"}
        assert calls[0]["label_values"] is calls[1]["label_values"]
        assert calls[0]["values"] is not calls[1]["values"]
        assert not np.array_equal(
            np.isfinite(np.asarray(calls[0]["values"], dtype=float)),
            np.isfinite(np.asarray(calls[1]["values"], dtype=float)),
        )

    labels_by_cell = result.labels.pivot(index=["time", "asset"], columns="period", values="value")
    label_any = np.isfinite(labels_by_cell.loc[factors.index, ["1D", "2D"]].to_numpy(dtype=float)).any(axis=1)
    coverage = result.coverage.set_index("factor")
    for name in factors.columns:
        finite_factor = np.isfinite(factors[name].to_numpy(dtype=float))
        expected_evaluable = int(np.count_nonzero(finite_factor & label_any))
        assert int(coverage.loc[name, "finite_cells"]) == int(np.count_nonzero(finite_factor))
        assert int(coverage.loc[name, "evaluable_cells"]) == expected_evaluable
        assert coverage.loc[name, "status"] == "OK"
    assert int(coverage.loc["first", "evaluable_cells"]) != int(coverage.loc["second", "evaluable_cells"])
