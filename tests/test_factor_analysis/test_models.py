"""Compute-once model contracts for the enhanced factor-analysis surface."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import FrozenInstanceError, fields, is_dataclass
from typing import get_type_hints

import numpy as np
import pandas as pd
import pytest

from fincore.factor_analysis import performance, portfolio
from fincore.factor_analysis.calendar import get_forward_returns_columns


def _only_periods(factor_data: pd.DataFrame, periods: tuple[str, ...]) -> pd.DataFrame:
    """Return a copied clean table with exactly the requested forward columns."""

    copied = factor_data.copy(deep=True)
    forward = get_forward_returns_columns(copied.columns)
    return copied.drop(columns=[column for column in forward if column not in periods])


def _event_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Build a real return matrix suitable for the event kernel."""

    return prices.pct_change(fill_method=None).fillna(0.0)


def _assert_serializable_data_only(value: object) -> None:
    """Reject renderer objects and executable cache payloads recursively."""

    assert not callable(value)
    qualified_name = f"{type(value).__module__}.{type(value).__qualname__}"
    assert "matplotlib" not in qualified_name
    if is_dataclass(value) and not isinstance(value, type):
        for item in fields(value):
            _assert_serializable_data_only(getattr(value, item.name))
    elif isinstance(value, Mapping):
        for key, item in value.items():
            _assert_serializable_data_only(key)
            _assert_serializable_data_only(item)
    elif isinstance(value, tuple):
        for item in value:
            _assert_serializable_data_only(item)


def test_factor_analysis_model_declares_every_renderer_required_field() -> None:
    """The model contract is explicit rather than a loose ``Mapping[str, Any]``."""

    from fincore.factor_analysis.models import FactorAnalysisModel

    names = {item.name for item in fields(FactorAnalysisModel)}
    assert {
        "config",
        "factor_data",
        "forward_periods",
        "quantile_statistics",
        "factor_weights",
        "factor_returns",
        "factor_cumulative_returns",
        "factor_positions",
        "alpha_beta",
        "mean_returns_by_quantile",
        "std_error_by_quantile",
        "mean_returns_by_date",
        "mean_return_spread",
        "mean_return_spread_std",
        "information_coefficient",
        "mean_information_coefficient",
        "quantile_turnover",
        "rank_autocorrelation",
        "grouped_results",
        "time_aggregated_results",
        "pyfolio_inputs",
        "event_returns",
        "result_fingerprint",
    } <= names

    from fincore.factor_analysis.models import EventAnalysisModel, FactorGroupAnalysis

    assert {"group", "quantile_statistics", "factor_returns", "information_coefficient", "quantile_turnover"} <= {
        item.name for item in fields(FactorGroupAnalysis)
    }
    assert {"event_windows", "mean_returns", "return_distribution", "quantile_average_returns"} <= {
        item.name for item in fields(EventAnalysisModel)
    }


def test_public_model_and_entrypoint_annotations_resolve_at_runtime() -> None:
    """Renderer integrations can reflect the checked-in typed public contract."""

    from fincore.factor_analysis.analysis import analyze_factor
    from fincore.factor_analysis.models import FactorAnalysisModel

    assert "periods" in get_type_hints(analyze_factor)
    assert "pyfolio_inputs" in get_type_hints(FactorAnalysisModel)


def test_analyze_factor_computes_ic_once_and_owns_input_snapshot(
    clean_factor_data: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Analysis performs computation once and render consumers read its snapshot."""

    from fincore.factor_analysis.analysis import analyze_factor

    calls = {"ic": 0}
    original = performance.factor_information_coefficient

    def counted(*args: object, **kwargs: object) -> pd.DataFrame:
        calls["ic"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(performance, "factor_information_coefficient", counted)
    model = analyze_factor(
        clean_factor_data,
        periods=("1D", "5D"),
        turnover_periods=(1,),
        include_pyfolio=False,
    )
    # Pandas 3 rejects NaN assignment into the integer quantile column; a
    # factor-only mutation still proves the model owns its input snapshot.
    clean_factor_data.loc[:, "factor"] = np.nan

    assert calls["ic"] == 1
    assert not model.factor_data.isna().all().all()
    assert model.forward_periods == ("1D", "5D")

    calls_before_consumption = calls["ic"]
    first_renderer_input = model.information_coefficient.copy(deep=True)
    second_renderer_input = model.mean_returns_by_quantile.copy(deep=True)
    assert not first_renderer_input.empty
    assert not second_renderer_input.empty
    assert calls["ic"] == calls_before_consumption


def test_model_fields_match_the_existing_enhanced_kernel_outputs(clean_factor_data: pd.DataFrame) -> None:
    """Model fields are snapshots of audited kernel output, not renderer placeholders."""

    from fincore.factor_analysis.analysis import analyze_factor

    source = _only_periods(clean_factor_data, ("1D", "5D"))
    model = analyze_factor(source, periods=("1D", "5D"), turnover_periods=(1,), include_pyfolio=False)

    expected_weights = performance.factor_weights(source).to_frame("factor")
    expected_returns = performance.factor_returns(source)
    expected_alpha_beta = performance.factor_alpha_beta(source, returns=expected_returns)
    expected_mean, expected_std = performance.mean_return_by_quantile(source)
    expected_by_date, _ = performance.mean_return_by_quantile(source, by_date=True)
    expected_ic = performance.factor_information_coefficient(source)

    pd.testing.assert_frame_equal(model.factor_weights, expected_weights)
    pd.testing.assert_frame_equal(model.factor_returns, expected_returns)
    pd.testing.assert_frame_equal(model.alpha_beta, expected_alpha_beta)
    pd.testing.assert_frame_equal(model.mean_returns_by_quantile, expected_mean)
    pd.testing.assert_frame_equal(model.std_error_by_quantile, expected_std)
    pd.testing.assert_frame_equal(model.mean_returns_by_date, expected_by_date)
    pd.testing.assert_frame_equal(model.information_coefficient, expected_ic)
    pd.testing.assert_series_equal(model.mean_information_coefficient, expected_ic.mean())
    for period in model.forward_periods:
        pd.testing.assert_series_equal(
            model.factor_cumulative_returns[period],
            portfolio.factor_cumulative_returns(source, period),
        )
        pd.testing.assert_frame_equal(
            model.factor_positions[period],
            portfolio.factor_positions(source, period),
        )

    expected_statistics = source.groupby("factor_quantile", observed=False, sort=True)["factor"].agg(
        ["min", "max", "mean", "std", "count"]
    )
    expected_statistics["count %"] = expected_statistics["count"] / expected_statistics["count"].sum() * 100.0
    pd.testing.assert_frame_equal(model.quantile_statistics, expected_statistics)
    assert tuple(model.quantile_turnover) == (1,)
    assert list(model.rank_autocorrelation.columns) == [1]
    assert model.pyfolio_inputs is None


def test_config_and_result_fingerprints_cover_options_and_input(
    clean_factor_data: pd.DataFrame,
    prices: pd.DataFrame,
) -> None:
    """All compute-affecting options and the owned input snapshot change fingerprints."""

    from fincore.factor_analysis.analysis import analyze_factor
    from fincore.factor_analysis.models import FactorAnalysisConfig

    base = FactorAnalysisConfig(periods=("1D", "5D"))
    variants = (
        FactorAnalysisConfig(long_short=False, periods=("1D", "5D")),
        FactorAnalysisConfig(group_neutral=True, periods=("1D", "5D")),
        FactorAnalysisConfig(equal_weight=True, periods=("1D", "5D")),
        FactorAnalysisConfig(by_group=True, periods=("1D", "5D")),
        FactorAnalysisConfig(periods=("5D",)),
        FactorAnalysisConfig(periods=("1D", "5D"), event_before=1, event_after=2),
        FactorAnalysisConfig(periods=("1D", "5D"), turnover_periods=(2,)),
        FactorAnalysisConfig(periods=("1D", "5D"), time_aggregation=("W",)),
        FactorAnalysisConfig(periods=("1D", "5D"), include_pyfolio=False),
        FactorAnalysisConfig(periods=("1D", "5D"), pyfolio_capital=100_000.0),
        FactorAnalysisConfig(periods=("1D", "5D"), pyfolio_benchmark_period="5D"),
    )
    assert len({base.fingerprint, *(item.fingerprint for item in variants)}) == len(variants) + 1

    model = analyze_factor(clean_factor_data, periods=("1D",), turnover_periods=(1,), include_pyfolio=False)
    equivalent = analyze_factor(clean_factor_data, periods=("1D",), turnover_periods=(1,), include_pyfolio=False)
    changed_input = clean_factor_data.copy(deep=True)
    changed_input.iloc[0, changed_input.columns.get_loc("factor")] += 0.25
    changed = analyze_factor(changed_input, periods=("1D",), turnover_periods=(1,), include_pyfolio=False)

    assert (
        model.config.fingerprint != changed.config.fingerprint or model.result_fingerprint != changed.result_fingerprint
    )
    assert model.result_fingerprint != changed.result_fingerprint
    assert model.result_fingerprint == equivalent.result_fingerprint

    next_representable = clean_factor_data.copy(deep=True)
    factor_column = next_representable.columns.get_loc("factor")
    next_representable.iloc[0, factor_column] = np.nextafter(float(next_representable.iloc[0, factor_column]), np.inf)
    next_model = analyze_factor(
        next_representable,
        periods=("1D",),
        turnover_periods=(1,),
        include_pyfolio=False,
    )

    assert model.result_fingerprint != next_model.result_fingerprint
    assert len(model.config.fingerprint) == 64
    assert len(model.result_fingerprint) == 64

    event_input = _event_returns(prices)
    incomplete_event = analyze_factor(
        clean_factor_data,
        periods=("1D",),
        turnover_periods=(1,),
        include_pyfolio=False,
        event_returns=event_input,
    )
    changed_event_input = event_input.copy(deep=True)
    changed_event_input.iloc[1, 0] += 0.01
    changed_incomplete_event = analyze_factor(
        clean_factor_data,
        periods=("1D",),
        turnover_periods=(1,),
        include_pyfolio=False,
        event_returns=changed_event_input,
    )
    assert incomplete_event.result_fingerprint != changed_incomplete_event.result_fingerprint


def test_config_owns_sequence_options_and_rejects_lossy_integer_capital() -> None:
    """A frozen config cannot retain caller-owned lists or rounded capital."""

    from fincore.factor_analysis.models import FactorAnalysisConfig

    periods = ["1D"]
    turnover_periods = [1]
    aggregations = ["M"]
    config = FactorAnalysisConfig(
        periods=periods,  # type: ignore[arg-type]
        turnover_periods=turnover_periods,  # type: ignore[arg-type]
        time_aggregation=aggregations,  # type: ignore[arg-type]
        pyfolio_capital=2**53,
    )
    fingerprint = config.fingerprint
    periods.append("5D")
    turnover_periods.append(2)
    aggregations.append("W")

    assert config.periods == ("1D",)
    assert config.turnover_periods == (1,)
    assert config.time_aggregation == ("M",)
    assert config.pyfolio_capital == 2**53
    assert config.fingerprint == fingerprint

    with pytest.raises(ValueError, match="exactly"):
        FactorAnalysisConfig(pyfolio_capital=2**53 + 1)


def test_model_exposes_defensive_snapshots_for_all_renderer_data(
    clean_factor_data: pd.DataFrame,
    prices: pd.DataFrame,
) -> None:
    """Public table access cannot mutate the frozen model or its provenance."""

    from fincore.factor_analysis.analysis import analyze_factor

    model = analyze_factor(
        clean_factor_data,
        periods=("1D",),
        turnover_periods=(1,),
        by_group=True,
        include_pyfolio=True,
        event_returns=_event_returns(prices),
        event_before=1,
        event_after=2,
    )
    fingerprint = model.result_fingerprint
    factor_snapshot = model.factor_data.copy(deep=True)
    cumulative_snapshot = model.factor_cumulative_returns["1D"].copy(deep=True)
    group_key = next(iter(model.grouped_results))
    group_snapshot = model.grouped_results[group_key].factor_returns.copy(deep=True)
    assert model.event_returns is not None
    event_snapshot = model.event_returns.event_windows.copy(deep=True)
    assert model.pyfolio_inputs is not None
    positions_snapshot = model.pyfolio_inputs.positions.copy(deep=True)

    changed_factor_data = model.factor_data
    changed_factor_data.iloc[0, changed_factor_data.columns.get_loc("factor")] = 999.0
    changed_cumulative = model.factor_cumulative_returns["1D"]
    changed_cumulative.iloc[0] = 999.0
    changed_group = model.grouped_results[group_key].factor_returns
    changed_group.iloc[0, 0] = 999.0
    assert model.event_returns is not None
    changed_event = model.event_returns.event_windows
    changed_event.iloc[0, 0] = 999.0
    assert model.pyfolio_inputs is not None
    changed_positions = model.pyfolio_inputs.positions
    changed_positions.iloc[0, 0] = 999.0

    pd.testing.assert_frame_equal(model.factor_data, factor_snapshot)
    pd.testing.assert_series_equal(model.factor_cumulative_returns["1D"], cumulative_snapshot)
    pd.testing.assert_frame_equal(model.grouped_results[group_key].factor_returns, group_snapshot)
    assert model.event_returns is not None
    pd.testing.assert_frame_equal(model.event_returns.event_windows, event_snapshot)
    assert model.pyfolio_inputs is not None
    pd.testing.assert_frame_equal(model.pyfolio_inputs.positions, positions_snapshot)
    assert model.result_fingerprint == fingerprint
    with pytest.raises(TypeError):
        model.factor_positions["new"] = pd.DataFrame()  # type: ignore[index]


def test_serializable_handoff_round_trips_exact_values_keys_and_pandas_metadata() -> None:
    """JSON handoff retains adjacent floats, typed keys, categories, and timezone."""

    from fincore.factor_analysis.models import deserialize_serializable_value, serializable_value

    index = pd.date_range("2024-03-08 09:30", periods=2, freq="D", tz="America/New_York", name="when")
    frame = pd.DataFrame(
        {
            "factor": [np.nextafter(1.0, np.inf), 2.0],
            "group": pd.Categorical(["b", "a"], categories=["unused", "b", "a"], ordered=True),
        },
        index=index,
    )
    source = {1: "integer-key", "1": "text-key", "frame": frame}

    payload = serializable_value(source)
    restored = deserialize_serializable_value(json.loads(json.dumps(payload, allow_nan=False)))

    assert isinstance(restored, Mapping)
    assert restored[1] == "integer-key"
    assert restored["1"] == "text-key"
    pd.testing.assert_frame_equal(restored["frame"], frame)


def test_group_and_event_sections_are_optional_typed_models(
    clean_factor_data: pd.DataFrame,
    prices: pd.DataFrame,
) -> None:
    """Missing optional inputs omit their sections without leaking untyped dictionaries."""

    from fincore.factor_analysis.analysis import analyze_factor
    from fincore.factor_analysis.models import EventAnalysisModel, FactorGroupAnalysis

    without_group = clean_factor_data.drop(columns="group")
    no_group = analyze_factor(
        without_group,
        periods=("1D",),
        by_group=True,
        turnover_periods=(1,),
        include_pyfolio=False,
    )
    assert no_group.grouped_results == {}
    assert no_group.event_returns is None

    grouped = analyze_factor(
        clean_factor_data,
        periods=("1D",),
        by_group=True,
        turnover_periods=(1,),
        include_pyfolio=False,
    )
    assert set(grouped.grouped_results) == set(clean_factor_data["group"].unique())
    assert all(isinstance(item, FactorGroupAnalysis) for item in grouped.grouped_results.values())

    event = analyze_factor(
        clean_factor_data,
        periods=("1D",),
        turnover_periods=(1,),
        include_pyfolio=False,
        event_returns=_event_returns(prices),
        event_before=1,
        event_after=2,
    )
    assert isinstance(event.event_returns, EventAnalysisModel)
    assert not event.event_returns.event_windows.empty
    assert not event.event_returns.mean_returns.empty
    assert isinstance(event.event_returns.return_distribution, pd.Series)
    assert not event.event_returns.return_distribution.empty


def test_model_is_frozen_json_serializable_and_contains_no_render_objects(clean_factor_data: pd.DataFrame) -> None:
    """Task 6 ends in renderer-ready data, never figures, axes, or executable cache state."""

    from fincore.factor_analysis.analysis import analyze_factor

    model = analyze_factor(clean_factor_data, periods=("1D",), turnover_periods=(1,), include_pyfolio=False)

    with pytest.raises(FrozenInstanceError):
        model.config = model.config  # type: ignore[misc]
    assert not hasattr(model, "cache")
    _assert_serializable_data_only(model)
    payload = model.to_serializable()
    assert (
        json.loads(json.dumps(payload, sort_keys=True, allow_nan=False))["result_fingerprint"]
        == model.result_fingerprint
    )


def test_pyfolio_bridge_is_optional_typed_data_not_a_renderer(clean_factor_data: pd.DataFrame) -> None:
    """The model may include the Task 5 bridge without importing external Pyfolio."""

    from fincore.factor_analysis.analysis import analyze_factor
    from fincore.factor_analysis.portfolio import PyfolioFactorInputs

    model = analyze_factor(
        clean_factor_data,
        periods=("1D",),
        turnover_periods=(1,),
        include_pyfolio=True,
        pyfolio_capital=100_000.0,
        pyfolio_benchmark_period="5D",
    )

    assert isinstance(model.pyfolio_inputs, PyfolioFactorInputs)
    _assert_serializable_data_only(model.pyfolio_inputs)
    bridge_payload = json.loads(json.dumps(model.to_serializable(), sort_keys=True, allow_nan=False))["pyfolio_inputs"]
    assert set(bridge_payload) == {"benchmark_rets", "positions", "returns"}
