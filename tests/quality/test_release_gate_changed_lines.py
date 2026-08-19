"""Targeted coverage for lines changed since the Task 1 quality baseline.

``scripts/check_coverage_baseline.py`` requires >= 95% coverage of the lines
changed in ``fincore/**`` since the baseline commit.  These tests exercise
the remaining defensive/error branches of the converged validation,
dispatch, plugin-registry, engine and reporting layers so the release gate
stays enforceable without widening the metric.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
import pytest

from fincore import Empyrical
from fincore.contracts import validation as contracts_validation
from fincore.contracts.portfolio import ExposureBundle, VolumeExposureBundle
from fincore.contracts.time_series import validate_time_series_timezones
from fincore.exceptions import DataAlignmentError, MissingDataError, NumericalError, ValidationError

# =============================================================================
# contracts/validation.py — schema rejection branches
# =============================================================================


def test_returns_schema_rejects_range_index_when_datetime_required() -> None:
    value = pd.Series([0.01, 0.02], index=[0, 1])
    with pytest.raises(ValidationError, match="DatetimeIndex"):
        contracts_validation.validate_returns_schema(value, require_datetime_index=True)


def test_returns_schema_rejects_duplicate_labels() -> None:
    index = pd.to_datetime(["2024-01-02", "2024-01-02"])
    with pytest.raises(DataAlignmentError, match="duplicate"):
        contracts_validation.validate_returns_schema(pd.Series([0.01, 0.02], index=index))


def test_returns_schema_rejects_unsorted_index() -> None:
    index = pd.to_datetime(["2024-01-03", "2024-01-01"])
    with pytest.raises(DataAlignmentError, match="sorted"):
        contracts_validation.validate_returns_schema(pd.Series([0.01, 0.02], index=index))


def test_require_finite_rejects_object_arrays() -> None:
    with pytest.raises(ValidationError, match="numeric"):
        contracts_validation._require_finite(np.array(["a", "b"]), name="returns")


def test_returns_schema_rejects_frames_when_disallowed() -> None:
    frame = pd.DataFrame({"a": [0.01]}, index=pd.to_datetime(["2024-01-02"]))
    with pytest.raises(ValidationError, match="one-dimensional"):
        contracts_validation.validate_returns_schema(frame, allow_frame=False)


def test_returns_schema_rejects_non_numeric_frame() -> None:
    frame = pd.DataFrame({"a": ["x"]}, index=pd.to_datetime(["2024-01-02"]))
    with pytest.raises(ValidationError, match="numeric"):
        contracts_validation.validate_returns_schema(frame, allow_frame=True)


def test_returns_schema_rejects_empty_series_and_non_numeric_series() -> None:
    with pytest.raises(ValidationError, match="empty"):
        contracts_validation.validate_returns_schema(pd.Series([], dtype=float))
    with pytest.raises(ValidationError, match="numeric"):
        contracts_validation.validate_returns_schema(pd.Series(["a", "b"]))


def test_returns_schema_array_shape_and_emptiness() -> None:
    with pytest.raises(ValidationError, match="one-dimensional"):
        contracts_validation.validate_returns_schema(np.array(1.0), allow_array=True)
    with pytest.raises(ValidationError, match="empty"):
        contracts_validation.validate_returns_schema(np.array([]), allow_array=True)


def test_returns_schema_rejects_unknown_types_and_missing() -> None:
    with pytest.raises(MissingDataError):
        contracts_validation.validate_returns_schema(None)
    with pytest.raises(ValidationError, match="numeric pandas Series"):
        contracts_validation.validate_returns_schema("not a series")


def test_positions_schema_rejection_branches() -> None:
    with pytest.raises(ValidationError, match="Series or DataFrame"):
        contracts_validation.validate_positions_schema("nope")
    with pytest.raises(ValidationError, match="empty"):
        contracts_validation.validate_positions_schema(pd.DataFrame())
    frame = pd.DataFrame(
        [[1.0, 2.0]],
        index=pd.to_datetime(["2024-01-02"]),
        columns=["a", "a"],
    )
    with pytest.raises(ValidationError, match="duplicate columns"):
        contracts_validation.validate_positions_schema(frame)
    cash_dup = pd.DataFrame(
        [[1.0, 2.0]],
        index=pd.to_datetime(["2024-01-02"]),
        columns=["cash", "CASH"],
    )
    with pytest.raises(ValidationError, match="duplicate cash"):
        contracts_validation.validate_positions_schema(cash_dup)
    no_cash = pd.DataFrame(
        [[1.0]],
        index=pd.to_datetime(["2024-01-02"]),
        columns=["asset"],
    )
    with pytest.raises(ValidationError, match="cash"):
        contracts_validation.validate_positions_schema(no_cash, require_cash=True)
    with pytest.raises(ValidationError, match="numeric"):
        contracts_validation.validate_positions_schema(
            pd.Series(["a", "b"], index=pd.to_datetime(["2024-01-02", "2024-01-03"]))
        )


def test_multiindex_validation_branches() -> None:
    unnamed = pd.DataFrame(
        {"value": [1.0, 2.0]},
        index=pd.MultiIndex.from_arrays(
            [pd.to_datetime(["2024-01-02", "2024-01-03"]), ["a", "b"]],
            names=[None, None],
        ),
    )
    with pytest.raises(ValidationError, match="levels must be named"):
        contracts_validation.validate_positions_schema(unnamed)
    undated = pd.DataFrame(
        {"value": [1.0, 2.0]},
        index=pd.MultiIndex.from_arrays([["x", "y"], ["a", "b"]], names=["date", "asset"]),
    )
    with pytest.raises(ValidationError, match="first MultiIndex level must be datetime"):
        contracts_validation.validate_positions_schema(undated)
    dup_index = pd.MultiIndex.from_arrays(
        [pd.to_datetime(["2024-01-02", "2024-01-02"]), ["a", "a"]],
        names=["date", "asset"],
    )
    with pytest.raises(DataAlignmentError, match="duplicate"):
        contracts_validation.validate_positions_schema(pd.DataFrame({"value": [1.0, 2.0]}, index=dup_index))


def test_transactions_schema_rejection_branches() -> None:
    with pytest.raises(ValidationError, match="DataFrame"):
        contracts_validation.validate_transactions_schema(pd.Series([1.0]))
    with pytest.raises(ValidationError, match="empty"):
        contracts_validation.validate_transactions_schema(pd.DataFrame())
    dup = pd.DataFrame(
        [[1.0, 2.0]],
        columns=["a", "a"],
    )
    with pytest.raises(ValidationError, match="duplicate columns"):
        contracts_validation.validate_transactions_schema(dup)
    with pytest.raises(ValidationError, match="missing required columns"):
        contracts_validation.validate_transactions_schema(pd.DataFrame({"x": [1.0]}))
    with pytest.raises(ValidationError, match="numeric"):
        contracts_validation.validate_transactions_schema(
            pd.DataFrame({"amount": ["x"], "price": [1.0], "symbol": ["AAPL"]})
        )
    with pytest.raises(MissingDataError, match="symbol"):
        contracts_validation.validate_transactions_schema(
            pd.DataFrame({"amount": [1.0], "price": [1.0], "symbol": [np.nan]})
        )


def test_factor_schema_rejection_branches() -> None:
    with pytest.raises(ValidationError, match="Series or DataFrame"):
        contracts_validation.validate_factors_schema(42)
    with pytest.raises(ValidationError, match="nonempty unique columns"):
        contracts_validation.validate_factors_schema(pd.DataFrame())
    dup = pd.DataFrame([[1.0, 2.0]], columns=["a", "a"])
    with pytest.raises(ValidationError, match="nonempty unique columns"):
        contracts_validation.validate_factors_schema(dup)


def test_market_data_schema_rejection_branches() -> None:
    with pytest.raises(ValidationError, match="mapping"):
        contracts_validation.validate_market_data_schema(["not", "a", "mapping"])
    with pytest.raises(ValidationError, match="missing required entries"):
        contracts_validation.validate_market_data_schema({"price": pd.DataFrame()})
    base = pd.DataFrame(
        {"asset": [1.0, 2.0]},
        index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
    )
    with pytest.raises(ValidationError, match="DataFrame"):
        contracts_validation.validate_market_data_schema({"price": base, "volume": "nope"})
    shifted = pd.DataFrame(
        {"asset": [1.0, 2.0]},
        index=pd.to_datetime(["2024-01-03", "2024-01-04"]),
    )
    with pytest.raises(DataAlignmentError, match="indices must match"):
        contracts_validation.validate_market_data_schema({"price": base, "volume": shifted})
    other_cols = pd.DataFrame(
        {"other": [1.0, 2.0]},
        index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
    )
    with pytest.raises(DataAlignmentError, match="columns must match"):
        contracts_validation.validate_market_data_schema({"price": base, "volume": other_cols})
    with pytest.raises(ValidationError, match="cannot be negative"):
        contracts_validation.validate_market_data_schema({"price": base, "volume": -base})


def test_validate_overlap_empty_and_by_date() -> None:
    reference = pd.Series([], dtype=float)
    value = pd.Series([0.01], index=pd.to_datetime(["2024-01-02"]))
    with pytest.raises(DataAlignmentError, match="nonempty overlap"):
        contracts_validation._validate_overlap(reference, value, name="factor_returns")
    with pytest.raises(DataAlignmentError, match="datetime labels"):
        contracts_validation._validate_overlap(
            pd.Series([0.01], index=[0]),
            pd.Series([0.01], index=[0]),
            name="factor_returns",
            by_date=True,
        )


# =============================================================================
# contracts/portfolio.py and contracts/time_series.py
# =============================================================================


def test_exposure_components_rejection_branches() -> None:
    frame = pd.DataFrame(
        {"a": [1.0, 2.0], "b": [0.5, 0.5]},
        index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
    )
    with pytest.raises(ValidationError, match="DataFrame"):
        ExposureBundle(long="nope", short=frame, gross=frame, net=frame)  # type: ignore[arg-type]
    dup = pd.DataFrame([[1.0, 2.0]], index=pd.to_datetime(["2024-01-02"]), columns=["a", "a"])
    with pytest.raises(ValidationError, match="unique"):
        ExposureBundle(long=dup, short=frame, gross=frame, net=frame)
    other_index = pd.DataFrame(
        {"a": [1.0], "b": [1.0]},
        index=pd.to_datetime(["2024-01-04"]),
    )
    with pytest.raises(ValidationError, match="same index"):
        ExposureBundle(long=frame, short=other_index, gross=frame, net=frame)
    other_cols = pd.DataFrame(
        {"x": [1.0, 2.0]},
        index=frame.index,
    )
    with pytest.raises(ValidationError, match="same category columns"):
        ExposureBundle(long=frame, short=other_cols, gross=frame, net=frame)


def test_volume_exposure_components_rejection_branches() -> None:
    series = pd.Series([1.0, 2.0], index=pd.to_datetime(["2024-01-02", "2024-01-03"]))
    with pytest.raises(ValidationError, match="Series"):
        VolumeExposureBundle(long="nope", short=series, gross=series)  # type: ignore[arg-type]
    other_index = pd.Series([1.0], index=pd.to_datetime(["2024-01-04"]))
    with pytest.raises(ValidationError, match="same index"):
        VolumeExposureBundle(long=series, short=other_index, gross=series)


def test_time_series_alignment_policy_validation() -> None:
    from fincore.contracts.time_series import _validate_alignment_policy, align_time_series

    with pytest.raises(ValueError, match="alignment policy"):
        _validate_alignment_policy("bogus")  # type: ignore[arg-type]
    assert align_time_series(policy="inner") == ()
    assert validate_time_series_timezones() is None


# =============================================================================
# plugin/registry.py — registration lifecycle branches
# =============================================================================


def test_plugin_registry_lifecycle_branches() -> None:
    from fincore.plugin.registry import ExtensionKind, ExtensionRegistry, Scope

    scratch = ExtensionRegistry()
    with pytest.raises(ValueError, match="Unknown extension kind"):
        scratch.register("metrics", "x", lambda: None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unknown extension kind"):
        scratch.unregister("metrics", "nope")  # type: ignore[arg-type]
    with pytest.raises(KeyError, match="is registered"):
        scratch.unregister(ExtensionKind.METRIC, "nope", raise_if_missing=True)

    scratch.register(ExtensionKind.METRIC, "gate_metric", lambda x: x, scope=Scope.GLOBAL)
    scratch.register(ExtensionKind.METRIC, "gate_other", lambda x: x, scope=Scope.BUILTIN, family="alt")
    scratch.register(ExtensionKind.VIZ_BACKEND, "gate_viz", object, scope=Scope.GLOBAL)
    scratch.register(ExtensionKind.HOOK, "gate_event", lambda: None, scope=Scope.GLOBAL)
    assert "gate_metric" in scratch.metric_names()
    removed = scratch.unregister(ExtensionKind.METRIC, "gate_metric")
    assert removed is not None
    assert scratch.unregister(ExtensionKind.HOOK, "gate_event") is not None
    assert scratch.unregister(ExtensionKind.VIZ_BACKEND, "gate_viz") is not None
    assert scratch.unregister(ExtensionKind.HOOK, "gate_event") is None


def test_plugin_registry_clear_and_isolated() -> None:
    from fincore.plugin.registry import ExtensionKind, ExtensionRegistry, Scope

    scratch = ExtensionRegistry()
    scratch.register(ExtensionKind.METRIC, "gate_metric", lambda x: x, scope=Scope.GLOBAL)
    scratch.register(ExtensionKind.METRIC, "gate_other", lambda x: x, scope=Scope.BUILTIN)
    scratch.register(ExtensionKind.VIZ_BACKEND, "gate_viz", object, scope=Scope.GLOBAL)
    scratch.register(ExtensionKind.HOOK, "gate_event", lambda: None, scope=Scope.GLOBAL)

    scratch.clear(ExtensionKind.METRIC, scope=Scope.GLOBAL)
    assert "gate_metric" not in scratch.metric_names()
    assert "gate_other" in scratch.metric_names()
    scratch.clear(ExtensionKind.METRIC, name="gate_other", include_builtins=True)
    assert "gate_other" not in scratch.metric_names()
    scratch.clear(ExtensionKind.METRIC, family="nonexistent")
    scratch.clear(ExtensionKind.VIZ_BACKEND, scope=Scope.GLOBAL)
    scratch.clear(ExtensionKind.VIZ_BACKEND, name="gate_viz")
    scratch.clear(ExtensionKind.HOOK, scope=Scope.GLOBAL)
    scratch.clear(ExtensionKind.HOOK, name="gate_event")
    scratch.clear()  # kind=None clears every non-builtin scope

    with scratch.isolated():
        scratch.register(ExtensionKind.METRIC, "gate_transient", lambda x: x, scope=Scope.GLOBAL)
        assert "gate_transient" in scratch.metric_names()
    assert "gate_transient" not in scratch.metric_names()


def test_plugin_clear_unknown_registry_type() -> None:
    from fincore.plugin.registry import clear_registry

    with pytest.raises(ValueError, match="Unknown registry type"):
        clear_registry(registry_type="bogus")


# =============================================================================
# core/engine.py + core/rolling_moments.py — rolling engine kernels
# =============================================================================


def test_rolling_engine_all_metrics_and_beta_without_factor() -> None:
    from fincore.core.engine import RollingEngine

    index = pd.bdate_range("2024-01-01", periods=300)
    returns = pd.Series(np.random.default_rng(1).normal(0.001, 0.01, len(index)), index=index)
    factor = pd.Series(np.random.default_rng(2).normal(0.0005, 0.008, len(index)), index=index)
    engine = RollingEngine(returns, factor_returns=factor, window=63)
    results = engine.compute(["sharpe", "volatility", "beta", "sortino", "mean_return", "max_drawdown"])
    assert set(results) == {"sharpe", "volatility", "beta", "sortino", "mean_return", "max_drawdown"}
    assert all(not results[name].dropna().empty for name in results)

    lone = RollingEngine(returns, window=63)
    with pytest.raises(ValueError, match="factor_returns required"):
        lone.compute(["beta"])


def test_rolling_moments_missing_moment_error() -> None:
    from fincore.core.rolling_moments import RollingMoments, sharpe_from_moments

    with pytest.raises(ValueError, match="not built"):
        sharpe_from_moments(RollingMoments(window=63), 252.0, np.sqrt(252.0))


# =============================================================================
# metrics/rolling.py — aligned out-parameter paths
# =============================================================================


def test_rolling_aligned_kernels_write_out() -> None:
    from fincore.metrics.rolling import (
        roll_alpha_aligned,
        roll_alpha_beta_aligned,
        roll_annual_volatility,
        roll_beta_aligned,
        roll_sortino_ratio,
    )

    index = pd.bdate_range("2024-01-01", periods=100)
    returns = pd.Series(np.random.default_rng(3).normal(0.001, 0.01, len(index)), index=index)
    factor = pd.Series(np.random.default_rng(4).normal(0.0005, 0.008, len(index)), index=index)
    window = 20
    out_alpha = np.full(roll_alpha_aligned(returns, factor, window).shape, np.nan)
    assert roll_alpha_aligned(returns, factor, window, out=out_alpha) is out_alpha
    out_beta = np.full(roll_beta_aligned(returns, factor, window).shape, np.nan)
    assert roll_beta_aligned(returns, factor, window, out=out_beta) is out_beta
    out_ab = roll_alpha_beta_aligned(returns, factor, window)
    out_ab_buffer = np.full(out_ab.shape, np.nan)
    assert roll_alpha_beta_aligned(returns, factor, window, out=out_ab_buffer) is out_ab_buffer
    out_vol = np.full(roll_annual_volatility(returns, window).shape, np.nan)
    assert roll_annual_volatility(returns, window, out=out_vol) is out_vol
    out_sortino = np.full(roll_sortino_ratio(returns, window).shape, np.nan)
    assert roll_sortino_ratio(returns, window, out=out_sortino) is out_sortino


# =============================================================================
# empyrical.py — legacy capture/risk adapters and strict helpers
# =============================================================================


def test_legacy_capture_adapters() -> None:
    index = pd.bdate_range("2024-01-01", periods=60)
    returns = pd.Series(np.random.default_rng(5).normal(0.001, 0.01, len(index)), index=index)
    factor = pd.Series(np.random.default_rng(6).normal(0.0005, 0.008, len(index)), index=index)
    assert np.isfinite(Empyrical.capture(returns, factor))
    assert np.isfinite(Empyrical.up_capture(returns, factor))
    assert np.isfinite(Empyrical.down_capture(returns, factor))
    assert np.isfinite(Empyrical.up_down_capture(returns, factor))
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        Empyrical.capture(returns, factor, bogus_kwarg=1)


def test_legacy_risk_adapters() -> None:
    from fincore._empyrical_legacy import _legacy_rolling_window

    index = pd.bdate_range("2024-01-01", periods=60)
    returns = pd.Series(np.random.default_rng(7).normal(0.001, 0.01, len(index)), index=index)
    assert np.isfinite(Empyrical.calmar_ratio(returns))
    assert np.isfinite(Empyrical.value_at_risk(returns, cutoff=0.05))
    assert np.isfinite(Empyrical.conditional_value_at_risk(returns, cutoff=0.05))
    out = np.empty(())
    result = Empyrical.max_drawdown(returns, out=out)
    assert result == out.item()
    with pytest.raises(ValueError, match="0-length window"):
        _legacy_rolling_window(returns.to_numpy(), 0)
    with pytest.raises(IndexError, match="scalar"):
        _legacy_rolling_window(np.array(1.0), 1)
    with pytest.raises(IndexError, match="window length"):
        _legacy_rolling_window(np.ones(3), 4)


# =============================================================================
# report/artifacts.py — figure close deduplication
# =============================================================================


def test_report_artifacts_close_deduplicates_and_tolerates_non_figures() -> None:
    from fincore.report.artifacts import ReportArtifacts

    artifacts = ReportArtifacts(backend="test", figures=[object(), object()])
    artifacts.close()
    shared = object()
    artifacts = ReportArtifacts(backend="test", figures=[shared, shared])
    artifacts.close()


# =============================================================================
# report/model.py — classification of series and nested mappings
# =============================================================================


def test_classify_mapping_series_and_nested() -> None:
    from fincore.report.model import _classify_mapping

    series = pd.Series([1.0, 2.0])
    model = _classify_mapping(
        "root",
        {
            "table": pd.DataFrame({"a": [1.0]}),
            "series": series,
            "nested": {"scalar": 3.0, "inner_table": pd.DataFrame({"b": [2.0]})},
            "meta_scalar": 42,
        },
    )
    assert set(model.tables) == {"table", "inner_table"}
    assert set(model.series) == {"series"}
    assert model.metrics["nested.scalar"] == 3.0
    assert model.metrics["meta_scalar"] == 42


# =============================================================================
# metrics/returns.py — ndarray path and weekly grouping policies
# =============================================================================


def test_simple_returns_ndarray_and_aggregate_weekly_policies() -> None:
    from fincore.metrics.returns import aggregate_returns, simple_returns

    prices = np.array([100.0, 110.0, 99.0])
    result = simple_returns(prices)
    assert result.shape == (2,)
    index = pd.bdate_range("2024-01-01", periods=40)
    returns = pd.Series(np.random.default_rng(8).normal(0.001, 0.01, len(index)), index=index)
    calendar = aggregate_returns(returns, convert_to="weekly", week_year="calendar")
    iso = aggregate_returns(returns, convert_to="weekly", week_year="iso")
    assert not calendar.empty and not iso.empty
    with pytest.raises(ValueError, match="week_year"):
        aggregate_returns(returns, convert_to="weekly", week_year="bogus")


# =============================================================================
# metrics/drawdown.py + metrics/ratios.py + metrics/yearly.py
# =============================================================================


def test_drawdown_table_without_underwater_period() -> None:
    from fincore.metrics.drawdown import _get_max_drawdown_positions, get_max_drawdown_underwater

    underwater = pd.Series(0.0, index=pd.to_datetime(["2024-01-02", "2024-01-03"]))
    assert _get_max_drawdown_positions(underwater) is None
    result = get_max_drawdown_underwater(underwater)
    assert result[0] is pd.NaT and result[1] is pd.NaT and result[2] is pd.NaT


def test_information_ratio_zero_tracking_error_returns_nan() -> None:
    from fincore.metrics.ratios import information_ratio

    index = pd.bdate_range("2024-01-01", periods=20)
    constant = pd.Series(0.01, index=index)
    result = information_ratio(constant, constant)
    assert np.isnan(result)


def test_information_ratio_per_year_array() -> None:
    from fincore.metrics.yearly import information_ratio_by_year

    index = pd.bdate_range("2023-06-01", periods=300)
    returns = pd.Series(np.random.default_rng(9).normal(0.001, 0.01, len(index)), index=index)
    factor = pd.Series(np.random.default_rng(10).normal(0.0005, 0.008, len(index)), index=index)
    as_array = information_ratio_by_year(returns.to_numpy(), factor.to_numpy())
    assert isinstance(as_array, np.ndarray)
    as_series = information_ratio_by_year(returns, factor)
    assert not as_series.empty


# =============================================================================
# metrics/transactions.py — nested transaction records
# =============================================================================


def test_transaction_records_rejection_branches() -> None:
    from fincore.metrics.transactions import make_transaction_frame, map_transaction

    nested_sid = {
        "dt": pd.Timestamp("2024-01-02"),
        "sid": {"symbol": "AAPL"},
        "amount": 1,
        "price": 1.0,
        "commission": 0.0,
        "order_id": "x",
    }
    with pytest.raises(ValidationError, match="sid"):
        map_transaction(nested_sid)
    with pytest.raises(ValidationError, match="mappings"):
        make_transaction_frame({"2024-01-02": ["not-a-mapping"]})
    with pytest.raises(ValidationError, match="mappings or date-to-list"):
        make_transaction_frame({"2024-01-02": 42})


# =============================================================================
# core/context.py — empty snapshot cached properties
# =============================================================================


def test_context_leverage_and_turnover_without_snapshots() -> None:
    from fincore.core.context import AnalysisContext

    index = pd.bdate_range("2024-01-01", periods=20)
    returns = pd.Series(np.random.default_rng(11).normal(0.001, 0.01, len(index)), index=index)
    ctx = AnalysisContext(returns)
    assert ctx.gross_leverage.empty
    assert ctx.turnover.empty


# =============================================================================
# pyfolio.py — module attribute protocol
# =============================================================================


def test_pyfolio_module_missing_attribute() -> None:
    import fincore.pyfolio as pyfolio

    with pytest.raises(AttributeError, match="has no attribute"):
        _ = pyfolio.definitely_not_a_pyfolio_symbol
    assert "Pyfolio" in dir(pyfolio)


# =============================================================================
# metrics/perf_attrib.py — attribution alignment rejection branches
# =============================================================================


def _attribution_frames() -> tuple[pd.Series, pd.Series, pd.DataFrame, pd.DataFrame]:
    index = pd.bdate_range("2024-01-01", periods=30)
    returns = pd.Series(np.random.default_rng(12).normal(0.001, 0.01, len(index)), index=index)
    factor_returns = pd.DataFrame(
        {"momentum": np.random.default_rng(13).normal(0.0005, 0.008, len(index))}, index=index
    )
    positions = pd.DataFrame({"AAPL": 1.0}, index=index)
    loadings_index = pd.MultiIndex.from_product([index[:2], ["AAPL"]], names=["dt", "ticker"])
    loadings = pd.DataFrame({"momentum": [1.0, 1.0]}, index=loadings_index)
    return returns, factor_returns.iloc[:, 0], positions, loadings


def _loadings_for(dates: list[str]) -> pd.DataFrame:
    loadings_index = pd.MultiIndex.from_product(
        [pd.to_datetime(dates), ["AAPL"]],
        names=["dt", "ticker"],
    )
    return pd.DataFrame({"momentum": [1.0] * len(loadings_index)}, index=loadings_index)


def test_perf_attrib_rejection_branches() -> None:
    from fincore.metrics.perf_attrib import (
        _align_factor_columns,
        _normalize_attribution_index,
        _normalize_date_index,
        compute_exposures,
        perf_attrib,
    )

    _, factor_returns, _, loadings = _attribution_frames()
    overlap_returns = pd.Series([0.01, 0.01], index=pd.to_datetime(["2024-01-02", "2024-01-03"]))
    overlap_factor = pd.Series([0.001, 0.001], index=overlap_returns.index)
    far_positions = pd.DataFrame({"AAPL": [1.0]}, index=pd.to_datetime(["2024-01-05"]))
    far_loadings = _loadings_for(["2024-01-05"])
    with pytest.raises(DataAlignmentError, match="no common dates"):
        perf_attrib(overlap_returns, far_positions, overlap_factor.rename("momentum").to_frame(), far_loadings)
    with pytest.raises(ValueError, match="UTC"):
        _normalize_date_index(pd.DatetimeIndex(["2024-01-02"]), "EST")
    with pytest.raises(DataAlignmentError, match="duplicate attribution labels"):
        duplicate_index = pd.MultiIndex.from_arrays(
            [pd.to_datetime(["2024-01-02", "2024-01-02"]), ["a", "a"]],
            names=["date", "asset"],
        )
        _normalize_attribution_index(pd.Series([1.0, 2.0], index=duplicate_index), None)
    duplicate_columns = pd.DataFrame(
        [[0.001, 0.002]],
        index=pd.to_datetime(["2024-01-02"]),
        columns=["momentum", "momentum"],
    )
    with pytest.raises(DataAlignmentError, match="duplicate factor columns"):
        _align_factor_columns(duplicate_columns, loadings, policy="strict")
    with pytest.raises(DataAlignmentError, match="identical factor columns"):
        _align_factor_columns(
            factor_returns.to_frame().rename(columns={"momentum": "value"}),
            loadings,
            policy="strict",
        )
    with pytest.raises(TypeError, match="DataFrame or a stacked Series"):
        compute_exposures("not positions", loadings, stack_positions=True)


# =============================================================================
# _dispatch.py — contract and projection error paths
# =============================================================================


def test_dispatch_contract_and_projection_errors() -> None:
    import inspect
    from dataclasses import replace

    from fincore import _dispatch

    def no_out_kernel(returns):
        return returns

    spec = _dispatch.get_metric_spec("metrics", "sharpe_ratio", "enhanced")
    bad_spec = replace(spec, out_policy="write_and_return")
    with pytest.raises(ValueError, match="requires a kernel out parameter"):
        _dispatch._check_contract(bad_spec, inspect.signature(no_out_kernel))
    with pytest.raises(ValueError, match="unknown result projection"):
        _dispatch._apply_projection(replace(spec, result_projection="bogus"), 1.0)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="scalar"):
        _dispatch._apply_projection(replace(spec, result_projection="scalar"), pd.Series([1.0]))
    with pytest.raises(TypeError, match="Series"):
        _dispatch._apply_projection(replace(spec, result_projection="series"), 1.0)
    with pytest.raises(TypeError, match="DataFrame"):
        _dispatch._apply_projection(replace(spec, result_projection="frame"), 1.0)
    with pytest.raises(TypeError, match="tuple"):
        _dispatch._apply_projection(replace(spec, result_projection="legacy_tuple"), 1.0)


# =============================================================================
# fincore/validation.py — decorated validator framework
# =============================================================================


def test_validate_input_framework_error_and_limits() -> None:
    from fincore.validation import validate_input

    def _explode(_value: float) -> None:
        raise ValueError("boom")

    @validate_input(_explode)
    def wrapped(value: float) -> float:
        return value

    with pytest.raises(ValidationError, match="Input validation failed"):
        wrapped(1.0)

    @validate_input(_explode, _explode)
    def limited(only: float, extra: float = 0.0) -> float:
        return only + extra

    with pytest.raises(ValidationError, match="Input validation failed"):
        limited(1.0)
