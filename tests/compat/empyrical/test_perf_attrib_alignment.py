from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

import fincore.empyrical as ep
from fincore.exceptions import DataAlignmentError
from fincore.metrics.perf_attrib import perf_attrib


@dataclass(frozen=True)
class AttributionInputs:
    returns: pd.Series
    positions: pd.DataFrame
    factor_returns: pd.DataFrame
    factor_loadings: pd.DataFrame


def _inputs(
    *,
    return_dates: list[str],
    position_dates: list[str],
    factor_return_dates: list[str] | None = None,
) -> AttributionInputs:
    returns_index = pd.to_datetime(return_dates)
    positions_index = pd.to_datetime(position_dates)
    factor_returns_index = pd.to_datetime(factor_return_dates or position_dates)
    loadings_index = pd.MultiIndex.from_product(
        [positions_index, ["A"]],
        names=["dt", "ticker"],
    )
    return AttributionInputs(
        returns=pd.Series(np.linspace(0.10, 0.20, len(returns_index)), index=returns_index),
        positions=pd.DataFrame({"A": np.ones(len(positions_index))}, index=positions_index),
        factor_returns=pd.DataFrame(
            {"factor": np.linspace(0.01, 0.02, len(factor_returns_index))},
            index=factor_returns_index,
        ),
        factor_loadings=pd.DataFrame(
            {"factor": np.linspace(2.0, 3.0, len(loadings_index))},
            index=loadings_index,
        ),
    )


def _assert_daily_attribution_identity(perf_attribution: pd.DataFrame) -> None:
    np.testing.assert_allclose(
        perf_attribution["total_returns"],
        perf_attribution["common_returns"] + perf_attribution["specific_returns"],
    )


def test_enhanced_partial_date_intersection_uses_labels_not_equal_lengths() -> None:
    inputs = _inputs(
        return_dates=["2024-01-02", "2024-01-03"],
        position_dates=["2024-01-01", "2024-01-02"],
    )

    risk_exposures, attribution = perf_attrib(
        inputs.returns,
        inputs.positions,
        inputs.factor_returns,
        inputs.factor_loadings,
        pos_in_dollars=False,
        alignment="inner",
    )

    expected_index = pd.to_datetime(["2024-01-02"])
    pd.testing.assert_index_equal(risk_exposures.index, expected_index)
    pd.testing.assert_index_equal(attribution.index, expected_index)
    np.testing.assert_allclose(risk_exposures["factor"], [3.0])
    np.testing.assert_allclose(attribution["factor"], [0.06])
    np.testing.assert_allclose(attribution["common_returns"], [0.06])
    np.testing.assert_allclose(attribution["specific_returns"], [0.04])
    _assert_daily_attribution_identity(attribution)


def test_legacy_partial_dates_keep_pinned_outer_label_alignment_without_relabeling() -> None:
    inputs = _inputs(
        return_dates=["2024-01-02", "2024-01-03"],
        position_dates=["2024-01-01", "2024-01-02"],
    )

    positions = inputs.positions.stack()
    positions.index = positions.index.set_names(["dt", "ticker"])

    risk_exposures, attribution = ep.perf_attrib(
        inputs.returns,
        positions,
        inputs.factor_returns,
        inputs.factor_loadings,
    )

    pd.testing.assert_index_equal(
        risk_exposures.index,
        pd.DatetimeIndex(pd.to_datetime(["2024-01-01", "2024-01-02"]), name="dt"),
    )
    pd.testing.assert_index_equal(
        attribution.index,
        pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
    )
    assert risk_exposures.loc[pd.Timestamp("2024-01-01"), "factor"] == 0.0
    assert attribution.loc[pd.Timestamp("2024-01-02"), "factor"] == pytest.approx(0.06)
    assert attribution.loc[pd.Timestamp("2024-01-03"), "total_returns"] == pytest.approx(0.2)


def test_enhanced_no_common_dates_raise_alignment_error() -> None:
    inputs = _inputs(
        return_dates=["2024-02-01", "2024-02-02"],
        position_dates=["2024-01-01", "2024-01-02"],
    )

    with pytest.raises(DataAlignmentError, match="common"):
        perf_attrib(
            inputs.returns,
            inputs.positions,
            inputs.factor_returns,
            inputs.factor_loadings,
            pos_in_dollars=False,
            alignment="inner",
        )


def test_strict_attribution_rejects_partial_date_coverage() -> None:
    inputs = _inputs(
        return_dates=["2024-01-02", "2024-01-03"],
        position_dates=["2024-01-01", "2024-01-02"],
    )

    with pytest.raises(DataAlignmentError, match="strict"):
        perf_attrib(
            inputs.returns,
            inputs.positions,
            inputs.factor_returns,
            inputs.factor_loadings,
            pos_in_dollars=False,
            alignment="strict",
        )


def test_complete_attribution_preserves_inputs_and_daily_identity() -> None:
    inputs = _inputs(
        return_dates=["2024-01-01", "2024-01-02"],
        position_dates=["2024-01-01", "2024-01-02"],
    )
    before = AttributionInputs(
        returns=inputs.returns.copy(),
        positions=inputs.positions.copy(),
        factor_returns=inputs.factor_returns.copy(),
        factor_loadings=inputs.factor_loadings.copy(),
    )

    risk_exposures, attribution = perf_attrib(
        inputs.returns,
        inputs.positions,
        inputs.factor_returns,
        inputs.factor_loadings,
        pos_in_dollars=False,
        alignment="strict",
    )

    pd.testing.assert_index_equal(risk_exposures.index, inputs.returns.index)
    pd.testing.assert_index_equal(attribution.index, inputs.returns.index)
    _assert_daily_attribution_identity(attribution)
    pd.testing.assert_series_equal(inputs.returns, before.returns)
    pd.testing.assert_frame_equal(inputs.positions, before.positions)
    pd.testing.assert_frame_equal(inputs.factor_returns, before.factor_returns)
    pd.testing.assert_frame_equal(inputs.factor_loadings, before.factor_loadings)


def test_attribution_can_explicitly_normalize_timezone_indices_to_utc() -> None:
    utc_index = pd.date_range("2024-01-01", periods=2, tz="UTC")
    shanghai_index = utc_index.tz_convert("Asia/Shanghai")
    returns = pd.Series([0.10, 0.20], index=utc_index.tz_localize(None))
    positions = pd.DataFrame({"A": [1.0, 1.0]}, index=shanghai_index)
    factor_returns = pd.DataFrame({"factor": [0.01, 0.02]}, index=utc_index)
    loadings_index = pd.MultiIndex.from_product([shanghai_index, ["A"]], names=["dt", "ticker"])
    factor_loadings = pd.DataFrame({"factor": [2.0, 3.0]}, index=loadings_index)

    risk_exposures, attribution = perf_attrib(
        returns,
        positions,
        factor_returns,
        factor_loadings,
        pos_in_dollars=False,
        alignment="strict",
        normalize_tz="UTC",
    )

    pd.testing.assert_index_equal(risk_exposures.index, utc_index)
    pd.testing.assert_index_equal(attribution.index, utc_index)
    _assert_daily_attribution_identity(attribution)


def test_outer_dropna_uses_real_factor_return_completeness() -> None:
    inputs = _inputs(
        return_dates=["2024-01-01", "2024-01-02"],
        position_dates=["2024-01-01", "2024-01-02"],
    )
    factor_returns = inputs.factor_returns.copy()
    factor_returns.loc[pd.Timestamp("2024-01-02"), "factor"] = np.nan

    risk_exposures, attribution = perf_attrib(
        inputs.returns,
        inputs.positions,
        factor_returns,
        inputs.factor_loadings,
        pos_in_dollars=False,
        alignment="outer_dropna",
    )

    expected_index = pd.to_datetime(["2024-01-01"])
    pd.testing.assert_index_equal(risk_exposures.index, expected_index)
    pd.testing.assert_index_equal(attribution.index, expected_index)


def test_legacy_attribution_keeps_pinned_nan_and_outer_sum_semantics() -> None:
    inputs = _inputs(
        return_dates=["2024-01-01", "2024-01-02"],
        position_dates=["2024-01-01", "2024-01-02"],
    )
    factor_returns = inputs.factor_returns.copy()
    factor_returns.loc[pd.Timestamp("2024-01-02"), "factor"] = np.nan

    positions = inputs.positions.stack()
    positions.index = positions.index.set_names(["dt", "ticker"])

    risk_exposures, attribution = ep.perf_attrib(
        inputs.returns,
        positions,
        factor_returns,
        inputs.factor_loadings,
    )

    expected_index = pd.to_datetime(["2024-01-01", "2024-01-02"])
    pd.testing.assert_index_equal(risk_exposures.index, pd.DatetimeIndex(expected_index, name="dt"))
    pd.testing.assert_index_equal(attribution.index, expected_index)
    assert np.isnan(attribution.loc[pd.Timestamp("2024-01-02"), "factor"])
    assert attribution.loc[pd.Timestamp("2024-01-02"), "common_returns"] == 0.0
    assert attribution.loc[pd.Timestamp("2024-01-02"), "specific_returns"] == pytest.approx(0.2)


def test_legacy_attribution_keeps_pinned_outer_factor_columns() -> None:
    inputs = _inputs(
        return_dates=["2024-01-01", "2024-01-02"],
        position_dates=["2024-01-01", "2024-01-02"],
    )
    positions = inputs.positions.stack()
    positions.index = positions.index.set_names(["dt", "ticker"])
    factor_returns = inputs.factor_returns.rename(columns={"factor": "other"})

    risk_exposures, attribution = ep.perf_attrib(
        inputs.returns,
        positions,
        factor_returns,
        inputs.factor_loadings,
    )

    assert list(risk_exposures.columns) == ["factor"]
    assert {"factor", "other"}.issubset(attribution.columns)
    assert attribution[["factor", "other"]].isna().all().all()
    np.testing.assert_allclose(attribution["common_returns"], [0.0, 0.0])


def test_legacy_attribution_rejects_unstacked_positions_like_pinned_oracle() -> None:
    inputs = _inputs(
        return_dates=["2024-01-01", "2024-01-02"],
        position_dates=["2024-01-01", "2024-01-02"],
    )

    with pytest.raises(ValueError, match="Length of new names"):
        ep.perf_attrib(
            inputs.returns,
            inputs.positions,
            inputs.factor_returns,
            inputs.factor_loadings,
        )


@pytest.mark.parametrize("factor_return_columns", [["other"], ["factor", "other"]])
def test_strict_attribution_rejects_nonidentical_factor_columns(
    factor_return_columns: list[str],
) -> None:
    inputs = _inputs(
        return_dates=["2024-01-01", "2024-01-02"],
        position_dates=["2024-01-01", "2024-01-02"],
    )
    factor_returns = pd.DataFrame(
        np.full((2, len(factor_return_columns)), 0.01),
        index=inputs.factor_returns.index,
        columns=factor_return_columns,
    )

    with pytest.raises(DataAlignmentError, match="factor columns"):
        perf_attrib(
            inputs.returns,
            inputs.positions,
            factor_returns,
            inputs.factor_loadings,
            pos_in_dollars=False,
            alignment="strict",
        )


def test_inner_attribution_uses_factor_column_intersection() -> None:
    inputs = _inputs(
        return_dates=["2024-01-01", "2024-01-02"],
        position_dates=["2024-01-01", "2024-01-02"],
    )
    factor_returns = inputs.factor_returns.assign(other=[0.5, 0.6])
    factor_loadings = inputs.factor_loadings.assign(unused=[7.0, 8.0])

    risk_exposures, attribution = perf_attrib(
        inputs.returns,
        inputs.positions,
        factor_returns,
        factor_loadings,
        pos_in_dollars=False,
        alignment="inner",
    )

    assert list(risk_exposures.columns) == ["factor"]
    assert "factor" in attribution.columns
    assert "other" not in attribution.columns
    assert "unused" not in attribution.columns


def test_inner_attribution_rejects_disjoint_factor_columns() -> None:
    inputs = _inputs(
        return_dates=["2024-01-01", "2024-01-02"],
        position_dates=["2024-01-01", "2024-01-02"],
    )
    factor_returns = inputs.factor_returns.rename(columns={"factor": "other"})

    with pytest.raises(DataAlignmentError, match="factor columns"):
        perf_attrib(
            inputs.returns,
            inputs.positions,
            factor_returns,
            inputs.factor_loadings,
            pos_in_dollars=False,
            alignment="inner",
        )


def test_inner_attribution_drops_dates_without_any_usable_ticker() -> None:
    dates = pd.to_datetime(["2024-01-01", "2024-01-02"])
    returns = pd.Series([0.1, 0.2], index=dates)
    positions = pd.Series(
        [1.0, 1.0],
        index=pd.MultiIndex.from_tuples(
            [(dates[0], "A"), (dates[1], "B")],
            names=["dt", "ticker"],
        ),
    )
    factor_returns = pd.DataFrame({"factor": [0.01, 0.02]}, index=dates)
    factor_loadings = pd.DataFrame(
        {"factor": [2.0, 3.0, 4.0]},
        index=pd.MultiIndex.from_tuples(
            [(dates[0], "A"), (dates[0], "B"), (dates[1], "A")],
            names=["dt", "ticker"],
        ),
    )

    risk_exposures, attribution = perf_attrib(
        returns,
        positions,
        factor_returns,
        factor_loadings,
        pos_in_dollars=False,
        alignment="inner",
    )

    pd.testing.assert_index_equal(risk_exposures.index, dates[:1])
    pd.testing.assert_index_equal(attribution.index, dates[:1])
    np.testing.assert_allclose(risk_exposures["factor"], [2.0])


def test_attribution_validates_timezone_option_for_nondatetime_indices() -> None:
    returns = pd.Series([0.1, 0.2])
    positions = pd.DataFrame({"A": [1.0, 1.0]})
    factor_returns = pd.DataFrame({"factor": [0.01, 0.02]})
    factor_loadings = pd.DataFrame(
        {"factor": [2.0, 3.0]},
        index=pd.MultiIndex.from_product([[0, 1], ["A"]], names=["dt", "ticker"]),
    )

    with pytest.raises(ValueError, match="only 'UTC'"):
        perf_attrib(
            returns,
            positions,
            factor_returns,
            factor_loadings,
            pos_in_dollars=False,
            alignment="strict",
            normalize_tz="Asia/Shanghai",
        )
