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


def test_legacy_partial_date_intersection_never_relabels_exposures_by_length() -> None:
    inputs = _inputs(
        return_dates=["2024-01-02", "2024-01-03"],
        position_dates=["2024-01-01", "2024-01-02"],
    )

    risk_exposures, attribution = ep.perf_attrib(
        inputs.returns,
        inputs.positions,
        inputs.factor_returns,
        inputs.factor_loadings,
    )

    expected_index = pd.to_datetime(["2024-01-02"])
    pd.testing.assert_index_equal(risk_exposures.index, expected_index)
    pd.testing.assert_index_equal(attribution.index, expected_index)
    _assert_daily_attribution_identity(attribution)


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
