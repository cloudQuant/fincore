from __future__ import annotations

import pandas as pd
import pytest

import fincore.empyrical as strict_empyrical
from fincore.metrics.perf_attrib import compute_exposures, perf_attrib


def _attribution_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame]:
    dates = pd.date_range("2024-01-02", periods=2, freq="B", name="dt")
    wide = pd.DataFrame(
        {"AAA": [60.0, 30.0], "BBB": [-20.0, 10.0], "cash": [60.0, 60.0]},
        index=dates,
    )
    loadings_index = pd.MultiIndex.from_product([dates, ["AAA", "BBB"]], names=["dt", "ticker"])
    factor_loadings = pd.DataFrame(
        {"market": [1.0, 2.0, 1.0, 2.0]},
        index=loadings_index,
    )
    returns = pd.Series([0.01, 0.02], index=dates, name="returns")
    factor_returns = pd.DataFrame({"market": [0.005, 0.006]}, index=dates)
    return wide, factor_loadings, returns, factor_returns


def test_wide_and_stacked_perf_attrib_are_equivalent_under_pinned_net_asset_normalization() -> None:
    wide, factor_loadings, _returns, _factor_returns = _attribution_inputs()
    normalized = wide.divide(wide.sum(axis="columns"), axis="rows").drop(columns="cash")
    stacked = normalized.stack()
    stacked.index = stacked.index.set_names(["dt", "ticker"])

    wide_result = compute_exposures(
        wide,
        factor_loadings,
        stack_positions=True,
        pos_in_dollars=True,
    )
    stacked_result = compute_exposures(
        stacked,
        factor_loadings,
        stack_positions=False,
        pos_in_dollars=True,
    )

    expected = pd.DataFrame({"market": [0.2, 0.5]}, index=wide.index)
    pd.testing.assert_frame_equal(wide_result, stacked_result)
    pd.testing.assert_frame_equal(wide_result, expected, check_freq=False)


def test_percentage_positions_are_stacked_without_renormalizing() -> None:
    wide, factor_loadings, _returns, _factor_returns = _attribution_inputs()
    percentages = wide.divide(wide.sum(axis="columns"), axis="rows")
    actual = compute_exposures(
        percentages,
        factor_loadings,
        stack_positions=True,
        pos_in_dollars=False,
    )
    expected = pd.DataFrame({"market": [0.2, 0.5]}, index=wide.index)
    pd.testing.assert_frame_equal(actual, expected, check_freq=False)


def test_strict_empyrical_compute_exposures_keeps_stacked_input_contract() -> None:
    wide, factor_loadings, _returns, _factor_returns = _attribution_inputs()
    normalized = wide.divide(wide.sum(axis="columns"), axis="rows").drop(columns="cash")
    stacked = normalized.stack()
    stacked.index = stacked.index.set_names(["dt", "ticker"])

    expected = pd.DataFrame({"market": [0.2, 0.5]}, index=wide.index)
    actual = strict_empyrical.compute_exposures(stacked, factor_loadings)
    pd.testing.assert_frame_equal(actual, expected, check_freq=False)


def test_regression_style_is_not_silently_ignored() -> None:
    wide, factor_loadings, returns, factor_returns = _attribution_inputs()
    with pytest.raises(ValueError, match="regression_style"):
        perf_attrib(
            returns,
            wide,
            factor_returns,
            factor_loadings,
            regression_style="unsupported",
        )
