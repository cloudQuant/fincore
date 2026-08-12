from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import fincore.empyrical as ep
from fincore.metrics.risk import conditional_value_at_risk as enhanced_conditional_value_at_risk


def test_legacy_cvar_uses_fixed_tail_count_for_quantile_ties() -> None:
    values = np.array([-0.2, -0.1, -0.1, -0.1, 1.0])

    assert ep.conditional_value_at_risk(values, cutoff=0.25) == pytest.approx(-0.15)


def test_enhanced_cvar_keeps_threshold_inclusive_expected_shortfall() -> None:
    values = np.array([-0.2, -0.1, -0.1, -0.1, 1.0])

    assert enhanced_conditional_value_at_risk(values, cutoff=0.25) == pytest.approx(-0.125)


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        (np.array([], dtype=np.float64), np.nan),
        (np.array([0.1], dtype=np.float64), 0.1),
        (np.array([np.nan, np.nan]), np.nan),
        (np.array([-0.2, np.nan, 0.1]), -0.2),
        (np.array([-0.2, np.inf, 0.1]), -0.2),
        (np.array([-0.2, -np.inf, 0.1]), -np.inf),
        (np.array([0.1, 0.1, 0.1]), 0.1),
        (np.array([-1.2, -1.0, -0.5]), -1.2),
    ],
    ids=[
        "empty",
        "single",
        "all-nan",
        "partial-nan",
        "positive-inf",
        "negative-inf",
        "constant",
        "loss-at-or-below-minus-one",
    ],
)
def test_legacy_cvar_matches_pinned_non_finite_and_boundary_semantics(
    values: np.ndarray,
    expected: float,
) -> None:
    result = ep.conditional_value_at_risk(values)

    if np.isnan(expected):
        assert np.isnan(result)
    else:
        assert result == pytest.approx(expected)


def test_legacy_empty_var_keeps_pinned_exception_contract() -> None:
    with pytest.raises(IndexError, match="out of bounds"):
        ep.value_at_risk(np.array([], dtype=np.float64))


@pytest.mark.parametrize(
    ("dtype", "expected_type"),
    [
        (np.float32, np.float32),
        (np.float64, np.float64),
        (np.int64, np.float64),
    ],
)
def test_legacy_cvar_preserves_pinned_dtype_projection(
    dtype: type[np.generic],
    expected_type: type[np.generic],
) -> None:
    values = np.array([-2, 1, 3, -1], dtype=dtype)

    result = ep.conditional_value_at_risk(values, cutoff=0.25)

    assert isinstance(result, expected_type)
    assert result == pytest.approx(-2.0)


@pytest.mark.parametrize(
    ("period", "expected"),
    [
        (ep.DAILY, 0.3519943181359608),
        (ep.WEEKLY, 0.15989579940281942),
        (ep.MONTHLY, 0.07681145747868608),
        (ep.QUARTERLY, 0.0443471156521669),
        (ep.YEARLY, 0.02217355782608345),
    ],
)
def test_legacy_annual_volatility_honors_all_period_constants(period: str, expected: float) -> None:
    returns = np.array([-0.02, 0.01, 0.03, -0.01])

    assert ep.annual_volatility(returns, period=period) == pytest.approx(expected)


def test_legacy_annual_volatility_custom_annualization_overrides_period() -> None:
    returns = np.array([-0.02, 0.01, 0.03, -0.01])

    result = ep.annual_volatility(returns, period="not-a-period", annualization=7)

    assert result == pytest.approx(0.05866571968932681)


def test_legacy_dataframe_result_and_out_buffer_match_pinned_projection() -> None:
    values = np.array([-0.02, 0.01, 0.03, -0.01])
    returns = pd.DataFrame({"first": values, "second": values * 2})
    out = np.full(2, 999.0)

    result = ep.annual_volatility(returns, period=ep.YEARLY, out=out)

    assert result is out
    np.testing.assert_allclose(result, [0.02217355782608345, 0.0443471156521669])


@pytest.mark.parametrize(
    "constructor",
    [
        pytest.param(lambda values: values.copy(), id="ndarray"),
        pytest.param(lambda values: pd.Series(values, index=pd.date_range("2024-01-01", periods=4)), id="series"),
        pytest.param(
            lambda values: pd.DataFrame({"first": values, "second": values * 2}),
            id="dataframe",
        ),
    ],
)
def test_legacy_cumulative_returns_do_not_mutate_inputs(constructor) -> None:
    original = constructor(np.array([-1.2, np.nan, 0.1, -0.5], dtype=np.float64))
    before = original.copy(deep=True) if isinstance(original, (pd.Series, pd.DataFrame)) else original.copy()

    ep.cum_returns(original)

    if isinstance(original, pd.Series):
        pd.testing.assert_series_equal(original, before)
    elif isinstance(original, pd.DataFrame):
        pd.testing.assert_frame_equal(original, before)
    else:
        np.testing.assert_array_equal(original, before)
