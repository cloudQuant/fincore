"""Branch-completion tests for metrics.risk, metrics.basic and metrics.drawdown."""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import pytest

from fincore.metrics.basic import aligned_series
from fincore.metrics.drawdown import (
    _identify_drawdown_periods,
    max_drawdown,
    max_drawdown_months,
    max_drawdown_recovery_days,
    max_drawdown_recovery_months,
    max_drawdown_recovery_weeks,
)
from fincore.metrics.risk import (
    _gpd_es_calculator,
    _gpd_loglikelihood_scale_and_shape,
    _gpd_loglikelihood_scale_only,
    _gpd_var_calculator,
    annual_volatility,
    conditional_value_at_risk,
    downside_risk,
    gpd_risk_estimates,
    tracking_error,
    value_at_risk,
)


def _daily_returns(n: int = 100, seed: int = 1) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0.0, 0.02, n), index=pd.date_range("2024-01-01", periods=n))


# ---------------------------------------------------------------------------
# risk.annual_volatility — deprecated alpha_ branch
# ---------------------------------------------------------------------------


def test_annual_volatility_alpha_deprecation_warns() -> None:
    returns = _daily_returns(50)
    with pytest.warns(DeprecationWarning, match="alpha_"):
        result = annual_volatility(returns, alpha_=2.0)
    assert result > 0.0


# ---------------------------------------------------------------------------
# risk.downside_risk — 2D empty and non-finite required_return
# ---------------------------------------------------------------------------


def test_downside_risk_empty_2d() -> None:
    df = pd.DataFrame(columns=["a", "b"])
    result = downside_risk(df)
    assert isinstance(result, np.ndarray)
    assert result.shape == (2,)
    assert np.isnan(result).all()


def test_downside_risk_nonfinite_required_return_scalar() -> None:
    returns = _daily_returns(50)
    result = downside_risk(returns, required_return=np.nan)
    assert np.isnan(result)


def test_downside_risk_nonfinite_required_return_dataframe() -> None:
    df = pd.DataFrame({"a": _daily_returns(50).values, "b": _daily_returns(50, seed=2).values})
    result = downside_risk(df, required_return=np.inf)
    assert isinstance(result, pd.Series)
    assert np.isnan(result).all()


# ---------------------------------------------------------------------------
# risk.value_at_risk / conditional_value_at_risk — invalid cutoff, non-finite
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cutoff", [-0.1, 1.1])
def test_value_at_risk_invalid_cutoff(cutoff: float) -> None:
    assert np.isnan(value_at_risk(_daily_returns(50), cutoff=cutoff))


@pytest.mark.parametrize("cutoff", [-0.1, 1.1])
def test_conditional_value_at_risk_invalid_cutoff(cutoff: float) -> None:
    assert np.isnan(conditional_value_at_risk(_daily_returns(50), cutoff=cutoff))


def test_value_at_risk_nonfinite_returns() -> None:
    returns = pd.Series([0.01, np.nan, 0.02])
    assert np.isnan(value_at_risk(returns))


def test_conditional_value_at_risk_nonfinite_returns() -> None:
    returns = pd.Series([0.01, np.inf, 0.02])
    assert np.isnan(conditional_value_at_risk(returns))


# ---------------------------------------------------------------------------
# risk.tracking_error — 2D empty branch
# ---------------------------------------------------------------------------


def test_tracking_error_empty_2d() -> None:
    df = pd.DataFrame(columns=["a", "b"])
    bench = pd.DataFrame(columns=["a", "b"])
    result = tracking_error(df, bench)
    assert isinstance(result, np.ndarray)
    assert np.isnan(result).all()


# ---------------------------------------------------------------------------
# risk._gpd_* private kernels — branch edges
# ---------------------------------------------------------------------------


def test_gpd_loglikelihood_scale_and_shape_zero_scale() -> None:
    out = _gpd_loglikelihood_scale_and_shape(0.0, 1.0, np.array([1.0, 2.0]))
    assert out == -sys.float_info.max


def test_gpd_loglikelihood_scale_only_negative_scale() -> None:
    out = _gpd_loglikelihood_scale_only(-1.0, np.array([1.0, 2.0]))
    assert out == -sys.float_info.max


def test_gpd_loglikelihood_scale_only_valid() -> None:
    out = _gpd_loglikelihood_scale_only(1.0, np.array([1.0, 2.0]))
    assert np.isfinite(out)


def test_gpd_var_calculator_no_exceedance() -> None:
    assert _gpd_var_calculator(0.2, 1.0, 0.1, 0.01, 100, 0) == 0.0


def test_gpd_var_calculator_valid() -> None:
    out = _gpd_var_calculator(0.2, 1.0, 0.1, 0.01, 100, 10)
    assert out > 0.0


def test_gpd_es_calculator_shape_one() -> None:
    assert _gpd_es_calculator(0.5, 0.2, 1.0, 1.0) == 0.0


def test_gpd_es_calculator_valid() -> None:
    out = _gpd_es_calculator(0.5, 0.2, 1.0, 0.1)
    assert out > 0.0


# ---------------------------------------------------------------------------
# risk.gpd_risk_estimates — short input and Series wrapping
# ---------------------------------------------------------------------------


def test_gpd_risk_estimates_short_series() -> None:
    result = gpd_risk_estimates(pd.Series([0.01, -0.01]))
    assert isinstance(result, pd.Series)
    assert len(result) == 5


def test_gpd_risk_estimates_short_ndarray() -> None:
    result = gpd_risk_estimates(np.array([0.01, -0.01]))
    assert isinstance(result, np.ndarray)
    assert len(result) == 5


def test_gpd_risk_estimates_valid_series() -> None:
    result = gpd_risk_estimates(_daily_returns(200, seed=5))
    assert isinstance(result, pd.Series)
    assert len(result) == 5


# ---------------------------------------------------------------------------
# basic.aligned_series — type-combination branches
# ---------------------------------------------------------------------------


def test_aligned_series_ndarrays_returned_unchanged() -> None:
    a = np.array([1.0, 2.0])
    b = np.array([3.0, 4.0])
    result = aligned_series(a, b)
    assert result[0] is a and result[1] is b


def test_aligned_series_matching_dataframes() -> None:
    idx = pd.date_range("2024-01-01", periods=2)
    a = pd.DataFrame({"x": [1.0, 2.0]}, index=idx)
    b = pd.DataFrame({"y": [3.0, 4.0]}, index=idx)
    result = aligned_series(a, b)
    assert result[0] is a and result[1] is b


def test_aligned_series_series_and_matching_dataframe() -> None:
    idx = pd.date_range("2024-01-01", periods=2)
    a = pd.Series([1.0, 2.0], index=idx)
    b = pd.DataFrame({"y": [3.0, 4.0]}, index=idx)
    result = aligned_series(a, b)
    assert result[0] is a and result[1] is b


def test_aligned_series_three_series_fallback() -> None:
    idx = pd.date_range("2024-01-01", periods=3)
    a = pd.Series([1.0, 2.0, 3.0], index=idx)
    b = pd.Series([4.0, 5.0, 6.0], index=idx)
    c = pd.Series([7.0, 8.0, 9.0], index=idx)
    result = aligned_series(a, b, c)
    assert len(result) == 3


# ---------------------------------------------------------------------------
# drawdown — 2D and ndarray branches
# ---------------------------------------------------------------------------


def test_max_drawdown_empty_2d() -> None:
    df = pd.DataFrame(columns=["a", "b"])
    result = max_drawdown(df)
    assert isinstance(result, np.ndarray)
    assert np.isnan(result).all()


def test_max_drawdown_2d_ndarray() -> None:
    arr = np.random.default_rng(1).normal(0.0, 0.02, (50, 3))
    result = max_drawdown(arr)
    assert isinstance(result, np.ndarray)
    assert result.shape == (3,)


def test_identify_drawdown_periods_from_ndarray() -> None:
    arr = np.array([0.01, -0.02, 0.03, -0.01, 0.005])
    result = _identify_drawdown_periods(arr)
    assert result is not None


def test_max_drawdown_months_finite() -> None:
    assert max_drawdown_months(_daily_returns(60)) >= 0


def test_max_drawdown_recovery_days_ndarray() -> None:
    arr = np.array([0.01, -0.02, 0.03])
    assert np.isnan(max_drawdown_recovery_days(arr))


def test_max_drawdown_recovery_weeks_finite() -> None:
    returns = pd.Series([0.01, -0.01, 0.02, 0.01], index=pd.date_range("2024-01-01", periods=4))
    assert max_drawdown_recovery_weeks(returns) >= 0


def test_max_drawdown_recovery_months_finite() -> None:
    returns = pd.Series([0.01, -0.01, 0.02, 0.01], index=pd.date_range("2024-01-01", periods=4))
    assert max_drawdown_recovery_months(returns) >= 0
