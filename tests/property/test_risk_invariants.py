"""Hypothesis property tests for risk-invariant contracts."""

from __future__ import annotations

import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from fincore.risk.backtesting import kupiec_lr
from fincore.risk.models import SIGN_LOSSES_NEGATIVE, forecast_es, forecast_var


@given(
    st.integers(min_value=1, max_value=2000),
    st.integers(min_value=0, max_value=2000),
    st.floats(min_value=0.5, max_value=0.9999, allow_nan=False),
)
@settings(max_examples=200, deadline=None)
def test_kupiec_lr_is_nonnegative(observations: int, exceptions: int, confidence_level: float) -> None:
    exceptions = min(exceptions, observations)
    lr = kupiec_lr(observations, exceptions, confidence_level)
    assert lr >= 0.0
    assert np.isfinite(lr)


@given(st.lists(st.floats(min_value=-0.2, max_value=0.2, allow_nan=False), min_size=10, max_size=200))
@settings(max_examples=100, deadline=None)
def test_forecast_var_matches_empirical_quantile(values: list[float]) -> None:
    returns = pd.Series(values, index=pd.RangeIndex(len(values)))

    estimate = forecast_var(returns, method="historical", confidence_level=0.99)

    assert estimate.sign_convention == SIGN_LOSSES_NEGATIVE
    expected = float(np.quantile(np.asarray(values), 0.01))
    assert np.isclose(estimate.estimate, expected, rtol=1e-12, atol=1e-12)


@given(st.lists(st.floats(min_value=-0.2, max_value=0.2, allow_nan=False), min_size=10, max_size=200))
@settings(max_examples=100, deadline=None)
def test_es_is_at_least_as_extreme_as_var(values: list[float]) -> None:
    returns = pd.Series(values, index=pd.RangeIndex(len(values)))

    var = forecast_var(returns, method="historical", confidence_level=0.99)
    es = forecast_es(returns, method="historical", confidence_level=0.99)

    assert es.estimate <= var.estimate + 1e-12


@given(st.lists(st.floats(min_value=-0.2, max_value=0.2, allow_nan=False), min_size=20, max_size=200))
@settings(max_examples=100, deadline=None)
def test_higher_confidence_yields_more_extreme_var(values: list[float]) -> None:
    returns = pd.Series(values, index=pd.RangeIndex(len(values)))

    var_95 = forecast_var(returns, method="historical", confidence_level=0.95)
    var_99 = forecast_var(returns, method="historical", confidence_level=0.99)

    assert var_99.estimate <= var_95.estimate + 1e-12


@given(st.lists(st.floats(min_value=-0.2, max_value=0.2, allow_nan=False), min_size=10, max_size=100))
@settings(max_examples=50, deadline=None)
def test_forecast_var_does_not_mutate_input(values: list[float]) -> None:
    returns = pd.Series(values, index=pd.RangeIndex(len(values)))
    original = returns.copy(deep=True)

    forecast_var(returns, method="historical", confidence_level=0.95)

    pd.testing.assert_series_equal(returns, original)
