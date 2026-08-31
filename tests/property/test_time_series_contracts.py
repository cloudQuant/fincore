"""Hypothesis property tests for time-series return contracts."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from fincore.metrics.ratios import sharpe_ratio
from fincore.metrics.returns import cum_returns, cum_returns_final


# Independent NumPy oracle for cumulative returns (never calls the function
# under test): cumulative growth is the product of (1 + r).
def _numpy_cumulative(returns: np.ndarray) -> np.ndarray:
    return np.cumprod(1.0 + returns)


@given(st.lists(st.floats(min_value=-0.2, max_value=0.2, allow_nan=False), min_size=3, max_size=40))
@settings(max_examples=200, deadline=None)
def test_cumulative_return_is_unchanged_by_series_copy(values: list[float]) -> None:
    original = pd.Series(values)
    copied = original.copy(deep=True)

    result_original = cum_returns(original)
    result_copied = cum_returns(copied)

    pd.testing.assert_series_equal(result_original, result_copied)


@given(st.lists(st.floats(min_value=-0.2, max_value=0.2, allow_nan=False), min_size=3, max_size=40))
@settings(max_examples=200, deadline=None)
def test_cum_returns_matches_numpy_oracle(values: list[float]) -> None:
    returns = pd.Series(values, index=pd.RangeIndex(len(values)))
    expected = _numpy_cumulative(np.asarray(values)) - 1.0

    actual = cum_returns(returns).to_numpy()

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


@given(st.lists(st.floats(min_value=-0.2, max_value=0.2, allow_nan=False), min_size=3, max_size=40))
@settings(max_examples=200, deadline=None)
def test_cum_returns_final_matches_compounded_product(values: list[float]) -> None:
    returns = pd.Series(values, index=pd.RangeIndex(len(values)))
    expected = float(np.prod(1.0 + np.asarray(values)) - 1.0)

    actual = cum_returns_final(returns)

    assert np.isclose(actual, expected, rtol=1e-12, atol=1e-12)


@given(
    st.lists(
        st.floats(min_value=-0.2, max_value=0.2, allow_nan=False).filter(lambda x: abs(x) >= 1e-6),
        min_size=3,
        max_size=40,
    )
)
@settings(max_examples=200, deadline=None)
def test_sharpe_ratio_is_scale_invariant(values: list[float]) -> None:
    returns = pd.Series(values, index=pd.RangeIndex(len(values)))
    scaled = returns * 2.0

    a = sharpe_ratio(returns)
    b = sharpe_ratio(scaled)
    assert (np.isnan(a) and np.isnan(b)) or np.isclose(a, b, rtol=1e-9, atol=1e-9)


@given(st.lists(st.floats(min_value=-0.5, max_value=0.5, allow_nan=False), min_size=1, max_size=30))
@settings(max_examples=100, deadline=None)
def test_cum_returns_handles_bounded_return_values(values: list[float]) -> None:
    returns = pd.Series(values, index=pd.RangeIndex(len(values)))

    result = cum_returns(returns)

    assert isinstance(result, pd.Series)
    assert len(result) == len(values)
