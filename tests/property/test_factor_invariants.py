"""Hypothesis property tests for factor-analysis invariants."""

from __future__ import annotations

import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from fincore.factor_analysis.data import quantize_factor


def _factor_frame(values: list[float], n_dates: int = 4) -> pd.DataFrame:
    dates = pd.to_datetime([f"2024-01-0{d + 1}" for d in range(n_dates)])
    assets = [f"A{i}" for i in range(len(values) // n_dates)]
    index = pd.MultiIndex.from_product([dates, assets], names=["date", "asset"])
    return pd.DataFrame({"factor": values[: len(index)]}, index=index)


@given(st.lists(st.floats(min_value=-5.0, max_value=5.0, allow_nan=False), min_size=8, max_size=80))
@settings(max_examples=50, deadline=None)
def test_quantize_factor_labels_stay_within_bucket_range(values: list[float]) -> None:
    frame = _factor_frame(values)
    quantiles = 5

    result = quantize_factor(frame, quantiles=quantiles, no_raise=True)

    finite = result.dropna()
    assert ((finite >= 1) & (finite <= quantiles)).all()


@given(st.lists(st.floats(min_value=-5.0, max_value=5.0, allow_nan=False), min_size=8, max_size=80))
@settings(max_examples=50, deadline=None)
def test_quantize_factor_does_not_mutate_input(values: list[float]) -> None:
    frame = _factor_frame(values)
    original = frame.copy(deep=True)

    quantize_factor(frame, quantiles=5, no_raise=True)

    pd.testing.assert_frame_equal(frame, original)


@given(st.lists(st.floats(min_value=-5.0, max_value=5.0, allow_nan=False), min_size=8, max_size=80))
@settings(max_examples=50, deadline=None)
def test_quantize_factor_preserves_index(values: list[float]) -> None:
    frame = _factor_frame(values)

    result = quantize_factor(frame, quantiles=5, no_raise=True)

    assert len(result) <= len(frame)
    assert result.index.isin(frame.index).all()
