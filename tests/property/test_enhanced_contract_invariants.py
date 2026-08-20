"""Property tests for enhanced contract invariants."""

from __future__ import annotations

import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from fincore.contracts.analysis import AnalysisInput


@given(st.lists(st.floats(min_value=-0.5, max_value=0.5, allow_nan=False), min_size=1, max_size=100))
@settings(max_examples=50, deadline=None)
def test_analysis_input_preserves_values_and_copies(values: list[float]) -> None:
    returns = pd.Series(values)
    ai = AnalysisInput.from_returns(returns)

    assert list(ai.returns) == list(returns)
    assert ai.returns is not returns


@given(st.lists(st.floats(min_value=-0.5, max_value=0.5, allow_nan=False), min_size=2, max_size=50))
@settings(max_examples=50, deadline=None)
def test_analysis_input_does_not_mutate_input(values: list[float]) -> None:
    returns = pd.Series(values)
    original = returns.copy(deep=True)

    AnalysisInput.from_returns(returns)

    pd.testing.assert_series_equal(returns, original)


@given(st.lists(st.floats(min_value=-0.5, max_value=0.5, allow_nan=False), min_size=1, max_size=50))
@settings(max_examples=30, deadline=None)
def test_config_digest_is_stable(values: list[float]) -> None:
    a = AnalysisInput.from_returns(pd.Series(values))
    b = AnalysisInput.from_returns(pd.Series(values))

    assert a.config_digest == b.config_digest


@given(st.lists(st.floats(min_value=-0.5, max_value=0.5, allow_nan=False), min_size=1, max_size=50))
@settings(max_examples=30, deadline=None)
def test_config_digest_changes_with_data(values: list[float]) -> None:
    a = AnalysisInput.from_returns(pd.Series(values))
    changed = [v + 0.01 for v in values]
    b = AnalysisInput.from_returns(pd.Series(changed))

    assert a.config_digest != b.config_digest
