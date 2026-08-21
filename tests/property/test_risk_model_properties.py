"""Property tests for risk model invariants."""

from __future__ import annotations

import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from fincore.risk.calibration import basel_traffic_light, expected_exception_count
from fincore.risk.diagnostics import walk_forward_var
from fincore.risk.specs import RiskModelSpec


@given(st.integers(min_value=0, max_value=100), st.integers(min_value=1, max_value=500))
@settings(max_examples=100, deadline=None)
def test_traffic_light_is_monotone(exceptions: int, observations: int) -> None:
    # More exceptions can never move to a milder zone.
    zones = {"green": 0, "yellow": 1, "red": 2}
    current = basel_traffic_light(exceptions, observations, 0.99)
    for extra in range(1, 5):
        more = basel_traffic_light(exceptions + extra, observations, 0.99)
        assert zones[more] >= zones[current]


@given(st.integers(min_value=1, max_value=1000))
@settings(max_examples=100, deadline=None)
def test_expected_exceptions_linear(observations: int) -> None:
    assert np.isclose(expected_exception_count(observations, 0.99), observations * 0.01)


@given(
    st.lists(st.floats(min_value=-0.2, max_value=0.2, allow_nan=False, allow_infinity=False), min_size=48, max_size=96)
)
@settings(max_examples=50, deadline=None)
def test_walk_forward_forecast_prefix_is_invariant_to_future_returns(values: list[float]) -> None:
    index = pd.date_range("2023-01-01", periods=len(values), freq="B", tz="UTC")
    returns = pd.Series(values, index=index)
    spec = RiskModelSpec(distribution="historical", confidence_level=0.95, window=24, refit_cadence=1)
    baseline = walk_forward_var(returns, spec)

    cutoff_position = len(values) - 12
    cutoff = index[cutoff_position]
    changed = returns.copy()
    changed.iloc[cutoff_position:] = changed.iloc[cutoff_position:] + 0.25
    perturbed = walk_forward_var(changed, spec)

    pd.testing.assert_series_equal(
        baseline.forecast.loc[:cutoff],
        perturbed.forecast.loc[:cutoff],
        check_names=False,
    )
