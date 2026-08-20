"""Property tests for risk model invariants."""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from fincore.risk.calibration import basel_traffic_light, expected_exception_count


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
