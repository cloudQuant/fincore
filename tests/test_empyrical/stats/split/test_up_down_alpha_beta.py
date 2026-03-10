"""Tests for up/down alpha-beta calculations.

Part of test_tracking_risk.py split - Up/Down alpha-beta tests with P1 markers.
"""

from __future__ import annotations

from unittest import TestCase

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_almost_equal
from parameterized import parameterized

from fincore.empyrical import Empyrical

DECIMAL_PLACES = 8

rand = np.random.RandomState(1337)


@pytest.mark.p1
class TestUpDownAlphaBeta(TestCase):
    """Tests for up/down alpha-beta calculations."""

    simple_benchmark = pd.Series(
        np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]) / 100,
        index=pd.date_range("2000-1-30", periods=9, freq="D"),
    )

    positive_returns = pd.Series(
        np.array([1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]) / 100,
        index=pd.date_range("2000-1-30", periods=9, freq="D"),
    )

    negative_returns = pd.Series(
        np.array([0.0, -6.0, -7.0, -1.0, -9.0, -2.0, -6.0, -8.0, -5.0]) / 100,
        index=pd.date_range("2000-1-30", periods=9, freq="D"),
    )

    mixed_returns = pd.Series(
        np.array([np.nan, 1.0, 10.0, -4.0, 2.0, 3.0, 2.0, 1.0, -10.0]) / 100,
        index=pd.date_range("2000-1-30", periods=9, freq="D"),
    )

    empty_returns = pd.Series(np.array([]) / 100, index=pd.date_range("2000-1-30", periods=0, freq="D"))

    one_return = pd.Series(np.array([1.0]) / 100, index=pd.date_range("2000-1-30", periods=1, freq="D"))

    @parameterized.expand(
        [
            (empty_returns, empty_returns, (np.nan, np.nan)),
            (one_return, one_return, (np.nan, np.nan)),
            (mixed_returns[1:], negative_returns[1:], (-0.9997853834885004, -0.71296296296296313)),
            (mixed_returns, mixed_returns, (0.0, 1.0)),
            (mixed_returns, -mixed_returns, (0.0, -1.0)),
        ]
    )
    def test_down_alpha_beta(self, returns, benchmark, expected):
        """Test down alpha and beta calculation."""
        down_alpha, down_beta = Empyrical(
            return_types=np.ndarray,
        ).down_alpha_beta(returns, benchmark)
        assert_almost_equal(down_alpha, expected[0], DECIMAL_PLACES)
        assert_almost_equal(down_beta, expected[1], DECIMAL_PLACES)

    @parameterized.expand(
        [
            (empty_returns, empty_returns, (np.nan, np.nan)),
            (one_return, one_return, (np.nan, np.nan)),
            (mixed_returns[1:], positive_returns[1:], (0.432961242076658, 0.4285714285)),
            (mixed_returns, mixed_returns, (0.0, 1.0)),
            (mixed_returns, -mixed_returns, (0.0, -1.0)),
        ]
    )
    def test_up_alpha_beta(self, returns, benchmark, expected):
        """Test up alpha and beta calculation."""
        up_alpha, up_beta = Empyrical(
            return_types=np.ndarray,
        ).up_alpha_beta(returns, benchmark)
        assert_almost_equal(up_alpha, expected[0], DECIMAL_PLACES)
        assert_almost_equal(up_beta, expected[1], DECIMAL_PLACES)
