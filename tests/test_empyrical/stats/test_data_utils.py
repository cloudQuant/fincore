"""Tests for data utility functions (roll, up, down).

This module tests the roll, up, and down helper functions.
Split from test_helpers.py for maintainability.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_almost_equal
from unittest import TestCase

from fincore.empyrical import Empyrical
from fincore.utils.data_utils import down, roll, up

rand = np.random.RandomState(1337)


class TestDataUtils(TestCase):
    """Tests for data utility functions (roll, up, down)."""

    def setUp(self):
        self.ser_length = 120
        self.window = 12

        # Pandas frequency alias compatibility
        try:
            pd.date_range("2000-1-1", periods=1, freq="ME")
            self.month_freq = "ME"
        except ValueError:
            self.month_freq = "M"

        self.returns = pd.Series(
            rand.randn(1, 120)[0] / 100.0,
            index=pd.date_range("2000-1-30", periods=120, freq=self.month_freq)
        )

        self.factor_returns = pd.Series(
            rand.randn(1, 120)[0] / 100.0,
            index=pd.date_range("2000-1-30", periods=120, freq=self.month_freq)
        )

    @pytest.mark.p1  # High: frequently used data utilities
    def test_roll_pandas(self):
        """Test roll function with pandas Series."""
        emp = Empyrical()
        res = roll(self.returns, self.factor_returns, window=12, function=emp.alpha_aligned)

        self.assertEqual(res.size, self.ser_length - self.window + 1)

    @pytest.mark.p1  # High: frequently used data utilities
    def test_roll_ndarray(self):
        """Test roll function with numpy arrays."""
        emp = Empyrical()
        res = roll(
            self.returns.values,
            self.factor_returns.values,
            window=12,
            function=emp.alpha_aligned
        )

        self.assertEqual(len(res), self.ser_length - self.window + 1)

    @pytest.mark.p1  # High: frequently used data utilities
    def test_down(self):
        """Test down function for down-market capture."""
        emp = Empyrical()
        pd_res = down(self.returns, self.factor_returns, function=emp.capture)
        np_res = down(self.returns.values, self.factor_returns.values, function=emp.capture)

        self.assertTrue(isinstance(pd_res, float))
        assert_almost_equal(pd_res, np_res, 8)

    @pytest.mark.p1  # High: frequently used data utilities
    def test_up(self):
        """Test up function for up-market capture."""
        emp = Empyrical()
        pd_res = up(self.returns, self.factor_returns, function=emp.capture)
        np_res = up(self.returns.values, self.factor_returns.values, function=emp.capture)

        self.assertTrue(isinstance(pd_res, float))
        assert_almost_equal(pd_res, np_res, 8)

    @pytest.mark.p2  # Medium: edge case - mixed types
    def test_roll_bad_types(self):
        """Test roll function raises ValueError for mixed types."""
        with self.assertRaises(ValueError):
            emp = Empyrical()
            roll(self.returns.values, self.factor_returns, window=12, function=emp.max_drawdown)

    @pytest.mark.p2  # Medium: edge case - window larger than data
    def test_roll_max_window(self):
        """Test roll function with window larger than series length."""
        emp = Empyrical()
        res = roll(
            self.returns,
            self.factor_returns,
            window=self.ser_length + 100,
            function=emp.max_drawdown
        )
        self.assertTrue(res.size == 0)
