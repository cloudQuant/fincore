"""Tests for Excess Sharpe ratio.

Split from test_other_ratios.py for maintainability.

Priority Markers:
- P1: Excess Sharpe tests (important relative performance metric)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_almost_equal
from parameterized import parameterized
from unittest import TestCase

from fincore import empyrical
from fincore.metrics import ratios as ratios_module

DECIMAL_PLACES = 8
rand = np.random.RandomState(1337)


class BaseTestCase(TestCase):
    """Base test case with index matching assertion."""

    def assert_indexes_match(self, result, expected):
        """Assert that two pandas objects have the same indices."""
        try:
            from pandas.testing import assert_index_equal
        except ImportError:
            from pandas.util.testing import assert_index_equal

        assert_index_equal(result.index, expected.index)

        if isinstance(result, pd.DataFrame) and isinstance(expected, pd.DataFrame):
            assert_index_equal(result.columns, expected.columns)


@pytest.mark.p1  # High: important relative performance metric
class TestExcessSharpe(BaseTestCase):
    """Tests for excess Sharpe ratio calculation."""

    # Test data
    empty_returns = pd.Series(
        np.array([]) / 100,
        index=pd.date_range("2000-1-30", periods=0, freq="D")
    )

    one_return = pd.Series(
        np.array([1.0]) / 100,
        index=pd.date_range("2000-1-30", periods=1, freq="D")
    )

    mixed_returns = pd.Series(
        np.array([np.nan, 1.0, 10.0, -4.0, 2.0, 3.0, 2.0, 1.0, -10.0]) / 100,
        index=pd.date_range("2000-1-30", periods=9, freq="D"),
    )

    flat_line_1 = pd.Series(
        np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]) / 100,
        index=pd.date_range("2000-1-30", periods=9, freq="D"),
    )

    flat_line_0 = pd.Series(
        np.linspace(0, 0, num=1000),
        index=pd.date_range("2000-1-30", periods=1000, freq="D", tz="UTC")
    )

    flat_line_1_tz = pd.Series(
        np.linspace(0.01, 0.01, num=1000),
        index=pd.date_range("2000-1-30", periods=1000, freq="D", tz="UTC")
    )

    pos_line = pd.Series(
        np.linspace(0, 1, num=1000),
        index=pd.date_range("2000-1-30", periods=1000, freq="D", tz="UTC")
    )

    neg_line = pd.Series(
        np.linspace(0, -1, num=1000),
        index=pd.date_range("2000-1-30", periods=1000, freq="D", tz="UTC")
    )

    noise = pd.Series(
        rand.normal(0, 0.001, 1000),
        index=pd.date_range("2000-1-30", periods=1000, freq="D", tz="UTC")
    )

    inv_noise = noise.multiply(-1)

    @property
    def empyrical(self):
        """Get empyrical module instance."""
        return empyrical

    @parameterized.expand([
        (empty_returns, 0.0, np.nan),
        (one_return, 0.0, np.nan),
        (pos_line, pos_line, np.nan),
        (mixed_returns, 0.0, 0.10859306069076737),
        (mixed_returns, flat_line_1, -0.06515583641446039),
    ])
    def test_excess_sharpe(self, returns, factor_returns, expected):
        """Test excess Sharpe ratio calculation."""
        assert_almost_equal(
            ratios_module.excess_sharpe(returns, factor_returns),
            expected,
            DECIMAL_PLACES,
        )

    @parameterized.expand([(flat_line_0, pos_line), (flat_line_1_tz, pos_line), (noise, pos_line)])
    def test_excess_sharpe_noisy(self, noise_line, benchmark):
        """Test excess Sharpe increases with uncorrelated returns."""
        noisy_returns_1 = noise_line[0:250].add(benchmark[250:], fill_value=0)
        noisy_returns_2 = noise_line[0:500].add(benchmark[500:], fill_value=0)
        noisy_returns_3 = noise_line[0:750].add(benchmark[750:], fill_value=0)
        ir_1 = ratios_module.excess_sharpe(noisy_returns_1, benchmark)
        ir_2 = ratios_module.excess_sharpe(noisy_returns_2, benchmark)
        ir_3 = ratios_module.excess_sharpe(noisy_returns_3, benchmark)
        assert abs(ir_1) < abs(ir_2)
        assert abs(ir_2) < abs(ir_3)

    @parameterized.expand([
        (pos_line, noise, flat_line_1_tz),
        (pos_line, inv_noise, flat_line_1_tz),
        (neg_line, noise, flat_line_1_tz),
        (neg_line, inv_noise, flat_line_1_tz),
    ])
    def test_excess_sharpe_trans(self, returns, add_noise, translation):
        """Test excess Sharpe changes with vertical translation."""
        ir = ratios_module.excess_sharpe(returns + add_noise, returns)
        raised_ir = ratios_module.excess_sharpe(returns + add_noise + translation, returns)
        depressed_ir = ratios_module.excess_sharpe(returns + add_noise - translation, returns)
        assert ir < raised_ir
        assert depressed_ir < ir


# Module-level reference
EMPYRICAL_MODULE = empyrical
