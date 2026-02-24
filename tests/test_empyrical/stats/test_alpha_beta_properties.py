"""Tests for alpha and beta property tests.

This module tests alpha/beta properties like translation invariance
and correlation effects.

Split from test_alpha_beta_core.py for maintainability.

Priority Markers:
- P1: Important property tests (translation, regression properties)
- P2: Edge cases with NaN handling
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_almost_equal
from parameterized import parameterized
from unittest import TestCase

from fincore import empyrical
from fincore.empyrical import Empyrical

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


@pytest.mark.p1  # High: important property tests
class TestAlphaBetaProperties(BaseTestCase):
    """Tests for alpha/beta properties like translation and correlation."""

    # Test data
    noise = pd.Series(
        rand.normal(0, 0.001, 1000),
        index=pd.date_range("2000-1-30", periods=1000, freq="D", tz="UTC")
    )

    sparse_noise = pd.Series(
        rand.normal(0, 0.001, 1000),
        index=pd.date_range("2000-1-30", periods=1000, freq="D", tz="UTC")
    )
    sparse_flat_line_1_tz = pd.Series(
        np.linspace(0.01, 0.01, num=1000),
        index=pd.date_range("2000-1-30", periods=1000, freq="D", tz="UTC")
    )

    # Add some NaN values to sparse_noise
    replace_nan = rand.choice(sparse_noise.index.tolist(), rand.randint(1, 10))
    sparse_noise = sparse_noise.replace(replace_nan, np.nan)
    replace_nan = rand.choice(sparse_flat_line_1_tz.index.tolist(), rand.randint(1, 10))
    sparse_flat_line_1_tz = sparse_flat_line_1_tz.replace(replace_nan, np.nan)

    # ========================================================================
    # Alpha/Beta Translation Tests
    # ========================================================================

    @parameterized.expand([
        (0, 0.001),
        (0.01, 0.001),
    ])
    def test_alpha_beta_translation(self, mean_returns, translation):
        """Test alpha/beta behavior under return translation."""
        # Generate correlated returns and benchmark.
        std_returns = 0.01
        correlation = 0.8
        std_bench = 0.001
        means = [mean_returns, 0.001]
        covs = [
            [std_returns**2, std_returns * std_bench * correlation],
            [std_returns * std_bench * correlation, std_bench**2],
        ]
        (ret, bench) = rand.multivariate_normal(means, covs, 1000).T
        returns = pd.Series(ret, index=pd.date_range("2000-1-30", periods=1000, freq="D"))
        benchmark = pd.Series(bench, index=pd.date_range("2000-1-30", periods=1000, freq="D"))

        # Translate returns and generate alphas and betas.
        returns_depressed = returns - translation
        returns_raised = returns + translation
        alpha_beta = Empyrical(return_types=np.ndarray).alpha_beta
        (alpha_depressed, beta_depressed) = alpha_beta(returns_depressed, benchmark)
        (alpha_standard, beta_standard) = alpha_beta(returns, benchmark)
        (alpha_raised, beta_raised) = alpha_beta(returns_raised, benchmark)

        # Alpha should change proportionally to how much returns were translated.
        assert_almost_equal(
            ((alpha_standard + 1) ** (1 / 252)) - ((alpha_depressed + 1) ** (1 / 252)),
            translation,
            DECIMAL_PLACES
        )
        assert_almost_equal(
            ((alpha_raised + 1) ** (1 / 252)) - ((alpha_standard + 1) ** (1 / 252)),
            translation,
            DECIMAL_PLACES
        )
        # Beta remains constant.
        assert_almost_equal(beta_standard, beta_depressed, DECIMAL_PLACES)
        assert_almost_equal(beta_standard, beta_raised, DECIMAL_PLACES)

    # Test alpha/beta with a smaller and larger correlation values.
    @parameterized.expand([(0.1, 0.9)])
    def test_alpha_beta_correlation(self, corr_less, corr_more):
        """Test alpha/beta relationship with correlation."""
        mean_returns = 0.01
        mean_bench = 0.001
        std_returns = 0.01
        std_bench = 0.001
        index = pd.date_range("2000-1-30", periods=1000, freq="D")

        # Generate less correlated returns
        means_less = [mean_returns, mean_bench]
        covs_less = [
            [std_returns**2, std_returns * std_bench * corr_less],
            [std_returns * std_bench * corr_less, std_bench**2],
        ]
        (ret_less, bench_less) = rand.multivariate_normal(means_less, covs_less, 1000).T
        returns_less = pd.Series(ret_less, index=index)
        benchmark_less = pd.Series(bench_less, index=index)

        # Genereate more highly correlated returns
        means_more = [mean_returns, mean_bench]
        covs_more = [
            [std_returns**2, std_returns * std_bench * corr_more],
            [std_returns * std_bench * corr_more, std_bench**2],
        ]
        (ret_more, bench_more) = rand.multivariate_normal(means_more, covs_more, 1000).T
        returns_more = pd.Series(ret_more, index=index)
        benchmark_more = pd.Series(bench_more, index=index)

        # Calculate alpha/beta values
        alpha_beta = Empyrical(return_types=np.ndarray).alpha_beta
        alpha_less, beta_less = alpha_beta(returns_less, benchmark_less)
        alpha_more, beta_more = alpha_beta(returns_more, benchmark_more)

        # Alpha determines by how much returns vary from the benchmark return.
        # A lower correlation leads to higher alpha.
        assert alpha_less > alpha_more
        # Beta measures the volatility of returns against benchmark returns.
        # Beta increases proportionally to correlation.
        assert beta_less < beta_more


@pytest.mark.p2  # Medium: NaN handling tests
class TestAlphaBetaNaN(BaseTestCase):
    """Tests for alpha/beta NaN handling."""

    # Test data with NaN
    sparse_noise = pd.Series(
        np.random.RandomState(1337).normal(0, 0.001, 1000),
        index=pd.date_range("2000-1-30", periods=1000, freq="D", tz="UTC")
    )

    # Add some NaN values
    rand = np.random.RandomState(1337)
    replace_nan = rand.choice(sparse_noise.index.tolist(), rand.randint(1, 10))
    sparse_noise = sparse_noise.replace(replace_nan, np.nan)

    # When faced with data containing np.nan, do not return np.nan. Calculate
    # alpha and beta using dates containing both.
    @parameterized.expand([
        (sparse_noise, sparse_noise),
    ])
    def test_alpha_beta_with_nan_inputs(self, returns, benchmark):
        """Test alpha/beta handles NaN inputs correctly."""
        alpha, beta = Empyrical(return_types=np.ndarray).alpha_beta(
            returns,
            benchmark,
        )
        self.assertFalse(np.isnan(alpha))
        self.assertFalse(np.isnan(beta))


@pytest.mark.p1  # High: important beta scaling tests
class TestBetaScaling(BaseTestCase):
    """Tests for beta scaling properties."""

    noise = pd.Series(
        np.random.RandomState(1337).normal(0, 0.001, 1000),
        index=pd.date_range("2000-1-30", periods=1000, freq="D")
    )

    @parameterized.expand([
        (noise, noise, 1.0),
        (2 * noise, noise, 2.0),
        (noise, -noise, -1.0),
        (2 * noise, -noise, -2.0),
    ])
    @pytest.mark.p0  # Critical: core financial metric
    def test_beta_scaling(self, returns, benchmark, expected):
        """Test beta calculation with scaling."""
        observed = Empyrical.beta(returns, benchmark)
        assert_almost_equal(observed, expected, 8)


# Module-level reference
EMPYRICAL_MODULE = empyrical
