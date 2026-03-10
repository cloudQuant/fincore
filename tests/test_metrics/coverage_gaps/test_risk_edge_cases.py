"""Tests targeting specific uncovered lines in risk.py.

Part of test_coverage_gaps.py split - Risk module edge cases.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fincore.metrics import risk as rm


@pytest.fixture
def daily_returns():
    """Generate daily returns for testing."""
    rng = np.random.RandomState(42)
    r = rng.normal(0.0005, 0.01, 300)
    idx = pd.bdate_range("2020-01-01", periods=300)
    return pd.Series(r, index=idx)


@pytest.fixture
def factor_returns():
    """Generate factor returns for testing."""
    rng = np.random.RandomState(99)
    r = rng.normal(0.0003, 0.008, 300)
    idx = pd.bdate_range("2020-01-01", periods=300)
    return pd.Series(r, index=idx)


@pytest.mark.p2
class TestAnnualVolatilityEdgeCases:
    """Tests for annual_volatility edge cases."""

    def test_short_input_returns_nan(self):
        """Cover lines 92-95: len(returns) < 2 early return."""
        r = np.array([0.01])
        result = rm.annual_volatility(r)
        assert np.isnan(result)

    def test_2d_short_input(self):
        """Cover 2D branch of short-input path."""
        r = np.array([[0.01, 0.02]])
        result = rm.annual_volatility(r)
        assert isinstance(result, np.ndarray)
        assert np.all(np.isnan(result))


@pytest.mark.p2
class TestDownsideRiskEdgeCases:
    """Tests for downside_risk edge cases."""

    def test_empty_returns_nan(self):
        """Cover lines 148-151: len(returns) < 1."""
        r = np.array([], dtype=float)
        assert np.isnan(rm.downside_risk(r))

    def test_dataframe_input(self):
        """Cover lines 168-169: DataFrame branch."""
        idx = pd.bdate_range("2020-01-01", periods=50)
        df = pd.DataFrame(
            {"a": np.random.default_rng(0).normal(0, 0.01, 50), "b": np.random.default_rng(1).normal(0, 0.01, 50)},
            index=idx,
        )
        result = rm.downside_risk(df)
        assert isinstance(result, pd.Series)
        assert len(result) == 2


@pytest.mark.p2
class TestConditionalValueAtRiskEdge:
    """Tests for conditional_value_at_risk edge cases."""

    def test_empty_returns_nan(self):
        """Cover lines 217-220."""
        assert np.isnan(rm.conditional_value_at_risk(pd.Series([], dtype=float)))


@pytest.mark.p2
class TestTrackingError:
    """Tests for tracking_error function."""

    def test_short_input_returns_nan(self):
        """Cover lines 294-298: len(returns) < 2."""
        r = np.array([0.01])
        f = np.array([0.02])
        assert np.isnan(rm.tracking_error(r, f))

    def test_normal_computation(self, daily_returns, factor_returns):
        """Cover lines 287-310: full tracking_error path."""
        result = rm.tracking_error(daily_returns, factor_returns)
        assert isinstance(result, float)
        assert result > 0


@pytest.mark.p2
class TestVarExcessReturn:
    """Tests for var_excess_return function."""

    def test_short_returns_nan(self):
        """Cover line 410."""
        assert np.isnan(rm.var_excess_return(pd.Series([0.01])))


@pytest.mark.p2
class TestVarCovVarNormal:
    """Tests for var_cov_var_normal function."""

    def test_basic(self):
        """Cover lines 442-445."""
        result = rm.var_cov_var_normal(100_000, 0.95, mu=0.001, sigma=0.01)
        assert isinstance(result, float)
        assert result > 0


@pytest.mark.p2
class TestGpdRiskEstimates:
    """Tests for gpd_risk_estimates function."""

    def test_short_returns_zeros(self):
        """Cover lines 507-510: len(returns) < 3."""
        r = pd.Series([0.01, -0.02])
        result = rm.gpd_risk_estimates(r)
        assert isinstance(result, pd.Series)
        assert (result == 0).all()

    def test_short_ndarray_returns_zeros(self):
        r = np.array([0.01, -0.02])
        result = rm.gpd_risk_estimates(r)
        assert isinstance(result, np.ndarray)
        assert (result == 0).all()


@pytest.mark.p2
class TestBetaFragilityHeuristic:
    """Tests for beta_fragility_heuristic function."""

    def test_normal_computation(self, daily_returns, factor_returns):
        """Cover lines 655-680."""
        result = rm.beta_fragility_heuristic(daily_returns, factor_returns)
        assert isinstance(result, (float, np.floating))

    def test_aligned_wrapper(self, daily_returns, factor_returns):
        """Cover line 704."""
        result = rm.beta_fragility_heuristic_aligned(daily_returns, factor_returns)
        assert isinstance(result, (float, np.floating))


@pytest.mark.p2
class TestGpdRiskEstimatesAligned:
    """Tests for gpd_risk_estimates_aligned function."""

    def test_wrapper(self, daily_returns):
        """Cover line 622."""
        result = rm.gpd_risk_estimates_aligned(daily_returns)
        assert len(result) == 5
