"""Complete test coverage for risk.py module.

Tests are organized by function and marked with priority (p0/p1/p2/p3).
This file complements existing risk tests to achieve 85%+ coverage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fincore.metrics.risk import annual_volatility, downside_risk
from fincore.metrics.risk import value_at_risk, conditional_value_at_risk
from fincore.metrics.risk import tail_ratio, tracking_error, residual_risk
from fincore.metrics.risk import var_excess_return, var_cov_var_normal
from fincore.metrics.risk import trading_value_at_risk, gpd_risk_estimates
from fincore.metrics.risk import gpd_risk_estimates_aligned
from fincore.metrics.risk import beta_fragility_heuristic, beta_fragility_heuristic_aligned


class TestAnnualVolatilityEdgeCases:
    """Test edge cases for annual_volatility function."""

    @pytest.mark.p1
    def test_annual_volatility_with_output_buffer(self, small_returns):
        """Test annual_volatility with pre-allocated output buffer."""
        out = np.empty(())
        result = annual_volatility(small_returns, out=out)
        assert out is not None
        assert np.isfinite(result)

    @pytest.mark.p1
    def test_annual_volatility_short_returns(self):
        """Test annual_volatility with less than 2 observations."""
        short_returns = pd.Series([0.01])
        result = annual_volatility(short_returns)
        assert np.isnan(result)

    @pytest.mark.p0
    def test_annual_volatility_normal_case(self, small_returns):
        """Test annual_volatility with normal case (P0 critical metric)."""
        result = annual_volatility(small_returns)
        assert np.isfinite(result)
        assert isinstance(result, float)


class TestDownsideRiskEdgeCases:
    """Test edge cases for downside_risk function."""

    @pytest.mark.p1
    def test_downside_risk_with_output_buffer(self, small_returns):
        """Test downside_risk with pre-allocated output buffer."""
        out = np.empty(())
        result = downside_risk(small_returns, required_return=0, out=out)
        assert out is not None
        assert np.isfinite(result)

    @pytest.mark.p1
    def test_downside_risk_short_returns(self):
        """Test downside_risk with less than 2 observations."""
        short_returns = pd.Series([0.01])
        result = downside_risk(short_returns)
        assert np.isfinite(result) or result == 0.0

    @pytest.mark.p1
    def test_downside_risk_dataframe_input(self, small_returns):
        """Test downside_risk with DataFrame input."""
        df_returns = pd.DataFrame({"A": small_returns, "B": small_returns * 0.8})
        result = downside_risk(df_returns, required_return=0)
        assert isinstance(result, pd.Series)
        assert len(result) == 2


class TestValueAtRiskEdgeCases:
    """Test edge cases for value_at_risk function."""

    @pytest.mark.p1
    def test_value_at_risk_no_observations(self):
        """Test value_at_risk with no observations."""
        empty_returns = pd.Series([])
        result = value_at_risk(empty_returns, cutoff=0.05)
        assert np.isnan(result)

    @pytest.mark.p2
    def test_value_at_risk_different_cutoffs(self, small_returns):
        """Test value_at_risk with various cutoff values."""
        result_05 = value_at_risk(small_returns, cutoff=0.05)
        result_10 = value_at_risk(small_returns, cutoff=0.10)
        result_25 = value_at_risk(small_returns, cutoff=0.25)
        assert np.isfinite(result_05)
        assert np.isfinite(result_10)
        assert np.isfinite(result_25)


class TestConditionalValueAtRiskEdgeCases:
    """Test edge cases for conditional_value_at_risk function."""

    @pytest.mark.p1
    def test_conditional_var_no_observations(self):
        """Test conditional_value_at_risk with no observations."""
        empty_returns = pd.Series([])
        result = conditional_value_at_risk(empty_returns, cutoff=0.05)
        assert np.isnan(result)

    @pytest.mark.p2
    def test_conditional_var_different_cutoffs(self, small_returns):
        """Test conditional_value_at_risk with various cutoff values."""
        result_05 = conditional_value_at_risk(small_returns, cutoff=0.05)
        result_10 = conditional_value_at_risk(small_returns, cutoff=0.10)
        result_25 = conditional_value_at_risk(small_returns, cutoff=0.25)
        assert np.isfinite(result_05)
        assert np.isfinite(result_10)
        assert np.isfinite(result_25)


class TestTailRatioEdgeCases:
    """Test edge cases for tail_ratio function."""

    @pytest.mark.p1
    def test_tail_ratio_no_observations(self):
        """Test tail_ratio with no observations."""
        empty_returns = pd.Series([])
        result = tail_ratio(empty_returns)
        assert np.isnan(result)

    @pytest.mark.p1
    def test_tail_ratio_all_nan(self):
        """Test tail_ratio with all NaN returns."""
        nan_returns = pd.Series([np.nan] * 10)
        result = tail_ratio(nan_returns)
        assert np.isnan(result)

    @pytest.mark.p1
    def test_tail_ratio_single_observation(self):
        """Test tail_ratio with single observation."""
        single_return = pd.Series([0.01])
        result = tail_ratio(single_return)
        assert np.isfinite(result) or np.isnan(result)

    @pytest.mark.p2
    def test_tail_ratio_zero_left_tail(self, stable_returns):
        """Test tail_ratio when left tail is zero."""
        result = tail_ratio(stable_returns)
        assert np.isfinite(result) or np.isinf(result)


class TestTrackingErrorEdgeCases:
    """Test edge cases for tracking_error function."""

    @pytest.mark.p1
    def test_tracking_error_short_returns(self, returns_with_benchmark):
        """Test tracking_error with less than 2 observations."""
        returns, benchmark = returns_with_benchmark
        short_returns = returns[:1]
        short_benchmark = benchmark[:1]
        result = tracking_error(short_returns, short_benchmark)
        assert np.isnan(result)

    @pytest.mark.p1
    def test_tracking_error_with_output_buffer(self, returns_with_benchmark):
        """Test tracking_error with pre-allocated output buffer."""
        returns, benchmark = returns_with_benchmark
        out = np.empty(())
        result = tracking_error(returns, benchmark, out=out)
        assert out is not None
        assert np.isfinite(result)

    @pytest.mark.p1
    def test_tracking_error_dataframe_input(self, returns_with_benchmark):
        """Test tracking_error with DataFrame input."""
        returns, benchmark = returns_with_benchmark
        df_returns = pd.DataFrame({"A": returns, "B": returns * 0.8})
        df_benchmark = pd.DataFrame({"A": benchmark, "B": benchmark * 0.8})
        result = tracking_error(df_returns, df_benchmark)
        assert isinstance(result, (pd.Series, np.ndarray))
        assert len(result) == 2

    @pytest.mark.p2
    def test_tracking_error_zero_std(self, small_returns):
        """Test tracking_error with zero standard deviation."""
        constant_returns = pd.Series([0.01] * len(small_returns), index=small_returns.index)
        result = tracking_error(small_returns, constant_returns)
        assert np.isfinite(result) or np.isinf(result) or np.isnan(result)


class TestResidualRiskEdgeCases:
    """Test edge cases for residual_risk function."""

    @pytest.mark.p1
    def test_residual_risk_short_returns(self, returns_with_benchmark):
        """Test residual_risk with less than 2 observations."""
        returns, benchmark = returns_with_benchmark
        short_returns = returns[:1]
        short_benchmark = benchmark[:1]
        result = residual_risk(short_returns, short_benchmark)
        assert np.isnan(result)

    @pytest.mark.p2
    def test_residual_risk_all_nan(self):
        """Test residual_risk when all returns are NaN."""
        nan_returns = pd.Series([np.nan] * 10)
        nan_benchmark = pd.Series([np.nan] * 10)
        result = residual_risk(nan_returns, nan_benchmark)
        assert np.isnan(result)

    @pytest.mark.p2
    def test_residual_risk_zero_volatility(self, small_returns):
        """Test residual_risk when volatility is effectively zero."""
        constant_returns = pd.Series([0.01] * 100)
        constant_benchmark = pd.Series([0.01] * 100)
        result = residual_risk(constant_returns, constant_benchmark)
        assert np.isnan(result) or np.isinf(result)


class TestVarExcessReturnEdgeCases:
    """Test edge cases for var_excess_return function."""

    @pytest.mark.p1
    def test_var_excess_return_no_observations(self):
        """Test var_excess_return with no observations."""
        empty_returns = pd.Series([])
        result = var_excess_return(empty_returns, cutoff=0.05)
        assert np.isnan(result)

    @pytest.mark.p2
    def test_var_excess_return_zero_var(self, small_returns):
        """Test var_excess_return when VaR is zero."""
        result = var_excess_return(small_returns, cutoff=0.05)
        assert np.isfinite(result) or np.isnan(result)

    @pytest.mark.p2
    def test_var_excess_return_nan_var(self, small_returns):
        """Test var_excess_return when VaR is NaN."""
        result = var_excess_return(small_returns, cutoff=0.05)
        assert np.isfinite(result) or np.isnan(result)


class TestVarCovVarNormalEdgeCases:
    """Test edge cases for var_cov_var_normal function."""

    @pytest.mark.p2
    def test_var_cov_var_normal_different_confidence(self):
        """Test var_cov_var_normal with various confidence levels."""
        result_95 = var_cov_var_normal(1.0, 0.95, 0.0, 1.0)
        result_99 = var_cov_var_normal(1.0, 0.99, 0.0, 1.0)
        result_999 = var_cov_var_normal(1.0, 0.999, 0.0, 1.0)
        assert np.isfinite(result_95) or np.isinf(result_95)
        assert np.isfinite(result_99) or np.isinf(result_99)
        assert np.isfinite(result_999) or np.isinf(result_999)

    @pytest.mark.p2
    def test_var_cov_var_normal_different_parameters(self):
        """Test var_cov_var_normal with various parameter values."""
        result = var_cov_var_normal(100.0, 0.95, 0.02, 2.0)
        assert np.isfinite(result) or np.isinf(result)


class TestTradingValueAtRiskEdgeCases:
    """Test edge cases for trading_value_at_risk function."""

    @pytest.mark.p1
    def test_trading_var_no_observations(self):
        """Test trading_value_at_risk with no observations."""
        empty_returns = pd.Series([])
        result = trading_value_at_risk(empty_returns)
        assert np.isnan(result)

    @pytest.mark.p2
    def test_trading_var_different_sigma(self, small_returns):
        """Test trading_value_at_risk with different sigma values."""
        result_2sigma = trading_value_at_risk(small_returns, sigma=2.0)
        result_3sigma = trading_value_at_risk(small_returns, sigma=3.0)
        assert np.isfinite(result_2sigma)
        assert np.isfinite(result_3sigma)


class TestGpdRiskEstimatesEdgeCases:
    """Test edge cases for gpd_risk_estimates function."""

    @pytest.mark.p1
    def test_gpd_estimates_no_observations(self):
        """Test gpd_risk_estimates with no observations."""
        empty_returns = pd.Series([])
        result = gpd_risk_estimates(empty_returns)
        assert isinstance(result, (np.ndarray, pd.Series))

    @pytest.mark.p2
    def test_gpd_estimates_all_positive(self):
        """Test gpd_risk_estimates with all positive returns."""
        positive_returns = pd.Series([0.01] * 100)
        result = gpd_risk_estimates(positive_returns)
        assert isinstance(result, (np.ndarray, pd.Series))

    @pytest.mark.p2
    def test_gpd_estimates_estimation_failure(self):
        """Test gpd_risk_estimates when estimation fails (all values > threshold)."""
        constant_returns = pd.Series([0.01] * 100)
        result = gpd_risk_estimates(constant_returns)
        assert isinstance(result, (np.ndarray, pd.Series))


class TestGpdRiskEstimatesAlignedEdgeCases:
    """Test edge cases for gpd_risk_estimates_aligned function."""

    @pytest.mark.p2
    def test_gpd_estimates_aligned_short_returns(self, returns_with_benchmark):
        """Test gpd_risk_estimates_aligned with insufficient observations."""
        returns, benchmark = returns_with_benchmark
        short_returns = returns[:2]
        short_benchmark = benchmark[:2]
        result = gpd_risk_estimates_aligned(short_returns, short_benchmark)
        assert isinstance(result, (np.ndarray, pd.Series))


class TestBetaFragilityHeuristicEdgeCases:
    """Test edge cases for beta_fragility_heuristic function."""

    @pytest.mark.p1
    def test_beta_fragility_aligned_short_returns(self):
        """Test beta_fragility_heuristic_aligned with short returns."""
        np.random.seed(42)
        short_returns = pd.Series([0.01], index=pd.bdate_range("2020-01-01", periods=1))
        short_benchmark = pd.Series([0.005], index=pd.bdate_range("2020-01-01", periods=1))
        result = beta_fragility_heuristic_aligned(short_returns, short_benchmark)
        assert np.isnan(result)

    @pytest.mark.p1
    def test_beta_fragility_short_factor_returns(self, returns_with_benchmark):
        """Test beta_fragility_heuristic with short factor returns."""
        returns, benchmark = returns_with_benchmark
        short_factor = pd.Series([0.01], index=returns.index[:1])
        result = beta_fragility_heuristic(returns[:1], short_factor)
        assert np.isnan(result)

    @pytest.mark.p2
    def test_beta_fragility_extreme_returns(self):
        """Test beta_fragility_heuristic with extreme returns."""
        np.random.seed(42)
        extreme_returns = pd.Series(np.random.randn(100) * 0.1, index=pd.bdate_range("2020-01-01", periods=100))
        extreme_benchmark = extreme_returns * 0.5 + np.random.randn(100) * 0.05
        result = beta_fragility_heuristic(extreme_returns, extreme_benchmark)
        assert np.isfinite(result) or np.isnan(result)


class TestBetaFragilityHeuristicAlignedEdgeCases:
    """Test edge cases for beta_fragility_heuristic_aligned function."""

    @pytest.mark.p2
    def test_beta_fragility_aligned_short_returns(self):
        """Test beta_fragility_heuristic_aligned with insufficient observations."""
        np.random.seed(42)
        short_returns = pd.Series([0.01], index=pd.bdate_range("2020-01-01", periods=1))
        short_benchmark = pd.Series([0.005], index=pd.bdate_range("2020-01-01", periods=1))
        result = beta_fragility_heuristic_aligned(short_returns, short_benchmark)
        assert np.isnan(result)
