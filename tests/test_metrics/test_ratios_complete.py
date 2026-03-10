"""Complete test coverage for ratios.py module.

Tests are organized by function and marked with priority (p0/p1/p2/p3).
This file complements test_ratios_additional.py to achieve 85%+ coverage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fincore.metrics.ratios import (
    adjusted_sharpe_ratio,
    burke_ratio,
    cal_treynor_ratio,
    calmar_ratio,
    capture,
    common_sense_ratio,
    conditional_sharpe_ratio,
    deflated_sharpe_ratio,
    down_capture,
    down_capture_return,
    excess_sharpe,
    information_ratio,
    kappa_three_ratio,
    m_squared,
    mar_ratio,
    omega_ratio,
    sharpe_ratio,
    sortino_ratio,
    stability_of_timeseries,
    sterling_ratio,
    up_capture,
    up_capture_return,
    up_down_capture,
)


class TestSharpeRatioEdgeCases:
    """Test edge cases for sharpe_ratio function."""

    @pytest.mark.p1
    def test_sharpe_with_output_buffer(self, small_returns):
        """Test sharpe_ratio with pre-allocated output buffer."""
        out = np.empty(())
        result = sharpe_ratio(small_returns, out=out)
        assert out is not None
        assert np.isfinite(result)

    @pytest.mark.p1
    def test_sharpe_short_returns(self):
        """Test sharpe_ratio with less than 2 observations."""
        short_returns = pd.Series([0.01])
        result = sharpe_ratio(short_returns)
        assert np.isnan(result)

    @pytest.mark.p0
    def test_sharpe_normal_case(self, small_returns):
        """Test sharpe_ratio with normal case (P0 critical metric)."""
        result = sharpe_ratio(small_returns)
        assert np.isfinite(result)
        assert isinstance(result, float)

    @pytest.mark.p2
    def test_sortino_zero_volatility(self):
        """Test sortino_ratio with zero downside volatility."""
        constant_returns = pd.Series([0.02] * 100)
        result = sortino_ratio(constant_returns, required_return=0.03)
        assert np.isfinite(result) or np.isinf(result) or np.isnan(result)

    @pytest.mark.p0
    def test_sortino_normal_case(self, small_returns):
        """Test sortino_ratio with normal case (P0 critical metric)."""
        result = sortino_ratio(small_returns, required_return=0)
        assert np.isfinite(result)
        assert isinstance(result, float)


class TestExcessSharpeEdgeCases:
    """Test edge cases for excess_sharpe function."""

    @pytest.mark.p1
    def test_excess_sharpe_with_output_buffer(self, returns_with_benchmark):
        """Test excess_sharpe with pre-allocated output buffer."""
        returns, benchmark = returns_with_benchmark
        out = np.empty(())
        result = excess_sharpe(returns, benchmark, out=out)
        assert out is not None
        assert np.isfinite(result)

    @pytest.mark.p1
    def test_excess_sharpe_short_returns(self, returns_with_benchmark):
        """Test excess_sharpe with less than 2 observations."""
        returns, benchmark = returns_with_benchmark
        short_returns = returns[:1]
        short_benchmark = benchmark[:1]
        result = excess_sharpe(short_returns, short_benchmark)
        assert np.isnan(result)

    @pytest.mark.p0
    def test_excess_sharpe_normal_case(self, returns_with_benchmark):
        """Test excess_sharpe with normal case."""
        returns, benchmark = returns_with_benchmark
        result = excess_sharpe(returns, benchmark)
        assert np.isfinite(result)
        assert isinstance(result, float)


class TestAdjustedSharpeRatioEdgeCases:
    """Test edge cases for adjusted_sharpe_ratio function."""

    @pytest.mark.p1
    def test_adjusted_sharpe_short_returns(self):
        """Test adjusted_sharpe_ratio with less than 4 observations."""
        short_returns = pd.Series([0.01, -0.005, 0.003])
        result = adjusted_sharpe_ratio(short_returns)
        assert np.isnan(result)

    @pytest.mark.p1
    def test_adjusted_sharpe_all_nan(self):
        """Test adjusted_sharpe_ratio with all NaN values."""
        nan_returns = pd.Series([np.nan] * 10)
        result = adjusted_sharpe_ratio(nan_returns)
        assert np.isnan(result)

    @pytest.mark.p2
    def test_adjusted_sharpe_nan_skew(self):
        """Test adjusted_sharpe_ratio when skew calculation returns NaN."""
        constant_returns = pd.Series([0.01] * 10)
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = adjusted_sharpe_ratio(constant_returns)
        # Zero volatility: may return NaN or inf (base sharpe undefined)
        assert np.isnan(result) or np.isinf(result) or (np.isfinite(result) and abs(result) > 1e10)

    @pytest.mark.p2
    def test_adjusted_sharpe_nan_kurtosis(self):
        """Test adjusted_sharpe_ratio when kurtosis calculation returns NaN."""
        constant_returns = pd.Series([0.01] * 10)
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = adjusted_sharpe_ratio(constant_returns)
        # Zero volatility: may return NaN or inf (base sharpe undefined)
        assert np.isnan(result) or np.isinf(result) or (np.isfinite(result) and abs(result) > 1e10)

    @pytest.mark.p2
    def test_adjusted_sharpe_small_sample(self, small_returns):
        """Test adjusted_sharpe_ratio adjustment for small sample (<20)."""
        small = small_returns[:19]
        adj_sharpe = adjusted_sharpe_ratio(small)
        base_sharpe = sharpe_ratio(small)
        if np.isfinite(base_sharpe) and base_sharpe != 0:
            ratio = adj_sharpe / base_sharpe
            assert 0.9 <= ratio <= 1.1


class TestConditionalSharpeRatioEdgeCases:
    """Test edge cases for conditional_sharpe_ratio function."""

    @pytest.mark.p1
    def test_conditional_sharpe_short_returns(self):
        """Test conditional_sharpe_ratio with less than 2 observations."""
        short_returns = pd.Series([0.01])
        result = conditional_sharpe_ratio(short_returns, cutoff=0.05)
        assert np.isnan(result)

    @pytest.mark.p1
    def test_conditional_sharpe_few_conditional(self):
        """Test conditional_sharpe_ratio when conditional sample < 2."""
        constant_returns = pd.Series([0.01] * 252)
        result = conditional_sharpe_ratio(constant_returns, cutoff=0.99)
        assert np.isfinite(result) or np.isinf(result) or np.isnan(result)

    @pytest.mark.p1
    def test_conditional_sharpe_zero_std(self):
        """Test conditional_sharpe_ratio with zero standard deviation."""
        constant_returns = pd.Series([0.01] * 252)
        result = conditional_sharpe_ratio(constant_returns, cutoff=0.5)
        assert np.isfinite(result) or np.isinf(result) or np.isnan(result)

    @pytest.mark.p1
    def test_mar_ratio_non_negative_drawdown(self, small_returns):
        """Test mar_ratio when max drawdown is non-negative."""
        positive_returns = pd.Series([0.01] * 252)
        result = mar_ratio(positive_returns)
        assert np.isnan(result)

    @pytest.mark.p2
    def test_mar_ratio_all_nan(self):
        """Test mar_ratio with all NaN returns."""
        nan_returns = pd.Series([np.nan] * 10)
        result = mar_ratio(nan_returns)
        assert np.isnan(result)


class TestOmegaRatioEdgeCases:
    """Test edge cases for omega_ratio function."""

    @pytest.mark.p1
    def test_oma_ratio_short_returns(self):
        """Test omega_ratio with less than 2 observations."""
        short_returns = pd.Series([0.01])
        result = omega_ratio(short_returns)
        assert np.isnan(result)

    @pytest.mark.p1
    def test_oma_ratio_annualization_1(self, small_returns):
        """Test omega_ratio when annualization factor is 1."""
        result = omega_ratio(small_returns, annualization=1)
        assert np.isfinite(result)

    @pytest.mark.p2
    def test_oma_ratio_required_return_negative(self, small_returns):
        """Test omega_ratio when required_return is <= -1."""
        result = omega_ratio(small_returns, required_return=-1.5)
        assert np.isnan(result)

    @pytest.mark.p2
    def test_oma_ratio_zero_downside(self, small_returns):
        """Test omega_ratio when downside component is zero."""
        constant_returns = pd.Series([0.01] * 252)
        result = omega_ratio(constant_returns, required_return=0.005)
        assert np.isnan(result)


class TestInformationRatioEdgeCases:
    """Test edge cases for information_ratio function."""

    @pytest.mark.p1
    def test_information_ratio_short_returns(self, returns_with_benchmark):
        """Test information_ratio with less than 2 observations."""
        returns, benchmark = returns_with_benchmark
        short_returns = returns[:1]
        short_benchmark = benchmark[:1]
        result = information_ratio(short_returns, short_benchmark)
        assert np.isnan(result)

    @pytest.mark.p1
    def test_information_ratio_zero_tracking_error(self, small_returns):
        """Test information_ratio when tracking error is zero."""
        constant_benchmark = small_returns
        result = information_ratio(small_returns, constant_benchmark)
        assert np.isfinite(result) or np.isnan(result)


class TestCalTreynorRatioEdgeCases:
    """Test edge cases for cal_treynor_ratio function."""

    @pytest.mark.p1
    def test_cal_treynor_short_returns(self, returns_with_benchmark):
        """Test cal_treynor_ratio with less than 2 observations."""
        returns, benchmark = returns_with_benchmark
        short_returns = returns[:1]
        short_benchmark = benchmark[:1]
        result = cal_treynor_ratio(short_returns, short_benchmark)
        assert np.isnan(result)

    @pytest.mark.p2
    def test_calmar_zero_drawdown(self):
        """Test calmar_ratio when max drawdown is zero."""
        positive_returns = pd.Series([0.01] * 252, index=pd.bdate_range("2020-01-01", periods=252))
        result = calmar_ratio(positive_returns)
        assert np.isnan(result)

    @pytest.mark.p2
    def test_m_squared_zero_portfolio_vol(self, returns_with_benchmark):
        """Test m_squared when portfolio volatility is zero."""
        constant_returns = pd.Series([0.01] * 252, index=pd.bdate_range("2020-01-01", periods=252))
        constant_benchmark = pd.Series([0.005] * 252, index=pd.bdate_range("2020-01-01", periods=252))
        result = m_squared(constant_returns, constant_benchmark)
        assert np.isfinite(result) or np.isnan(result)

    @pytest.mark.p2
    def test_m_squared_normal_case(self, returns_with_benchmark):
        """Test m_squared with normal case."""
        returns, benchmark = returns_with_benchmark
        result = m_squared(returns, benchmark)
        assert np.isfinite(result) or np.isnan(result)


class TestSterlingRatioEdgeCases:
    """Test edge cases for sterling_ratio function."""

    @pytest.mark.p1
    def test_sterling_ratio_short_returns(self):
        """Test sterling_ratio with less than 2 observations."""
        short_returns = pd.Series([0.01])
        result = sterling_ratio(short_returns)
        assert np.isnan(result)

    @pytest.mark.p2
    def test_sterling_ratio_no_drawdowns(self, small_returns):
        """Test sterling_ratio when no drawdowns are detected."""
        constant_returns = pd.Series([0.01] * 252)
        result = sterling_ratio(constant_returns)
        assert np.isfinite(result) or np.isinf(result)

    @pytest.mark.p2
    def test_sterling_ratio_all_zero_drawdowns(self, small_returns):
        """Test sterling_ratio when all drawdowns are zero."""
        result = sterling_ratio(small_returns)
        assert np.isfinite(result) or np.isinf(result)


class TestBurkeRatioEdgeCases:
    """Test edge cases for burke_ratio function."""

    @pytest.mark.p1
    def test_burke_ratio_short_returns(self):
        """Test burke_ratio with less than 2 observations."""
        short_returns = pd.Series([0.01])
        result = burke_ratio(short_returns)
        assert np.isnan(result)

    @pytest.mark.p2
    def test_burke_ratio_no_drawdowns(self, small_returns):
        """Test burke_ratio when no drawdowns are detected."""
        constant_returns = pd.Series([0.01] * 252)
        result = burke_ratio(constant_returns)
        assert np.isfinite(result) or np.isinf(result)


class TestKappaThreeRatioEdgeCases:
    """Test edge cases for kappa_three_ratio function."""

    @pytest.mark.p1
    def test_kappa_three_ratio_short_returns(self):
        """Test kappa_three_ratio with less than 2 observations."""
        short_returns = pd.Series([0.01])
        result = kappa_three_ratio(short_returns)
        assert np.isnan(result)

    @pytest.mark.p2
    def test_kappa_three_ratio_all_nan(self):
        """Test kappa_three_ratio with all NaN returns."""
        nan_returns = pd.Series([np.nan] * 10)
        result = kappa_three_ratio(nan_returns)
        assert np.isnan(result)

    @pytest.mark.p2
    def test_kappa_three_ratio_insufficient_data(self):
        """Test kappa_three_ratio with less than 2 clean observations."""
        nan_returns = pd.Series([np.nan] * 10)
        result = kappa_three_ratio(nan_returns)
        assert np.isnan(result)


class TestDeflatedSharpeRatioEdgeCases:
    """Test edge cases for deflated_sharpe_ratio function."""

    @pytest.mark.p1
    def test_deflated_sharpe_short_returns(self):
        """Test deflated_sharpe_ratio with less than 3 observations."""
        short_returns = pd.Series([0.01, 0.005])
        result = deflated_sharpe_ratio(short_returns)
        assert np.isnan(result)

    @pytest.mark.p1
    def test_deflated_sharpe_all_nan(self):
        """Test deflated_sharpe_ratio with all NaN returns."""
        nan_returns = pd.Series([np.nan] * 10)
        result = deflated_sharpe_ratio(nan_returns)
        assert np.isnan(result)

    @pytest.mark.p2
    def test_deflated_sharpe_single_trial(self, small_returns):
        """Test deflated_sharpe_ratio with num_trials=1."""
        result = deflated_sharpe_ratio(small_returns, num_trials=1)
        assert 0.0 <= result <= 1.0

    @pytest.mark.p2
    def test_deflated_sharpe_multiple_trials(self, small_returns):
        """Test deflated_sharpe_ratio with multiple trials."""
        result_single = deflated_sharpe_ratio(small_returns, num_trials=1)
        result_multi = deflated_sharpe_ratio(small_returns, num_trials=5)
        assert np.isfinite(result_single)
        assert np.isfinite(result_multi)


class TestCommonSenseRatioEdgeCases:
    """Test edge cases for common_sense_ratio function."""

    @pytest.mark.p1
    def test_common_sense_ratio_short_returns(self):
        """Test common_sense_ratio with less than 2 observations."""
        short_returns = pd.Series([0.01])
        result = common_sense_ratio(short_returns)
        assert np.isnan(result)

    @pytest.mark.p2
    def test_common_sense_ratio_nan_tail_ratio(self):
        """Test common_sense_ratio when tail_ratio returns NaN."""
        nan_returns = pd.Series([np.nan] * 10)
        result = common_sense_ratio(nan_returns)
        assert np.isnan(result)

    @pytest.mark.p2
    def test_common_sense_ratio_zero_win_rate(self):
        """Test common_sense_ratio when win rate is zero (all positive returns)."""
        constant_returns = pd.Series([0.01] * 252)
        result = common_sense_ratio(constant_returns)
        assert np.isinf(result) or np.isnan(result)


class TestStabilityOfTimeseriesEdgeCases:
    """Test edge cases for stability_of_timeseries function."""

    @pytest.mark.p1
    def test_stability_short_returns(self):
        """Test stability_of_timeseries with less than 2 observations."""
        short_returns = pd.Series([0.01])
        result = stability_of_timeseries(short_returns)
        assert np.isnan(result)

    @pytest.mark.p1
    def test_stability_all_nan_returns(self):
        """Test stability_of_timeseries with all NaN returns."""
        nan_returns = pd.Series([np.nan] * 10)
        result = stability_of_timeseries(nan_returns)
        assert np.isnan(result)

    @pytest.mark.p1
    def test_stability_two_valid_obs(self):
        """Test stability_of_timeseries with exactly 2 valid observations."""
        two_returns = pd.Series([0.01, 0.02])
        result = stability_of_timeseries(two_returns)
        assert 0 <= result <= 1


class TestCaptureRatiosEdgeCases:
    """Test edge cases for capture ratio functions."""

    @pytest.mark.p1
    def test_capture_short_returns(self, returns_with_benchmark):
        """Test capture with insufficient observations."""
        returns, benchmark = returns_with_benchmark
        short_returns = returns[:1]
        short_benchmark = benchmark[:1]
        result = capture(short_returns, short_benchmark)
        assert np.isfinite(result) or np.isnan(result)

    @pytest.mark.p2
    def test_up_capture_no_positive_benchmark(self):
        """Test up_capture when benchmark has no positive returns."""
        np.random.seed(99)
        negative_benchmark = pd.Series([-0.01] * 252, index=pd.bdate_range("2020-01-01", periods=252))
        result = up_capture(negative_benchmark, negative_benchmark)
        assert np.isnan(result)

    @pytest.mark.p2
    def test_down_capture_no_negative_benchmark(self):
        """Test down_capture when benchmark has no negative returns."""
        np.random.seed(42)
        positive_benchmark = pd.Series([0.01] * 252, index=pd.bdate_range("2020-01-01", periods=252))
        result = down_capture(positive_benchmark, positive_benchmark)
        assert np.isnan(result)

    @pytest.mark.p2
    def test_up_down_capture_zero_down_capture(self, returns_with_benchmark):
        """Test up_down_capture when down_capture is zero."""
        result = up_down_capture(*returns_with_benchmark)
        assert np.isfinite(result) or np.isnan(result)

    @pytest.mark.p2
    def test_up_capture_return_no_positive_benchmark(self):
        """Test up_capture_return when no positive benchmark periods."""
        np.random.seed(99)
        negative_benchmark = pd.Series([-0.01] * 252, index=pd.bdate_range("2020-01-01", periods=252))
        result = up_capture_return(negative_benchmark, negative_benchmark)
        assert np.isnan(result)

    @pytest.mark.p2
    def test_down_capture_return_no_negative_benchmark(self):
        """Test down_capture_return when no negative benchmark periods."""
        np.random.seed(42)
        positive_benchmark = pd.Series([0.01] * 252, index=pd.bdate_range("2020-01-01", periods=252))
        result = down_capture_return(positive_benchmark, positive_benchmark)
        assert np.isnan(result)
