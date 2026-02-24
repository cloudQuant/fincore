"""Tests targeting specific uncovered lines in risk.py, drawdown.py, rolling.py, and data_utils.py."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fincore.metrics import drawdown as dd
from fincore.metrics import risk as rm
from fincore.metrics import rolling as rl
from fincore.utils.data_utils import roll, rolling_window, up, down


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _daily_returns(n=300, seed=42):
    rng = np.random.RandomState(seed)
    r = rng.normal(0.0005, 0.01, n)
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.Series(r, index=idx)


def _factor_returns(n=300, seed=99):
    rng = np.random.RandomState(seed)
    r = rng.normal(0.0003, 0.008, n)
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.Series(r, index=idx)


# ===================================================================
# D1 – risk.py coverage gaps
# ===================================================================

class TestAnnualVolatilityEdgeCases:
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


class TestDownsideRiskEdgeCases:
    def test_empty_returns_nan(self):
        """Cover lines 148-151: len(returns) < 1."""
        r = np.array([], dtype=float)
        assert np.isnan(rm.downside_risk(r))

    def test_dataframe_input(self):
        """Cover lines 168-169: DataFrame branch."""
        idx = pd.bdate_range("2020-01-01", periods=50)
        df = pd.DataFrame({"a": np.random.default_rng(0).normal(0, 0.01, 50),
                           "b": np.random.default_rng(1).normal(0, 0.01, 50)}, index=idx)
        result = rm.downside_risk(df)
        assert isinstance(result, pd.Series)
        assert len(result) == 2


class TestConditionalValueAtRiskEdge:
    def test_empty_returns_nan(self):
        """Cover lines 217-220."""
        assert np.isnan(rm.conditional_value_at_risk(pd.Series([], dtype=float)))


class TestTrackingError:
    def test_short_input_returns_nan(self):
        """Cover lines 294-298: len(returns) < 2."""
        r = np.array([0.01])
        f = np.array([0.02])
        assert np.isnan(rm.tracking_error(r, f))

    def test_normal_computation(self):
        """Cover lines 287-310: full tracking_error path."""
        r = _daily_returns()
        f = _factor_returns()
        result = rm.tracking_error(r, f)
        assert isinstance(result, float)
        assert result > 0


class TestVarExcessReturn:
    def test_short_returns_nan(self):
        """Cover line 410."""
        assert np.isnan(rm.var_excess_return(pd.Series([0.01])))


class TestVarCovVarNormal:
    def test_basic(self):
        """Cover lines 442-445."""
        result = rm.var_cov_var_normal(100_000, 0.95, mu=0.001, sigma=0.01)
        assert isinstance(result, float)
        assert result > 0


class TestGpdRiskEstimates:
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


class TestBetaFragilityHeuristic:
    def test_normal_computation(self):
        """Cover lines 655-680."""
        r = _daily_returns(50)
        f = _factor_returns(50)
        result = rm.beta_fragility_heuristic(r, f)
        assert isinstance(result, (float, np.floating))

    def test_aligned_wrapper(self):
        """Cover line 704."""
        r = _daily_returns(50)
        f = _factor_returns(50)
        result = rm.beta_fragility_heuristic_aligned(r, f)
        assert isinstance(result, (float, np.floating))


class TestGpdRiskEstimatesAligned:
    def test_wrapper(self):
        """Cover line 622."""
        r = _daily_returns(100)
        result = rm.gpd_risk_estimates_aligned(r)
        assert len(result) == 5


# ===================================================================
# D2 – drawdown.py coverage gaps
# ===================================================================

class TestMaxDrawdownEdge:
    def test_empty_returns_nan(self):
        """Cover lines 81-85: len(returns) < 1."""
        r = np.array([], dtype=float)
        assert np.isnan(dd.max_drawdown(r))

    def test_dataframe_returns_series(self):
        """Cover line 102-103: DataFrame branch."""
        idx = pd.bdate_range("2020-01-01", periods=50)
        df = pd.DataFrame({"a": np.random.default_rng(0).normal(0, 0.01, 50),
                           "b": np.random.default_rng(1).normal(0, 0.01, 50)}, index=idx)
        result = dd.max_drawdown(df)
        assert isinstance(result, pd.Series)


class TestMaxDrawdownDays:
    def test_empty_returns_nan(self):
        """Cover line 435."""
        assert np.isnan(dd.max_drawdown_days(pd.Series([], dtype=float)))

    def test_ndarray_input(self):
        """Cover line 438: non-Series conversion."""
        r = np.array([0.01, -0.02, 0.03, -0.01, -0.05, 0.02])
        result = dd.max_drawdown_days(r)
        assert isinstance(result, (int, np.integer))

    def test_datetime_index(self):
        """Cover line 448: DatetimeIndex path."""
        r = _daily_returns(50)
        result = dd.max_drawdown_days(r)
        assert isinstance(result, (int, np.integer))


class TestMaxDrawdownWeeks:
    def test_empty_returns_nan(self):
        """Cover lines 469-472."""
        assert np.isnan(dd.max_drawdown_weeks(pd.Series([], dtype=float)))

    def test_normal(self):
        r = _daily_returns(50)
        result = dd.max_drawdown_weeks(r)
        assert isinstance(result, float)


class TestMaxDrawdownMonths:
    def test_empty_returns_nan(self):
        """Cover lines 489-492."""
        assert np.isnan(dd.max_drawdown_months(pd.Series([], dtype=float)))


class TestMaxDrawdownRecoveryDays:
    def test_empty_returns_nan(self):
        """Cover lines 513-514."""
        assert np.isnan(dd.max_drawdown_recovery_days(pd.Series([], dtype=float)))

    def test_no_recovery(self):
        """Cover line 536: recovery doesn't happen."""
        r = pd.Series([-0.10, -0.10, -0.10, -0.10], index=pd.bdate_range("2020-01-01", periods=4))
        result = dd.max_drawdown_recovery_days(r)
        assert np.isnan(result)

    def test_with_recovery(self):
        """Cover lines 529-534."""
        r = pd.Series([0.10, -0.05, -0.03, 0.15, 0.10], index=pd.bdate_range("2020-01-01", periods=5))
        result = dd.max_drawdown_recovery_days(r)
        assert isinstance(result, (int, np.integer))

    def test_integer_index_recovery(self):
        """Cover line 534: non-DatetimeIndex branch."""
        r = pd.Series([0.10, -0.05, -0.03, 0.15, 0.10])
        result = dd.max_drawdown_recovery_days(r)
        assert isinstance(result, (int, np.integer))


class TestMaxDrawdownRecoveryWeeks:
    def test_empty_returns_nan(self):
        """Cover lines 553-556."""
        assert np.isnan(dd.max_drawdown_recovery_weeks(pd.Series([], dtype=float)))


class TestMaxDrawdownRecoveryMonths:
    def test_empty_returns_nan(self):
        """Cover lines 573-576."""
        assert np.isnan(dd.max_drawdown_recovery_months(pd.Series([], dtype=float)))


class TestSecondMaxDrawdown:
    def test_normal(self):
        """Cover lines 601-602, 627-628."""
        r = _daily_returns(100)
        result = dd.second_max_drawdown(r)
        assert isinstance(result, float)
        assert result <= 0


class TestDrawdownDetailedAndTopN:
    def test_identify_drawdown_periods_empty(self):
        """Cover line 132."""
        result = dd._identify_drawdown_periods(pd.Series([], dtype=float))
        assert result is None

    def test_get_top_drawdowns(self):
        """Cover line 673 via get_top_drawdowns."""
        r = _daily_returns(200)
        result = dd.get_top_drawdowns(r, top=3)
        assert isinstance(result, list)
        assert len(result) <= 3

    def test_gen_drawdown_table(self):
        """Cover line 722 via gen_drawdown_table."""
        r = _daily_returns(200)
        result = dd.gen_drawdown_table(r, top=3)
        assert isinstance(result, pd.DataFrame)


# ===================================================================
# D3 – rolling.py coverage gaps
# ===================================================================

class TestRollAlphaBeta:
    def test_ndarray_input(self):
        """Cover lines 180-198: non-Series input path."""
        r = np.random.default_rng(0).normal(0, 0.01, 50)
        f = np.random.default_rng(1).normal(0, 0.01, 50)
        result = rl.roll_alpha_beta(r, f, window=10)
        assert isinstance(result, pd.DataFrame)
        assert "alpha" in result.columns
        assert "beta" in result.columns

    def test_short_series_returns_empty_df(self):
        """Cover lines 173-178."""
        r = pd.Series([0.01, 0.02], index=pd.bdate_range("2020-01-01", periods=2))
        f = pd.Series([0.01, 0.02], index=pd.bdate_range("2020-01-01", periods=2))
        result = rl.roll_alpha_beta(r, f, window=100)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0


class TestRollSharpeRatio:
    def test_ndarray_input(self):
        """Cover lines 222-249: ndarray path."""
        r = np.random.default_rng(0).normal(0, 0.01, 50)
        result = rl.roll_sharpe_ratio(r, window=10)
        assert isinstance(result, np.ndarray)
        assert len(result) == 41

    def test_short_ndarray(self):
        """Cover line 229."""
        r = np.array([0.01, 0.02])
        result = rl.roll_sharpe_ratio(r, window=100)
        assert isinstance(result, np.ndarray)
        assert len(result) == 0

    def test_short_series_datetime(self):
        """Cover lines 224-228."""
        r = pd.Series([0.01], index=pd.bdate_range("2020-01-01", periods=1))
        result = rl.roll_sharpe_ratio(r, window=100)
        assert isinstance(result, pd.Series)
        assert len(result) == 0


class TestRollMaxDrawdown:
    def test_ndarray_input(self):
        """Cover lines 267-303: ndarray path."""
        r = np.random.default_rng(0).normal(0, 0.01, 50)
        result = rl.roll_max_drawdown(r, window=10)
        assert isinstance(result, np.ndarray)
        assert len(result) == 41

    def test_short_ndarray(self):
        """Cover line 274."""
        r = np.array([0.01])
        result = rl.roll_max_drawdown(r, window=100)
        assert isinstance(result, np.ndarray)
        assert len(result) == 0


class TestRollUpCapture:
    def test_ndarray_input(self):
        """Cover lines 323-346: ndarray path."""
        r = np.random.default_rng(0).normal(0, 0.01, 20)
        f = np.random.default_rng(1).normal(0, 0.01, 20)
        result = rl.roll_up_capture(r, f, window=5)
        assert isinstance(result, np.ndarray)

    def test_short_ndarray(self):
        """Cover line 332."""
        r = np.array([0.01])
        f = np.array([0.02])
        result = rl.roll_up_capture(r, f, window=100)
        assert isinstance(result, np.ndarray)
        assert len(result) == 0


class TestRollDownCapture:
    def test_ndarray_input(self):
        """Cover lines 366-389: ndarray path."""
        r = np.random.default_rng(0).normal(0, 0.01, 20)
        f = np.random.default_rng(1).normal(0, 0.01, 20)
        result = rl.roll_down_capture(r, f, window=5)
        assert isinstance(result, np.ndarray)


class TestRollUpDownCapture:
    def test_basic(self):
        """Cover lines 409-413."""
        r = _daily_returns(30)
        f = _factor_returns(30)
        result = rl.roll_up_down_capture(r, f, window=10)
        assert isinstance(result, pd.Series)


class TestRollingRegression:
    def test_short_input(self):
        """Cover line 522."""
        r = pd.Series([0.01, 0.02])
        f = pd.Series([0.01, 0.02])
        result = rl.rolling_regression(r, f, rolling_window=100)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_ndarray_input(self):
        """Cover lines 524-526."""
        r = np.random.default_rng(0).normal(0, 0.01, 50)
        f = np.random.default_rng(1).normal(0, 0.01, 50)
        result = rl.rolling_regression(r, f, rolling_window=10)
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0


# ===================================================================
# D4 – data_utils.py coverage gaps
# ===================================================================

class TestRollingWindow:
    def test_2d_raises(self):
        """Cover line 47."""
        with pytest.raises(ValueError, match="1D"):
            rolling_window(np.ones((3, 3)), 2)

    def test_window_too_large_raises(self):
        """Cover line 50."""
        with pytest.raises(ValueError, match="greater"):
            rolling_window(np.array([1, 2, 3]), 5)

    def test_normal(self):
        """Cover lines 44-56."""
        result = rolling_window(np.arange(5), 3)
        assert result.shape == (3, 3)


class TestRollPandasSingleArg:
    def test_single_arg_path(self):
        """Cover lines 74-76: single-arg _roll_pandas path."""
        r = _daily_returns(30)
        result = roll(r, function=np.mean, window=10)
        assert isinstance(result, pd.Series)
        assert len(result) == 21


class TestRollNdarraySingleArg:
    def test_single_arg_path(self):
        """Cover lines 99-101: single-arg _roll_ndarray path."""
        r = np.random.default_rng(0).normal(0, 0.01, 30)
        result = roll(r, function=np.mean, window=10)
        assert isinstance(result, np.ndarray)
        assert len(result) == 21

    def test_short_ndarray(self):
        """Cover line 94."""
        r = np.array([0.01])
        result = roll(r, function=np.mean, window=10)
        assert isinstance(result, np.ndarray)
        assert len(result) == 0


class TestRollValidation:
    def test_too_many_args_raises(self):
        """Cover line 140."""
        r = _daily_returns(10)
        with pytest.raises(ValueError, match="more than 2"):
            roll(r, r, r, function=np.mean, window=5)

    def test_mismatched_types_raises(self):
        """Cover lines 142-144."""
        r = _daily_returns(10)
        n = np.random.default_rng(0).normal(0, 0.01, 10)
        with pytest.raises(ValueError, match="not the same"):
            roll(r, n, function=np.mean, window=5)


def _sum_two(returns, factor_returns):
    """Helper: sum of returns (ignores factor_returns)."""
    return float(np.sum(returns))


class TestUpDown:
    def test_up_filters_positive(self):
        r = pd.Series([0.01, -0.02, 0.03])
        f = pd.Series([0.01, -0.01, 0.02])
        result = up(r, f, function=_sum_two)
        assert isinstance(result, float)

    def test_down_filters_negative(self):
        r = pd.Series([0.01, -0.02, 0.03])
        f = pd.Series([0.01, -0.01, 0.02])
        result = down(r, f, function=_sum_two)
        assert isinstance(result, float)
