"""
Tests for new indicators implemented per docs/0025-校正指标计算方式/继续计算其他指标.md.

Tests for: relative_win_rate, mar_ratio, r_cubed_turtle, capm_r_squared,
           up_capture_return, down_capture_return.
"""

from unittest import TestCase

import numpy as np
import pandas as pd
from numpy.testing import assert_almost_equal

from fincore.empyrical import Empyrical
from fincore.metrics.ratios import down_capture_return, mar_ratio, up_capture_return
from fincore.metrics.stats import capm_r_squared, r_cubed_turtle, relative_win_rate

DECIMAL_PLACES = 4


def _make_series(values, start="2000-01-03", freq="D"):
    return pd.Series(
        np.array(values, dtype=float),
        index=pd.date_range(start, periods=len(values), freq=freq),
    )


# ---------------------------------------------------------------------------
# relative_win_rate
# ---------------------------------------------------------------------------


class TestRelativeWinRate(TestCase):
    def test_hand_calculation(self):
        """Hand calc: 3 out of 5 days strategy > benchmark."""
        returns = _make_series([0.02, -0.01, 0.03, 0.005, -0.02])
        benchmark = _make_series([0.01, 0.00, 0.01, 0.01, -0.01])
        # wins: day1(0.02>0.01), day3(0.03>0.01), day5(-0.02<-0.01 => no)
        # day2: -0.01<0.00 => no, day4: 0.005<0.01 => no
        # Actually: day1 win, day2 no, day3 win, day4 no, day5 no => 2/5
        # Wait: recheck: 0.02>0.01 yes, -0.01>0.00 no, 0.03>0.01 yes, 0.005>0.01 no, -0.02>-0.01 no
        expected = 2.0 / 5.0
        result = relative_win_rate(returns, benchmark)
        assert_almost_equal(result, expected, DECIMAL_PLACES)

    def test_always_outperform(self):
        returns = _make_series([0.02, 0.03, 0.04])
        benchmark = _make_series([0.01, 0.01, 0.01])
        result = relative_win_rate(returns, benchmark)
        assert_almost_equal(result, 1.0, DECIMAL_PLACES)

    def test_never_outperform(self):
        returns = _make_series([0.005, 0.005, 0.005])
        benchmark = _make_series([0.01, 0.01, 0.01])
        result = relative_win_rate(returns, benchmark)
        assert_almost_equal(result, 0.0, DECIMAL_PLACES)

    def test_identical_returns(self):
        """Ties don't count as wins."""
        returns = _make_series([0.01, 0.01, 0.01])
        result = relative_win_rate(returns, returns)
        assert_almost_equal(result, 0.0, DECIMAL_PLACES)

    def test_empty_nan(self):
        result = relative_win_rate(pd.Series([], dtype=float), pd.Series([], dtype=float))
        assert np.isnan(result)

    def test_via_empyrical(self):
        returns = _make_series([0.02, -0.01, 0.03, 0.005, -0.02])
        benchmark = _make_series([0.01, 0.00, 0.01, 0.01, -0.01])
        emp = Empyrical()
        result = emp.relative_win_rate(returns, benchmark)
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# mar_ratio
# ---------------------------------------------------------------------------


class TestMarRatio(TestCase):
    def test_hand_calculation(self):
        """MAR = mean(r) * ann_factor / |max_drawdown|."""
        np.random.seed(42)
        returns = _make_series(np.random.normal(0.001, 0.02, 252).tolist())

        from fincore.metrics.basic import annualization_factor
        from fincore.metrics.drawdown import max_drawdown

        ann_factor = annualization_factor("daily", None)
        mean_ret = np.nanmean(returns.values)
        max_dd = max_drawdown(returns)
        expected = (mean_ret * ann_factor) / abs(max_dd)

        result = mar_ratio(returns)
        assert_almost_equal(result, expected, DECIMAL_PLACES)

    def test_differs_from_calmar(self):
        """MAR uses arithmetic mean, Calmar uses CAGR => different values."""
        from fincore.metrics.ratios import calmar_ratio

        np.random.seed(42)
        returns = _make_series(np.random.normal(0.001, 0.02, 504).tolist())

        mar = mar_ratio(returns)
        calmar = calmar_ratio(returns)

        if not (np.isnan(mar) or np.isnan(calmar)):
            assert mar != calmar, f"MAR ({mar}) should differ from Calmar ({calmar})"

    def test_short_returns_nan(self):
        result = mar_ratio(_make_series([0.01]))
        assert np.isnan(result)

    def test_no_drawdown_nan(self):
        """All positive returns => no drawdown => NaN."""
        returns = _make_series([0.01, 0.02, 0.03])
        result = mar_ratio(returns)
        assert np.isnan(result)

    def test_via_empyrical(self):
        np.random.seed(42)
        returns = _make_series(np.random.normal(0.001, 0.02, 252).tolist())
        emp = Empyrical()
        result = emp.mar_ratio(returns)
        assert isinstance(result, (float, np.floating))


# ---------------------------------------------------------------------------
# r_cubed_turtle
# ---------------------------------------------------------------------------


class TestRCubedTurtle(TestCase):
    def test_positive_returns(self):
        """Positive trending returns should give positive R³ turtle."""
        np.random.seed(42)
        returns = _make_series(np.random.normal(0.002, 0.01, 504).tolist())
        result = r_cubed_turtle(returns)
        assert result > 0, f"Expected positive R³ turtle, got {result}"

    def test_short_returns_nan(self):
        result = r_cubed_turtle(_make_series([0.01]))
        assert np.isnan(result)

    def test_via_empyrical(self):
        np.random.seed(42)
        returns = _make_series(np.random.normal(0.001, 0.02, 504).tolist())
        emp = Empyrical()
        result = emp.r_cubed_turtle(returns)
        assert isinstance(result, (float, np.floating))


# ---------------------------------------------------------------------------
# capm_r_squared
# ---------------------------------------------------------------------------


class TestCapmRSquared(TestCase):
    def test_perfect_correlation(self):
        """If returns = a + b*benchmark, R² should be ~1."""
        np.random.seed(42)
        benchmark = _make_series(np.random.normal(0.001, 0.02, 252).tolist())
        returns = 0.001 + 1.2 * benchmark  # perfect linear
        result = capm_r_squared(returns, benchmark)
        assert_almost_equal(result, 1.0, 2)

    def test_uncorrelated(self):
        """Uncorrelated returns should have R² close to 0."""
        np.random.seed(42)
        returns = _make_series(np.random.normal(0.001, 0.02, 1000).tolist())
        np.random.seed(99)
        benchmark = _make_series(np.random.normal(0.001, 0.02, 1000).tolist())
        result = capm_r_squared(returns, benchmark)
        assert result < 0.1, f"Expected R² near 0 for uncorrelated, got {result}"

    def test_range_0_1(self):
        """R² should always be in [0, 1]."""
        np.random.seed(42)
        returns = _make_series(np.random.normal(0.001, 0.02, 252).tolist())
        benchmark = _make_series(np.random.normal(0.0005, 0.015, 252).tolist())
        result = capm_r_squared(returns, benchmark)
        assert 0 <= result <= 1, f"R² should be in [0,1], got {result}"

    def test_empty_nan(self):
        result = capm_r_squared(pd.Series([], dtype=float), pd.Series([], dtype=float))
        assert np.isnan(result)

    def test_via_empyrical(self):
        np.random.seed(42)
        returns = _make_series(np.random.normal(0.001, 0.02, 252).tolist())
        benchmark = _make_series(np.random.normal(0.0005, 0.015, 252).tolist())
        emp = Empyrical()
        result = emp.capm_r_squared(returns, benchmark)
        assert isinstance(result, (float, np.floating))


# ---------------------------------------------------------------------------
# up_capture_return / down_capture_return
# ---------------------------------------------------------------------------


class TestCaptureReturn(TestCase):
    def test_up_capture_return_positive_market(self):
        """Up capture return computed only on positive benchmark days."""
        np.random.seed(42)
        returns = _make_series(np.random.normal(0.001, 0.02, 252).tolist())
        benchmark = _make_series(np.random.normal(0.0005, 0.015, 252).tolist())
        result = up_capture_return(returns, benchmark)
        assert isinstance(result, (float, np.floating))
        assert not np.isnan(result)

    def test_down_capture_return_negative_market(self):
        """Down capture return computed only on negative benchmark days."""
        np.random.seed(42)
        returns = _make_series(np.random.normal(0.001, 0.02, 252).tolist())
        benchmark = _make_series(np.random.normal(0.0005, 0.015, 252).tolist())
        result = down_capture_return(returns, benchmark)
        assert isinstance(result, (float, np.floating))
        assert not np.isnan(result)

    def test_no_up_periods_nan(self):
        """All benchmark negative => up_capture_return is NaN."""
        returns = _make_series([-0.01, -0.02, -0.03])
        benchmark = _make_series([-0.01, -0.02, -0.03])
        result = up_capture_return(returns, benchmark)
        assert np.isnan(result)

    def test_no_down_periods_nan(self):
        """All benchmark positive => down_capture_return is NaN."""
        returns = _make_series([0.01, 0.02, 0.03])
        benchmark = _make_series([0.01, 0.02, 0.03])
        result = down_capture_return(returns, benchmark)
        assert np.isnan(result)

    def test_via_empyrical(self):
        np.random.seed(42)
        returns = _make_series(np.random.normal(0.001, 0.02, 252).tolist())
        benchmark = _make_series(np.random.normal(0.0005, 0.015, 252).tolist())
        emp = Empyrical()
        up = emp.up_capture_return(returns, benchmark)
        down = emp.down_capture_return(returns, benchmark)
        assert isinstance(up, (float, np.floating))
        assert isinstance(down, (float, np.floating))
