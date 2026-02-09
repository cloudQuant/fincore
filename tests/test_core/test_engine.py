"""Tests for RollingEngine and vectorized roll_max_drawdown."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fincore.core.engine import RollingEngine
from fincore.metrics.rolling import roll_max_drawdown


@pytest.fixture
def returns():
    np.random.seed(42)
    return pd.Series(
        np.random.randn(504) * 0.01,
        index=pd.bdate_range("2020-01-01", periods=504),
    )


@pytest.fixture
def factor_returns():
    np.random.seed(123)
    return pd.Series(
        np.random.randn(504) * 0.008,
        index=pd.bdate_range("2020-01-01", periods=504),
    )


class TestRollMaxDrawdownVectorized:
    def test_output_length(self, returns):
        window = 60
        result = roll_max_drawdown(returns, window=window)
        assert len(result) == len(returns) - window + 1

    def test_output_is_series(self, returns):
        result = roll_max_drawdown(returns, window=60)
        assert isinstance(result, pd.Series)

    def test_output_ndarray(self):
        np.random.seed(42)
        arr = np.random.randn(200) * 0.01
        result = roll_max_drawdown(arr, window=60)
        assert isinstance(result, np.ndarray)
        assert len(result) == 200 - 60 + 1

    def test_values_are_nonpositive(self, returns):
        result = roll_max_drawdown(returns, window=60)
        assert (result <= 0).all()

    def test_short_returns(self, returns):
        short = returns.iloc[:10]
        result = roll_max_drawdown(short, window=60)
        assert len(result) == 0

    def test_index_alignment(self, returns):
        window = 60
        result = roll_max_drawdown(returns, window=window)
        assert result.index[0] == returns.index[window - 1]
        assert result.index[-1] == returns.index[-1]

    def test_matches_loop_implementation(self):
        """Verify vectorized matches the original for-loop implementation."""
        np.random.seed(99)
        ret = np.random.randn(300) * 0.01
        window = 60
        from fincore.utils import nanmin as _nanmin

        # Original for-loop reference
        n = len(ret) - window + 1
        expected = np.empty(n, dtype=float)
        for i in range(n):
            window_ret = ret[i:i + window]
            cumulative = np.empty(window + 1, dtype='float64')
            cumulative[0] = 100.0
            np.cumprod(1 + window_ret, out=cumulative[1:])
            cumulative[1:] *= 100.0
            max_return = np.fmax.accumulate(cumulative)
            expected[i] = _nanmin((cumulative - max_return) / max_return)

        result = roll_max_drawdown(ret, window=window)
        np.testing.assert_allclose(result, expected, rtol=1e-12)


class TestRollingEngine:
    def test_compute_sharpe(self, returns):
        engine = RollingEngine(returns, window=60)
        results = engine.compute(['sharpe'])
        assert 'sharpe' in results
        assert len(results['sharpe']) > 0

    def test_compute_volatility(self, returns):
        engine = RollingEngine(returns, window=60)
        results = engine.compute(['volatility'])
        assert (results['volatility'] > 0).all()

    def test_compute_max_drawdown(self, returns):
        engine = RollingEngine(returns, window=60)
        results = engine.compute(['max_drawdown'])
        assert (results['max_drawdown'] <= 0).all()

    def test_compute_beta(self, returns, factor_returns):
        engine = RollingEngine(returns, factor_returns=factor_returns, window=60)
        results = engine.compute(['beta'])
        assert 'beta' in results

    def test_compute_beta_requires_factor(self, returns):
        engine = RollingEngine(returns, window=60)
        with pytest.raises(ValueError, match="factor_returns required"):
            engine.compute(['beta'])

    def test_compute_multiple(self, returns):
        engine = RollingEngine(returns, window=60)
        results = engine.compute(['sharpe', 'volatility', 'max_drawdown'])
        assert len(results) == 3

    def test_compute_mean_return(self, returns):
        engine = RollingEngine(returns, window=60)
        results = engine.compute(['mean_return'])
        assert 'mean_return' in results

    def test_unknown_metric(self, returns):
        engine = RollingEngine(returns, window=60)
        with pytest.raises(ValueError, match="Unknown metric"):
            engine.compute(['nonexistent'])

    def test_available_metrics(self, returns):
        engine = RollingEngine(returns, window=60)
        assert 'sharpe' in engine.available_metrics
        assert 'max_drawdown' in engine.available_metrics

    def test_compute_sortino(self, returns):
        engine = RollingEngine(returns, window=60)
        results = engine.compute(['sortino'])
        assert 'sortino' in results
