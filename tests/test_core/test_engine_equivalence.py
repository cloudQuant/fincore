"""Numeric equivalence gates between RollingEngine outputs and canonical metrics.

These tests pin the engine's rolling values to the canonical
``roll_*`` / enhanced rolling metrics so any drift in the shared-moment
implementation fails immediately.  The sortino gate is the anchor: the
engine previously divided by the sample standard deviation of clipped
returns, while the canonical ``roll_sortino_ratio`` divides by the
root-mean-square downside deviation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fincore.core.engine import RollingEngine
from fincore.metrics.rolling import (
    roll_beta,
    roll_max_drawdown,
    roll_sharpe_ratio,
    roll_sortino_ratio,
    rolling_volatility,
)

WINDOW = 63
ANN = 252.0


@pytest.fixture
def returns() -> pd.Series:
    rng = np.random.default_rng(42)
    index = pd.bdate_range("2015-01-01", periods=504)
    return pd.Series(rng.normal(0.0, 0.01, 504), index=index)


@pytest.fixture
def factor_returns() -> pd.Series:
    rng = np.random.default_rng(123)
    index = pd.bdate_range("2015-01-01", periods=504)
    return pd.Series(rng.normal(0.0, 0.008, 504), index=index)


def test_engine_sortino_matches_canonical_metric(returns: pd.Series) -> None:
    actual = RollingEngine(returns, window=WINDOW).compute(["sortino"])["sortino"]
    canonical = roll_sortino_ratio(returns, window=WINDOW)
    expected = pd.Series(canonical, index=returns.index[WINDOW - 1 :])
    pd.testing.assert_series_equal(actual, expected)


def test_engine_sharpe_matches_canonical_metric(returns: pd.Series) -> None:
    actual = RollingEngine(returns, window=WINDOW).compute(["sharpe"])["sharpe"]
    canonical = roll_sharpe_ratio(returns, window=WINDOW)
    expected = pd.Series(canonical, index=returns.index[WINDOW - 1 :])
    pd.testing.assert_series_equal(actual, expected)


def test_engine_volatility_matches_canonical_metric(returns: pd.Series) -> None:
    actual = RollingEngine(returns, window=WINDOW).compute(["volatility"])["volatility"]
    canonical = rolling_volatility(returns, rolling_vol_window=WINDOW)
    expected = canonical.iloc[WINDOW - 1 :]
    pd.testing.assert_series_equal(actual, expected)


def test_engine_max_drawdown_matches_canonical_metric(returns: pd.Series) -> None:
    actual = RollingEngine(returns, window=WINDOW).compute(["max_drawdown"])["max_drawdown"]
    canonical = roll_max_drawdown(returns, window=WINDOW)
    pd.testing.assert_series_equal(actual, canonical)


def test_engine_beta_matches_canonical_metric(returns: pd.Series, factor_returns: pd.Series) -> None:
    engine = RollingEngine(returns, factor_returns=factor_returns, window=WINDOW)
    actual = engine.compute(["beta"])["beta"]
    canonical = roll_beta(returns, factor_returns, window=WINDOW)
    pd.testing.assert_series_equal(actual, canonical)


def test_engine_mean_return_matches_canonical_formula(returns: pd.Series) -> None:
    actual = RollingEngine(returns, window=WINDOW).compute(["mean_return"])["mean_return"]
    canonical = (returns.rolling(WINDOW).mean() * ANN).iloc[WINDOW - 1 :]
    pd.testing.assert_series_equal(actual, canonical)


def test_batched_computation_matches_individual_computation(returns: pd.Series, factor_returns: pd.Series) -> None:
    engine = RollingEngine(returns, factor_returns=factor_returns, window=WINDOW)
    batched = engine.compute("all")
    for name in ("sharpe", "volatility", "sortino", "max_drawdown", "beta", "mean_return"):
        solo = engine.compute([name])[name]
        pd.testing.assert_series_equal(batched[name], solo)


def test_available_metrics_remain_registry_sourced(returns: pd.Series) -> None:
    from fincore.plugin.registry import registry as plugin_registry
    from fincore.plugin.specs import ROLLING_FAMILY

    engine = RollingEngine(returns, window=WINDOW)
    assert engine.available_metrics == plugin_registry.metric_names(family=ROLLING_FAMILY)


def test_engine_beta_requires_factor_data(returns: pd.Series) -> None:
    engine = RollingEngine(returns, window=WINDOW)
    with pytest.raises(ValueError, match="factor_returns required"):
        engine.compute(["beta"])
