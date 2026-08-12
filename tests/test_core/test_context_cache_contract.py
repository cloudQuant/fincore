from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fincore.core.context import AnalysisContext
from fincore.exceptions import NumericalError


def _returns() -> pd.Series:
    return pd.Series([0.01, -0.02, 0.03, 0.005], index=pd.date_range("2024-01-01", periods=4))


def test_context_takes_an_immutable_snapshot() -> None:
    returns = _returns()
    ctx = AnalysisContext(returns)
    before = ctx.sharpe_ratio

    returns.iloc[:] = 0.0

    assert ctx.sharpe_ratio == before
    assert not ctx._returns.equals(returns)


def test_replace_data_invalidates_all_cached_metrics() -> None:
    returns = _returns()
    ctx = AnalysisContext(returns)
    before = (ctx.sharpe_ratio, ctx.annual_return)

    ctx.replace_data(returns=returns * -1)

    assert (ctx.sharpe_ratio, ctx.annual_return) != before


def test_failed_replace_data_is_atomic_and_preserves_cached_values() -> None:
    ctx = AnalysisContext(_returns())
    before_returns = ctx._returns.copy(deep=True)
    before_sharpe = ctx.sharpe_ratio
    invalid = _returns()
    invalid.iloc[0] = np.inf

    with pytest.raises(NumericalError):
        ctx.replace_data(returns=invalid)

    pd.testing.assert_series_equal(ctx._returns, before_returns)
    assert ctx.sharpe_ratio is before_sharpe


def test_alpha_and_beta_share_one_cached_alpha_beta_calculation(monkeypatch) -> None:
    import fincore._dispatch as dispatch

    returns = _returns()
    factor_returns = returns * 0.7 + 0.001
    calls = 0
    original = dispatch.invoke_prevalidated_projections

    def recording(surface, public_names, variant, *args, **kwargs):
        nonlocal calls
        if surface == "context" and public_names == ("alpha", "beta"):
            calls += 1
        return original(surface, public_names, variant, *args, **kwargs)

    monkeypatch.setattr(dispatch, "invoke_prevalidated_projections", recording)
    ctx = AnalysisContext(returns, factor_returns=factor_returns)

    assert np.isfinite(ctx.alpha)
    assert np.isfinite(ctx.beta)
    assert calls == 1


def test_positions_and_transactions_contribute_leverage_and_turnover_outputs() -> None:
    returns = _returns()
    positions = pd.DataFrame(
        {"AAA": [100.0, 110.0, 90.0, 120.0], "cash": [50.0, 40.0, 60.0, 30.0]},
        index=returns.index,
    )
    transactions = pd.DataFrame(
        {"amount": [2.0, -1.0], "price": [10.0, 11.0], "symbol": ["AAA", "AAA"]},
        index=pd.DatetimeIndex([returns.index[1] + pd.Timedelta(hours=10), returns.index[2] + pd.Timedelta(hours=10)]),
    )

    stats = AnalysisContext(returns, positions=positions, transactions=transactions).perf_stats()

    assert np.isfinite(stats["Average gross leverage"])
    assert np.isfinite(stats["Average turnover"])
