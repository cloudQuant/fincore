"""Tests for AnalysisContext — new tests, no modification to existing tests."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from fincore.core.context import AnalysisContext, analyze


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


class TestAnalysisContextBasic:
    def test_repr(self, returns):
        ctx = AnalysisContext(returns)
        r = repr(ctx)
        assert r.startswith("AnalysisContext(")
        assert "504 obs" in r

    def test_sharpe_ratio_cached(self, returns):
        ctx = AnalysisContext(returns)
        sr1 = ctx.sharpe_ratio
        sr2 = ctx.sharpe_ratio
        assert sr1 is sr2

    def test_sharpe_ratio_is_float(self, returns):
        ctx = AnalysisContext(returns)
        assert isinstance(ctx.sharpe_ratio, float)

    def test_annual_return(self, returns):
        ctx = AnalysisContext(returns)
        assert np.isfinite(ctx.annual_return)

    def test_cumulative_returns(self, returns):
        ctx = AnalysisContext(returns)
        assert np.isfinite(ctx.cumulative_returns)

    def test_annual_volatility(self, returns):
        ctx = AnalysisContext(returns)
        assert ctx.annual_volatility > 0

    def test_max_drawdown(self, returns):
        ctx = AnalysisContext(returns)
        assert ctx.max_drawdown <= 0

    def test_calmar_ratio(self, returns):
        ctx = AnalysisContext(returns)
        assert np.isfinite(ctx.calmar_ratio)

    def test_stability(self, returns):
        ctx = AnalysisContext(returns)
        assert 0 <= ctx.stability <= 1

    def test_omega_ratio(self, returns):
        ctx = AnalysisContext(returns)
        assert np.isfinite(ctx.omega_ratio)

    def test_sortino_ratio(self, returns):
        ctx = AnalysisContext(returns)
        assert np.isfinite(ctx.sortino_ratio)

    def test_skew(self, returns):
        ctx = AnalysisContext(returns)
        assert np.isfinite(ctx.skew)

    def test_kurtosis(self, returns):
        ctx = AnalysisContext(returns)
        assert np.isfinite(ctx.kurtosis)

    def test_tail_ratio(self, returns):
        ctx = AnalysisContext(returns)
        assert ctx.tail_ratio > 0

    def test_daily_value_at_risk(self, returns):
        ctx = AnalysisContext(returns)
        assert ctx.daily_value_at_risk < 0

    def test_sharpe_ratio_nan_for_too_short_series(self):
        returns = pd.Series([0.01], index=pd.date_range("2020-01-01", periods=1))
        ctx = AnalysisContext(returns)
        assert np.isnan(ctx.sharpe_ratio)


class TestAnalysisContextPerfStats:
    def test_perf_stats_keys(self, returns):
        ctx = AnalysisContext(returns)
        stats = ctx.perf_stats()
        assert "Annual return" in stats.index
        assert "Max drawdown" in stats.index
        assert "Sharpe ratio" in stats.index

    def test_perf_stats_matches_standalone(self, returns, factor_returns):
        from fincore.metrics.perf_stats import perf_stats

        ctx = AnalysisContext(returns, factor_returns=factor_returns)
        ctx_stats = ctx.perf_stats()
        standalone = perf_stats(returns, factor_returns=factor_returns)
        for key in standalone.index:
            if key in ctx_stats.index:
                np.testing.assert_allclose(
                    ctx_stats[key],
                    standalone[key],
                    rtol=1e-10,
                    err_msg=f"Mismatch in {key}",
                )

    def test_perf_stats_no_factor_returns(self, returns):
        ctx = AnalysisContext(returns)
        stats = ctx.perf_stats()
        assert "Alpha" not in stats.index
        assert "Beta" not in stats.index

    def test_perf_stats_with_factor_returns(self, returns, factor_returns):
        ctx = AnalysisContext(returns, factor_returns=factor_returns)
        stats = ctx.perf_stats()
        assert "Alpha" in stats.index
        assert "Beta" in stats.index


class TestAnalysisContextFactorMetrics:
    def test_alpha_beta_without_factor_returns(self, returns):
        ctx = AnalysisContext(returns)
        assert np.isnan(ctx.alpha)
        assert np.isnan(ctx.beta)

    def test_alpha_beta_with_factor_returns(self, returns, factor_returns):
        ctx = AnalysisContext(returns, factor_returns=factor_returns)
        assert np.isfinite(ctx.alpha)
        assert np.isfinite(ctx.beta)

    def test_information_ratio_without_factor_returns(self, returns):
        ctx = AnalysisContext(returns)
        assert np.isnan(ctx.information_ratio)

    def test_information_ratio_with_factor_returns(self, returns, factor_returns):
        ctx = AnalysisContext(returns, factor_returns=factor_returns)
        assert np.isfinite(ctx.information_ratio)


class TestAnalysisContextSerialization:
    def test_to_dict(self, returns):
        ctx = AnalysisContext(returns)
        d = ctx.to_dict()
        assert isinstance(d, dict)
        assert "Sharpe ratio" in d

    def test_to_json(self, returns):
        ctx = AnalysisContext(returns)
        j = ctx.to_json()
        parsed = json.loads(j)
        assert isinstance(parsed, dict)
        assert "Sharpe ratio" in parsed


class TestAnalysisContextInvalidate:
    def test_invalidate_recomputes(self, returns):
        ctx = AnalysisContext(returns)
        sr1 = ctx.sharpe_ratio
        ctx.invalidate()
        sr2 = ctx.sharpe_ratio
        assert sr1 == sr2

    def test_invalidate_preserves_data(self, returns, factor_returns):
        ctx = AnalysisContext(returns, factor_returns=factor_returns)
        _ = ctx.perf_stats()
        ctx.invalidate()
        assert ctx._returns is returns
        assert ctx._factor_returns is factor_returns


class TestAnalyzeConvenience:
    def test_analyze_returns_context(self, returns):
        ctx = analyze(returns)
        assert isinstance(ctx, AnalysisContext)

    def test_analyze_with_factor_returns(self, returns, factor_returns):
        ctx = analyze(returns, factor_returns=factor_returns)
        assert np.isfinite(ctx.alpha)

    def test_fincore_analyze(self, returns):
        import fincore

        ctx = fincore.analyze(returns)
        assert isinstance(ctx, AnalysisContext)
