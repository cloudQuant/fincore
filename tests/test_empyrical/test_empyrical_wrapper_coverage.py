"""Coverage completion for Empyrical wrapper methods and legacy adapters."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fincore._empyrical_legacy import (
    _legacy_aligned_binary_adapter,
    _legacy_beta_adapter,
    _legacy_calmar_adapter,
    _legacy_capture_adapter,
    _legacy_conditional_alpha_beta_adapter,
)
from fincore.empyrical import Empyrical


def _returns(n: int = 120, seed: int = 1) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0.0, 0.02, n), index=pd.date_range("2024-01-01", periods=n))


def _factor(n: int = 120, seed: int = 2) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0.0, 0.015, n), index=pd.date_range("2024-01-01", periods=n))


# ---------------------------------------------------------------------------
# Empyrical wrapper methods (thin delegates)
# ---------------------------------------------------------------------------


class TestEmpyricalWrapperMethods:
    def test_cagr(self):
        assert np.isfinite(Empyrical.cagr(_returns()))

    def test_max_drawdown_weeks(self):
        assert Empyrical.max_drawdown_weeks(_returns()) >= 0

    def test_max_drawdown_months(self):
        assert Empyrical.max_drawdown_months(_returns()) >= 0

    def test_max_drawdown_recovery_weeks(self):
        result = Empyrical.max_drawdown_recovery_weeks(_returns())
        assert np.isnan(result) or result >= 0

    def test_max_drawdown_recovery_months(self):
        result = Empyrical.max_drawdown_recovery_months(_returns())
        assert np.isnan(result) or result >= 0

    def test_gpd_risk_estimates(self):
        result = Empyrical.gpd_risk_estimates(_returns(200))
        assert len(result) == 5

    def test_gpd_risk_estimates_aligned(self):
        result = Empyrical.gpd_risk_estimates_aligned(_returns(200))
        assert len(result) == 5

    def test_roll_sharpe_ratio(self):
        result = Empyrical.roll_sharpe_ratio(_returns(), window=30)
        assert len(result) > 0

    def test_roll_max_drawdown(self):
        result = Empyrical.roll_max_drawdown(_returns(), window=30)
        assert len(result) > 0

    def test_beta_fragility_heuristic(self):
        result = Empyrical.beta_fragility_heuristic(_returns(), _factor())
        assert np.isfinite(result)

    def test_beta_fragility_heuristic_aligned(self):
        result = Empyrical.beta_fragility_heuristic_aligned(_returns(), _factor())
        assert np.isfinite(result)

    def test_roll_alpha(self):
        result = Empyrical.roll_alpha(_returns(), _factor(), window=30)
        assert len(result) > 0

    def test_roll_beta(self):
        result = Empyrical.roll_beta(_returns(), _factor(), window=30)
        assert len(result) > 0

    def test_roll_alpha_beta(self):
        result = Empyrical.roll_alpha_beta(_returns(), _factor(), window=30)
        assert result is not None

    def test_roll_up_capture(self):
        result = Empyrical.roll_up_capture(_returns(), _factor(), window=30)
        assert len(result) > 0

    def test_roll_down_capture(self):
        result = Empyrical.roll_down_capture(_returns(), _factor(), window=30)
        assert len(result) > 0

    def test_roll_up_down_capture(self):
        result = Empyrical.roll_up_down_capture(_returns(), _factor(), window=30)
        assert len(result) > 0


class TestEmpyricalStatefulMethods:
    def test_perf_attrib_stateful_resolution(self):
        dts = pd.date_range("2024-01-01", periods=10)
        dts.name = "dt"
        tickers = ["A", "B"]

        returns = pd.Series(np.random.default_rng(3).normal(0.0, 0.02, 10), index=dts)
        factor_returns = pd.DataFrame(
            {
                "factor1": np.random.default_rng(4).normal(0.0, 0.01, 10),
                "factor2": np.random.default_rng(5).normal(0.0, 0.01, 10),
            },
            index=dts,
        )
        index = pd.MultiIndex.from_product([dts, tickers], names=["dt", "ticker"])
        positions = pd.Series(np.random.default_rng(6).uniform(0.1, 0.9, 20), index=index)
        factor_loadings = pd.DataFrame(
            {
                "factor1": np.random.default_rng(7).uniform(0.1, 0.5, 20),
                "factor2": np.random.default_rng(8).uniform(0.1, 0.5, 20),
            },
            index=index,
        )
        emp = Empyrical(
            returns=returns,
            factor_returns=factor_returns,
            positions=positions,
            factor_loadings=factor_loadings,
        )
        result = emp.perf_attrib()
        assert result is not None

    def test_regression_annual_return(self):
        assert np.isfinite(Empyrical.regression_annual_return(_returns(), _factor()))

    def test_groupby_consecutive(self):
        txn = pd.DataFrame(
            {
                "symbol": ["A", "A", "B"],
                "amount": [1.0, -1.0, 2.0],
                "price": [10.0, 11.0, 20.0],
            },
            index=pd.to_datetime(["2024-01-01 10:00", "2024-01-01 11:00", "2024-01-02 10:00"]),
        )
        result = Empyrical._groupby_consecutive(txn)
        assert result is not None


# ---------------------------------------------------------------------------
# Legacy adapters
# ---------------------------------------------------------------------------


def _named_kernel(name: str):
    def kernel(*args, **kwargs):
        return (args, kwargs)

    kernel.__name__ = name
    return kernel


class TestLegacyAdapters:
    def test_legacy_capture_adapter_capture(self):
        kernel = _named_kernel("capture")
        result = _legacy_capture_adapter(
            kernel, {"returns": _returns(), "factor_returns": _factor()}
        )
        assert np.isfinite(result)

    def test_legacy_capture_adapter_up_capture(self):
        kernel = _named_kernel("up_capture")
        result = _legacy_capture_adapter(
            kernel, {"returns": _returns(), "factor_returns": _factor()}
        )
        assert np.isfinite(result)

    def test_legacy_capture_adapter_down_capture(self):
        kernel = _named_kernel("down_capture")
        result = _legacy_capture_adapter(
            kernel, {"returns": _returns(), "factor_returns": _factor()}
        )
        assert np.isfinite(result)

    def test_legacy_capture_adapter_up_down_capture(self):
        kernel = _named_kernel("up_down_capture")
        result = _legacy_capture_adapter(
            kernel, {"returns": _returns(), "factor_returns": _factor()}
        )
        assert np.isfinite(result)

    def test_legacy_capture_adapter_unexpected_kwarg(self):
        kernel = _named_kernel("capture")
        with pytest.raises(TypeError, match="unexpected keyword"):
            _legacy_capture_adapter(
                kernel,
                {
                    "returns": _returns(),
                    "factor_returns": _factor(),
                    "kwargs": {"bogus": 1},
                },
            )

    def test_legacy_capture_adapter_period_from_arguments(self):
        kernel = _named_kernel("capture")
        result = _legacy_capture_adapter(
            kernel,
            {
                "returns": _returns(),
                "factor_returns": _factor(),
                "period": "daily",
            },
        )
        assert np.isfinite(result)

    def test_legacy_conditional_alpha_beta_adapter_up(self):
        kernel = _named_kernel("up_alpha_beta")
        result = _legacy_conditional_alpha_beta_adapter(
            kernel, {"returns": _returns(), "factor_returns": _factor()}
        )
        assert result is not None

    def test_legacy_conditional_alpha_beta_adapter_down(self):
        kernel = _named_kernel("down_alpha_beta")
        result = _legacy_conditional_alpha_beta_adapter(
            kernel, {"returns": _returns(), "factor_returns": _factor()}
        )
        assert result is not None

    def test_legacy_beta_adapter(self):
        result = _legacy_beta_adapter(
            _named_kernel("beta"), {"returns": _returns(), "factor_returns": _factor()}
        )
        assert result is not None

    def test_legacy_calmar_adapter(self):
        def calmar_kernel(returns, period, annualization):
            return returns.mean()

        result = _legacy_calmar_adapter(
            calmar_kernel, {"returns": _returns()}
        )
        assert np.isfinite(result)

    def test_legacy_aligned_binary_adapter_unknown_name(self):
        with pytest.raises(KeyError, match="no legacy aligned projection"):
            _legacy_aligned_binary_adapter(
                _named_kernel("unknown_name"),
                {"returns": _returns(), "factor_returns": _factor()},
            )


class TestLegacyKernels:
    def test_legacy_max_drawdown_empty(self):
        from fincore._empyrical_legacy import _legacy_max_drawdown

        result = _legacy_max_drawdown(np.array([]))
        assert np.isnan(result)

    def test_legacy_max_drawdown_with_out(self):
        from fincore._empyrical_legacy import _legacy_max_drawdown

        out = np.empty(())
        result = _legacy_max_drawdown(np.array([0.01, -0.02, 0.03]), out=out)
        assert np.isfinite(result)

    def test_legacy_sharpe_ratio_empty(self):
        from fincore._empyrical_legacy import _legacy_sharpe_ratio

        result = _legacy_sharpe_ratio(np.array([0.01]))
        assert np.isnan(result)

    def test_legacy_sharpe_ratio_with_out(self):
        from fincore._empyrical_legacy import _legacy_sharpe_ratio

        out = np.empty(())
        result = _legacy_sharpe_ratio(np.array([0.01, -0.02, 0.03, 0.01]), out=out)
        assert np.isfinite(result)

    def test_make_strict_wrapper_requires_manifest_key(self):
        from dataclasses import replace

        from fincore import _dispatch
        from fincore.empyrical import _make_strict_wrapper

        spec = _dispatch.get_metric_spec("metrics", "sharpe_ratio", "enhanced")
        with pytest.raises(KeyError, match="signature manifest key"):
            _make_strict_wrapper(replace(spec, signature_manifest_key=None))
