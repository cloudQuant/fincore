"""Unified invocation pipeline tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fincore.api import build_builtin_catalog
from fincore.api.adapters import is_strict_profile, route
from fincore.api.invoke import invoke, resolve_kernel
from fincore.exceptions import InputContractError
from fincore.results import AnalysisResult


def _catalog():
    return build_builtin_catalog()


def test_resolve_kernel_loads_metric() -> None:
    kernel = resolve_kernel("fincore.metrics.ratios:sharpe_ratio")
    assert callable(kernel)


def test_invoke_enhanced_returns_analysis_result() -> None:
    catalog = _catalog()
    returns = pd.Series([0.01, -0.005, 0.002, 0.004])
    result = invoke(catalog, "sharpe_ratio", "enhanced_v1", returns)
    assert isinstance(result, AnalysisResult)
    assert result.ok
    assert result.value is not None


def test_invoke_cashflow_performance_operation_uses_the_enhanced_catalog() -> None:
    catalog = _catalog()
    dates = pd.date_range("2024-01-31", periods=2, freq="ME", tz="UTC")
    valuations = pd.Series([100.0, 110.0], index=dates)
    cashflows = pd.Series([10.0], index=[dates[1]])

    result = invoke(catalog, "cashflow_adjusted_twr", "enhanced_v1", valuations, cashflows)

    assert result.ok
    assert result.value == pytest.approx(0.0, abs=1e-12)


def test_invoke_unknown_operation_raises() -> None:
    catalog = _catalog()
    with pytest.raises(KeyError, match="unknown operation"):
        invoke(catalog, "not_a_metric", "enhanced_v1")


def test_route_strict_returns_raw_value() -> None:
    catalog = _catalog()
    returns = pd.Series([0.01, -0.005, 0.002])
    value = route(catalog, "sharpe_ratio", "strict_empyrical_0_6_0", returns)
    assert isinstance(value, float)


def test_route_enhanced_returns_analysis_result() -> None:
    catalog = _catalog()
    returns = pd.Series([0.01, -0.005, 0.002])
    value = route(catalog, "sharpe_ratio", "enhanced_v1", returns)
    assert isinstance(value, AnalysisResult)


def test_is_strict_profile() -> None:
    assert is_strict_profile("strict_empyrical_0_6_0")
    assert not is_strict_profile("enhanced_v1")


def test_invoke_same_result_across_profiles_for_raw_kernel() -> None:
    catalog = _catalog()
    returns = pd.Series([0.01, -0.005, 0.002, 0.004])
    enhanced = invoke(catalog, "sharpe_ratio", "enhanced_v1", returns)
    strict = route(catalog, "sharpe_ratio", "strict_empyrical_0_6_0", returns)
    assert np.isclose(enhanced.value, strict, rtol=1e-9)
