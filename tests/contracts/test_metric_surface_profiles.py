from __future__ import annotations

import importlib
import inspect

import numpy as np
import pandas as pd
import pytest

import fincore
from fincore import Empyrical
from fincore import empyrical as legacy
from fincore._registry import METRIC_REGISTRY
from fincore.core.context import AnalysisContext
from fincore.exceptions import NumericalError

CONTEXT_KERNELS = {
    "annual_return": "annual_return",
    "cumulative_returns": "cum_returns_final",
    "annual_volatility": "annual_volatility",
    "sharpe_ratio": "sharpe_ratio",
    "calmar_ratio": "calmar_ratio",
    "stability": "stability_of_timeseries",
    "max_drawdown": "max_drawdown",
    "omega_ratio": "omega_ratio",
    "sortino_ratio": "sortino_ratio",
    "skew": "skewness",
    "kurtosis": "kurtosis",
    "tail_ratio": "tail_ratio",
    "daily_value_at_risk": "value_at_risk",
    "alpha": "alpha_beta",
    "beta": "alpha_beta",
    "information_ratio": "information_ratio",
    "gross_leverage": "gross_lev",
    "turnover": "get_turnover",
}


def _returns_with_nan() -> pd.Series:
    return pd.Series([0.01, np.nan, -0.01], index=pd.date_range("2024-01-01", periods=3))


@pytest.mark.parametrize(
    "call",
    [
        lambda returns: fincore.sharpe_ratio(returns),
        lambda returns: Empyrical.sharpe_ratio(returns),
        lambda returns: AnalysisContext(returns).sharpe_ratio,
    ],
)
def test_enhanced_surfaces_share_the_same_domain_exception(call) -> None:
    with pytest.raises(NumericalError, match="finite"):
        call(_returns_with_nan())


def test_strict_legacy_surface_bypasses_enhanced_validation() -> None:
    result = legacy.sharpe_ratio(_returns_with_nan())
    assert np.isfinite(result)


def test_context_registry_covers_every_public_cached_output_and_alias() -> None:
    entries = {
        name: spec
        for (surface, name, variant), spec in METRIC_REGISTRY.items()
        if surface == "context" and variant == "cached-property"
    }
    assert set(entries) == set(CONTEXT_KERNELS)
    for public_name, kernel_name in CONTEXT_KERNELS.items():
        assert entries[public_name].kernel_ref.rsplit(":", 1)[1] == kernel_name
        assert entries[public_name].validation_profile == "context"


def test_context_registry_describes_real_input_binding_and_projection() -> None:
    expected = {
        "alpha": ("returns_factor", "scalar"),
        "beta": ("returns_factor", "scalar"),
        "gross_leverage": ("positions", "series"),
        "turnover": ("positions_transactions", "series"),
    }

    for public_name, contract in expected.items():
        spec = METRIC_REGISTRY[("context", public_name, "cached-property")]
        assert (spec.binding, spec.result_projection) == contract


@pytest.mark.parametrize(
    "spec",
    [spec for spec in METRIC_REGISTRY.values() if spec.surface in {"metrics", "empyrical_class", "fincore_flat"}],
    ids=lambda spec: f"{spec.surface}-{spec.public_name}",
)
def test_enhanced_out_metadata_matches_the_real_kernel_signature(spec) -> None:
    module_name, attribute = spec.kernel_ref.split(":", 1)
    kernel = getattr(importlib.import_module(module_name), attribute)
    parameters = inspect.signature(kernel).parameters
    accepts_out = "out" in parameters

    assert spec.out_policy == ("write_and_return" if accepts_out else "unsupported")


def test_context_alpha_beta_projection_is_registry_driven_and_computed_once(monkeypatch) -> None:
    import fincore._dispatch as dispatch

    returns = pd.Series([0.01, -0.01, 0.02], index=pd.date_range("2024-01-01", periods=3))
    factor = pd.Series([0.005, -0.002, 0.01], index=returns.index)
    calls = 0
    original_resolve = dispatch._resolve

    def shared_kernel(returns, factor_returns, risk_free=0.0, period="daily", annualization=None, out=None):
        nonlocal calls
        calls += 1
        return 1.25, 0.75

    def resolve(reference):
        if reference == "fincore.metrics.alpha_beta:alpha_beta":
            return shared_kernel
        return original_resolve(reference)

    monkeypatch.setattr(dispatch, "_resolve", resolve)

    projections = dispatch.invoke_prevalidated_projections(
        "context",
        ("alpha", "beta"),
        "cached-property",
        returns,
        factor,
        period="daily",
    )

    assert projections == {"alpha": 1.25, "beta": 0.75}
    assert calls == 1


def test_flat_out_contract_uses_registry_metadata_and_kernel_behavior() -> None:
    returns = pd.Series([0.01, -0.005, 0.02], index=pd.date_range("2024-01-01", periods=3))
    out = np.full((), 999.0)

    result = fincore.sharpe_ratio(returns, out=out)

    assert out.item() == result
    assert out.item() != 999.0


def test_metrics_surface_is_reached_through_its_exact_dispatch_spec() -> None:
    from fincore.metrics import ratios

    returns = pd.Series([0.01, -0.005, 0.02], index=pd.date_range("2024-01-01", periods=3))
    assert ratios.sharpe_ratio(returns) == fincore.sharpe_ratio(returns)


def test_real_metrics_module_path_uses_the_enhanced_validation_profile() -> None:
    from fincore.metrics import ratios

    with pytest.raises(NumericalError, match="finite"):
        ratios.sharpe_ratio(_returns_with_nan())


@pytest.mark.parametrize(
    "spec",
    [spec for spec in METRIC_REGISTRY.values() if spec.surface == "metrics" and spec.variant == "enhanced"],
    ids=lambda spec: spec.public_name,
)
def test_every_metrics_registry_entry_is_a_real_signature_preserving_wrapper(spec) -> None:
    module_name, kernel_attribute = spec.kernel_ref.split(":", 1)
    module = importlib.import_module(module_name)
    public = getattr(module, spec.public_name)
    raw = vars(module)["_fincore_metric_originals"][spec.public_name]

    assert public is not raw
    assert public.__fincore_dispatch_spec__ == ("metrics", spec.public_name, "enhanced")
    assert inspect.signature(public) == inspect.signature(raw)
    assert raw is vars(module)[kernel_attribute]
    assert spec.public_name in module.__all__


def test_metrics_alias_is_materialized_and_import_star_reaches_dispatch() -> None:
    from fincore.metrics import yearly

    namespace: dict[str, object] = {}
    exec("from fincore.metrics.yearly import *", {}, namespace)

    assert vars(yearly)["cagr"] is vars(yearly)["annual_return"]
    assert namespace["cagr"].__fincore_dispatch_spec__ == ("metrics", "cagr", "enhanced")
    assert namespace["cagr"].__name__ == "cagr"
    assert inspect.signature(namespace["cagr"]) == inspect.signature(vars(yearly)["annual_return"])


def test_metrics_alias_still_uses_enhanced_validation_after_module_reload() -> None:
    from fincore.metrics import yearly

    original_kernel = vars(yearly)["annual_return"]
    reloaded = importlib.reload(yearly)

    assert reloaded.cagr.__fincore_dispatch_spec__ == ("metrics", "cagr", "enhanced")
    assert vars(reloaded)["annual_return"] is not original_kernel
    assert vars(reloaded)["_fincore_metric_originals"]["cagr"] is vars(reloaded)["annual_return"]
    assert reloaded.cagr.__wrapped__ is vars(reloaded)["annual_return"]
    with pytest.raises(NumericalError, match="finite"):
        reloaded.cagr(_returns_with_nan())
