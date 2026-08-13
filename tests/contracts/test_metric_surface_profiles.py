from __future__ import annotations

import importlib
import inspect
import pkgutil

import numpy as np
import pandas as pd
import pytest

import fincore
from fincore import Empyrical
from fincore import empyrical as legacy
from fincore._registry import METRIC_REGISTRY
from fincore.core.context import AnalysisContext
from fincore.exceptions import DataAlignmentError, NumericalError

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


@pytest.mark.parametrize("surface", ["flat", "class", "metrics"])
def test_information_ratio_uses_enhanced_validation_on_every_real_surface(surface: str) -> None:
    from fincore.metrics import ratios

    returns = _returns_with_nan()
    factor_returns = pd.Series([0.005, 0.001, -0.004], index=returns.index)
    calls = {
        "flat": fincore.information_ratio,
        "class": Empyrical.information_ratio,
        "metrics": ratios.information_ratio,
    }

    with pytest.raises(NumericalError, match="finite"):
        calls[surface](returns, factor_returns)


def test_information_ratio_instance_binds_stored_returns_and_factor() -> None:
    returns = pd.Series([0.01, -0.005, 0.02], index=pd.date_range("2024-01-01", periods=3))
    factor_returns = pd.Series([0.004, -0.002, 0.006], index=returns.index)
    instance = Empyrical(returns=returns, factor_returns=factor_returns)

    assert instance.information_ratio() == Empyrical.information_ratio(returns, factor_returns)
    assert list(inspect.signature(instance.information_ratio).parameters) == [
        "period",
        "annualization",
        "alignment",
        "normalize_tz",
    ]


@pytest.mark.parametrize("surface", ["flat", "class", "metrics"])
def test_information_ratio_rejects_original_unsorted_inputs_before_inner_alignment(surface: str) -> None:
    from fincore.metrics import ratios

    returns = pd.Series(
        [0.01, 0.02, -0.01],
        index=pd.to_datetime(["2024-01-03", "2024-01-01", "2024-01-02"]),
    )
    factor_returns = pd.Series(
        [0.004, -0.002, 0.006],
        index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
    )
    calls = {
        "flat": fincore.information_ratio,
        "class": Empyrical.information_ratio,
        "metrics": ratios.information_ratio,
    }

    with pytest.raises(DataAlignmentError, match="sorted"):
        calls[surface](returns, factor_returns, alignment="inner")


@pytest.mark.parametrize("surface", ["flat", "class", "metrics"])
def test_information_ratio_checks_finite_values_after_inner_retention(surface: str) -> None:
    from fincore.metrics import ratios

    returns = pd.Series(
        [np.nan, 0.01, 0.02, -0.01],
        index=pd.date_range("2023-12-31", periods=4),
    )
    factor_returns = pd.Series(
        [0.004, -0.002, 0.006],
        index=pd.date_range("2024-01-01", periods=3),
    )
    calls = {
        "flat": fincore.information_ratio,
        "class": Empyrical.information_ratio,
        "metrics": ratios.information_ratio,
    }

    assert np.isfinite(calls[surface](returns, factor_returns, alignment="inner"))


def test_actual_shared_enhanced_exports_have_complete_dispatch_coverage() -> None:
    import fincore.metrics as metrics_package

    flat_exports = {
        name for name in fincore.__all__ if not name.startswith("_") and callable(getattr(fincore, name, None))
    }
    class_exports = {
        name for name in dir(Empyrical) if not name.startswith("_") and callable(getattr(Empyrical, name, None))
    }
    module_exports: dict[str, list[object]] = {}
    for module_info in pkgutil.iter_modules(metrics_package.__path__, f"{metrics_package.__name__}."):
        module = importlib.import_module(module_info.name)
        for name in getattr(module, "__all__", ()):
            public = getattr(module, name, None)
            if not name.startswith("_") and callable(public):
                module_exports.setdefault(name, []).append(public)

    shared_exports = flat_exports & class_exports & set(module_exports)
    assert shared_exports == {
        "aggregate_returns",
        "alpha",
        "alpha_beta",
        "annual_return",
        "annual_volatility",
        "beta",
        "calmar_ratio",
        "capture",
        "cum_returns",
        "cum_returns_final",
        "downside_risk",
        "information_ratio",
        "max_drawdown",
        "omega_ratio",
        "sharpe_ratio",
        "simple_returns",
        "sortino_ratio",
        "stability_of_timeseries",
        "tail_ratio",
        "value_at_risk",
    }

    missing_specs: list[tuple[str, str, str]] = []
    missing_markers: list[tuple[str, str, object]] = []
    expected_specs = {
        "flat": ("fincore_flat", "enhanced-0.3.x"),
        "class": ("empyrical_class", "stateful-enhanced"),
        "metrics": ("metrics", "enhanced"),
    }
    for name in sorted(shared_exports):
        public_surfaces = {
            "flat": [getattr(fincore, name)],
            "class": [getattr(Empyrical, name)],
            "metrics": module_exports[name],
        }
        for label, callables in public_surfaces.items():
            surface, variant = expected_specs[label]
            registry_key = (surface, name, variant)
            if registry_key not in METRIC_REGISTRY:
                missing_specs.append(registry_key)
            expected_marker = registry_key
            for public in callables:
                marker = getattr(public, "__fincore_dispatch_spec__", None)
                if marker != expected_marker:
                    missing_markers.append((label, name, marker))

    assert (missing_specs, missing_markers) == ([], [])


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
    sorted(
        [spec for spec in METRIC_REGISTRY.values() if spec.surface in {"metrics", "empyrical_class", "fincore_flat"}],
        key=lambda spec: (spec.surface, spec.public_name),
    ),
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
    sorted(
        [spec for spec in METRIC_REGISTRY.values() if spec.surface == "metrics" and spec.variant == "enhanced"],
        key=lambda spec: spec.public_name,
    ),
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


def test_cached_flat_metric_refreshes_its_kernel_after_module_reload() -> None:
    from fincore.metrics import yearly

    cached_flat = fincore.annual_return
    original_kernel = vars(yearly)["annual_return"]
    reloaded = importlib.reload(yearly)
    current_kernel = vars(reloaded)["annual_return"]

    assert current_kernel is not original_kernel
    assert cached_flat.__wrapped__ is current_kernel
    assert fincore.annual_return.__wrapped__ is current_kernel


def test_already_bound_empyrical_metric_refreshes_its_kernel_after_module_reload() -> None:
    from fincore.metrics import yearly

    returns = pd.Series([0.01, -0.005, 0.02], index=pd.date_range("2024-01-01", periods=3))
    instance = Empyrical(returns=returns)
    bound = instance.annual_return
    original_kernel = vars(yearly)["annual_return"]
    reloaded = importlib.reload(yearly)
    current_kernel = vars(reloaded)["annual_return"]

    assert current_kernel is not original_kernel
    assert bound.__wrapped__.__wrapped__ is current_kernel
    assert instance.annual_return is bound
