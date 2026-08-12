from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

import fincore.empyrical as ep
from fincore import _registry, empyrical

MANIFEST = Path(__file__).parents[1] / "fixtures" / "empyrical-0.6.0-api.json"


def _manifest() -> dict[str, object]:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def test_module_exports_all_frozen_public_symbols() -> None:
    expected = set(_manifest()["public_symbols"])
    assert len(expected) == 54
    assert expected <= set(ep.__all__)
    assert expected <= set(dir(ep))


def test_package_and_module_imports_resolve_to_same_module() -> None:
    assert empyrical is ep


def test_period_constants_are_literal_legacy_exports() -> None:
    assert (ep.DAILY, ep.WEEKLY, ep.MONTHLY, ep.QUARTERLY, ep.YEARLY) == (
        "daily",
        "weekly",
        "monthly",
        "quarterly",
        "yearly",
    )


def test_registry_uses_exact_multi_surface_metric_spec_schema() -> None:
    MetricSpec = _registry.MetricSpec
    metric_registry = _registry.METRIC_REGISTRY
    assert set(MetricSpec.__dataclass_fields__) == {
        "surface",
        "public_name",
        "variant",
        "kernel_ref",
        "adapter_ref",
        "signature_manifest_key",
        "binding",
        "validation_profile",
        "result_contract_key",
        "result_projection",
        "out_policy",
    }
    assert all(key == (spec.surface, spec.public_name, spec.variant) for key, spec in metric_registry.items())
    assert all(isinstance(spec.kernel_ref, str) and ":" in spec.kernel_ref for spec in metric_registry.values())
    assert all(isinstance(spec.adapter_ref, str) and ":" in spec.adapter_ref for spec in metric_registry.values())

    calmar_surfaces = {spec.surface for spec in metric_registry.values() if spec.public_name == "calmar_ratio"}
    assert {"empyrical_module", "fincore_flat", "empyrical_class", "metrics"} <= calmar_surfaces


def _strict_spec(name: str):
    return _registry.METRIC_REGISTRY[("empyrical_module", name, "strict-0.6.0")]


def test_strict_wrapper_rejects_unknown_signature_manifest_key() -> None:
    spec = replace(_strict_spec("sharpe_ratio"), signature_manifest_key="empyrical-0.6.0:missing")
    with pytest.raises(KeyError, match="signature manifest key"):
        ep._make_strict_wrapper(spec)


def test_strict_wrapper_rejects_manifest_symbol_mismatch() -> None:
    spec = replace(_strict_spec("sharpe_ratio"), signature_manifest_key="empyrical-0.6.0:beta")
    with pytest.raises(ValueError, match="does not match public name"):
        ep._make_strict_wrapper(spec)


@pytest.mark.parametrize(
    ("name", "out_policy"),
    [("sharpe_ratio", "unsupported"), ("cagr", "write_and_return")],
)
def test_strict_wrapper_rejects_out_policy_signature_mismatch(name: str, out_policy: str) -> None:
    spec = replace(_strict_spec(name), out_policy=out_policy)
    with pytest.raises(ValueError, match="out policy"):
        ep._make_strict_wrapper(spec)
