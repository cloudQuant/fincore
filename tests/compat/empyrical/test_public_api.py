from __future__ import annotations

import json
from pathlib import Path

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
