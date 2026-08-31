"""Direct-domain contracts for canonical attribution operations."""

from __future__ import annotations

import importlib
import os
from pathlib import Path

from fincore.runtime import OperationCatalog


def test_attribution_root_is_a_namespace_without_leaf_reexports() -> None:
    module = importlib.import_module("fincore.attribution")

    assert module.__all__ == []
    assert "brinson_attribution" not in module.__dict__
    assert "FamaFrenchModel" not in module.__dict__


def test_attribution_operations_resolve_to_direct_domain_kernels() -> None:
    from fincore.attribution.brinson import brinson_attribution
    from fincore.attribution.operations import operations
    from fincore.attribution.performance import perf_attrib

    catalog = OperationCatalog(operations())

    assert catalog.resolve("attribution.brinson.brinson_attribution").callable is brinson_attribution
    assert catalog.resolve("attribution.performance.perf_attrib").callable is perf_attrib
    assert catalog.resolve("attribution.performance.perf_attrib").implementation_fingerprint == (
        "fincore.attribution.performance:perf_attrib"
    )


def test_attribution_domain_does_not_depend_on_legacy_dispatch_or_metric_aliases() -> None:
    package_root = (
        Path(os.environ.get("FINCORE_0042R2_SOURCE_ROOT", Path(__file__).parents[2])).resolve()
        / "fincore"
        / "attribution"
    )
    forbidden = (
        "fincore._dispatch",
        "fincore._registry",
        "fincore.metrics.perf_attrib",
        "fincore.empyrical",
        "fincore.pyfolio",
        "fincore.alphalens",
    )

    violations = {
        path.name: token
        for path in package_root.glob("*.py")
        for token in forbidden
        if f"import {token}" in path.read_text(encoding="utf-8")
        or f"from {token} import" in path.read_text(encoding="utf-8")
    }

    assert violations == {}


def test_attribution_provider_boundaries_have_no_process_global_registration_api() -> None:
    package_root = (
        Path(os.environ.get("FINCORE_0042R2_SOURCE_ROOT", Path(__file__).parents[2])).resolve()
        / "fincore"
        / "attribution"
    )
    forbidden = ("set_ff_provider", "clear_ff_factor_cache", "set_style_provider", "clear_style_factor_cache")

    violations = {
        path.name: token
        for path in (package_root / "fama_french.py", package_root / "style.py")
        for token in forbidden
        if token in path.read_text(encoding="utf-8")
    }

    assert violations == {}
