"""Regression checks that metrics kernels no longer install dynamic legacy surfaces."""

from __future__ import annotations

import os
from pathlib import Path

_DIRECT_MODULES = (
    "alpha_beta.py",
    "drawdown.py",
    "ratios.py",
    "returns.py",
    "risk.py",
    "rolling.py",
    "yearly.py",
)


def test_canonical_metric_kernels_do_not_import_or_install_legacy_dispatch_surfaces() -> None:
    metrics_root = (
        Path(os.environ.get("FINCORE_0042R2_SOURCE_ROOT", Path(__file__).parents[2])).resolve() / "fincore" / "metrics"
    )
    violations = {
        module_name: [
            forbidden
            for forbidden in ("fincore._dispatch", "install_metric_module_surface", "resolve_raw_metric")
            if forbidden in (metrics_root / module_name).read_text(encoding="utf-8")
        ]
        for module_name in _DIRECT_MODULES
    }

    assert {module_name: terms for module_name, terms in violations.items() if terms} == {}


def test_annual_return_has_one_canonical_kernel_path() -> None:
    from fincore.metrics import returns
    from fincore.metrics.yearly import annual_return

    assert not hasattr(returns, "annual_return")
    assert callable(annual_return)


def test_metrics_namespace_does_not_expose_legacy_module_aliases() -> None:
    import fincore.metrics as metrics

    assert not hasattr(metrics, "basic_module")
    assert not hasattr(metrics, "ratios_module")
