"""Canonical namespace contracts for the breaking 0.5 public surface."""

from __future__ import annotations

import importlib


_EMPTY_CANONICAL_NAMESPACES = (
    "fincore.attribution",
    "fincore.data",
    "fincore.extensions",
    "fincore.factor_analysis",
    "fincore.metrics",
    "fincore.optimization",
    "fincore.performance",
    "fincore.portfolio",
    "fincore.report",
    "fincore.report.factor",
    "fincore.report.portfolio",
    "fincore.report.renderers",
    "fincore.risk",
)


def test_canonical_namespaces_load_without_package_root_compatibility_exports() -> None:
    """Domain roots remain importable but keep leaf APIs at their owner paths."""

    namespaces = {
        name: importlib.reload(importlib.import_module(name))
        for name in _EMPTY_CANONICAL_NAMESPACES
    }
    runtime = importlib.reload(importlib.import_module("fincore.runtime"))
    root = importlib.reload(importlib.import_module("fincore"))

    assert root.__all__ == [
        "__version__",
        "attribution",
        "data",
        "errors",
        "extensions",
        "factor_analysis",
        "metrics",
        "optimization",
        "performance",
        "portfolio",
        "report",
        "risk",
        "runtime",
        "simulation",
        "viz",
    ]
    assert all(module.__all__ == [] for module in namespaces.values())
    assert {"OperationCatalog", "OperationSpec", "run"}.issubset(runtime.__all__)
    assert not hasattr(root, "empyrical")
    assert not hasattr(root, "pyfolio")
    assert not hasattr(root, "alphalens")
