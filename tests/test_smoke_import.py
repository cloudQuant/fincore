"""Smoke test: verify every public fincore submodule can be imported.

Prevents syntax errors or broken top-level code from hiding until runtime.
"""

from __future__ import annotations

import importlib

import pytest

# All submodules that should be importable without side-effects
_MODULES = [
    "fincore",
    "fincore.constants",
    "fincore.core",
    "fincore.core.context",
    "fincore.core.engine",
    "fincore.empyrical",
    "fincore.metrics",
    "fincore.metrics.basic",
    "fincore.metrics.returns",
    "fincore.metrics.drawdown",
    "fincore.metrics.ratios",
    "fincore.metrics.alpha_beta",
    "fincore.metrics.rolling",
    "fincore.metrics.yearly",
    "fincore.plugin",
    "fincore.plugin.registry",
    "fincore.hooks",
    "fincore.hooks.events",
    "fincore.attribution.fama_french",
    "fincore.attribution.style",
    "fincore.data",
    "fincore.data.providers",
    "fincore._registry",
    "fincore._types",
    "fincore.utils",
    "fincore.optimization",
    "fincore.optimization.frontier",
    "fincore.optimization.risk_parity",
    "fincore.optimization.objectives",
    "fincore.report",
    "fincore.report.compute",
    "fincore.report.format",
    "fincore.report.render_html",
    "fincore.report.render_pdf",
    "fincore.simulation",
    "fincore.simulation.monte_carlo",
    "fincore.simulation.bootstrap",
]


@pytest.mark.parametrize("module_name", _MODULES)
def test_import(module_name: str) -> None:
    """Each module should import without error."""
    mod = importlib.import_module(module_name)
    assert mod is not None
