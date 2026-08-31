"""Direct-domain contracts for the canonical factor-analysis package."""

from __future__ import annotations

import importlib
import os
from pathlib import Path

from fincore.runtime import OperationCatalog


def test_factor_operations_resolve_to_direct_domain_kernels() -> None:
    from fincore.factor_analysis.analysis import analyze_factor
    from fincore.factor_analysis.operations import operations
    from fincore.factor_analysis.performance import factor_weights

    catalog = OperationCatalog(operations())

    assert catalog.resolve("factor_analysis.analysis.analyze_factor").callable is analyze_factor
    assert catalog.resolve("factor_analysis.performance.factor_weights").callable is factor_weights
    assert catalog.resolve("factor_analysis.analysis.analyze_factor").implementation_fingerprint == (
        "fincore.factor_analysis.analysis:analyze_factor"
    )


def test_factor_workflows_are_explicit_model_plans_not_rendering_wrappers() -> None:
    from fincore.factor_analysis.workflows import FACTOR_WORKFLOWS

    assert {workflow.workflow_id for workflow in FACTOR_WORKFLOWS} == {
        "event_returns",
        "event_study",
        "full",
        "information",
        "returns",
        "summary",
        "turnover",
    }
    assert {workflow.model_operation_id for workflow in FACTOR_WORKFLOWS} == {"factor_analysis.analysis.analyze_factor"}
    assert all(workflow.renderer is None for workflow in FACTOR_WORKFLOWS)


def test_factor_portfolio_input_model_has_no_legacy_tuple_projection() -> None:
    from fincore.factor_analysis import portfolio

    assert hasattr(portfolio, "FactorPortfolioInputs")
    assert hasattr(portfolio, "build_factor_portfolio_inputs")
    assert not hasattr(portfolio, "PyfolioFactorInputs")
    assert not hasattr(portfolio, "create_pyfolio_input")


def test_factor_root_is_namespace_only_without_reexported_leaf_callables() -> None:
    module = importlib.import_module("fincore.factor_analysis")

    assert module.__all__ == []
    assert "analyze_factor" not in module.__dict__
    assert "FactorAnalysisModel" not in module.__dict__


def test_factor_domain_does_not_import_legacy_facades_or_dispatch() -> None:
    package_root = Path(os.environ.get("FINCORE_0042R2_SOURCE_ROOT", Path(__file__).parents[2])).resolve() / "fincore" / "factor_analysis"
    forbidden = (
        "fincore._dispatch",
        "fincore._registry",
        "fincore.alphalens",
        "fincore.empyrical",
        "fincore.pyfolio",
    )

    violations = {
        path.name: token
        for path in package_root.glob("*.py")
        for token in forbidden
        if f"import {token}" in path.read_text(encoding="utf-8")
        or f"from {token} import" in path.read_text(encoding="utf-8")
    }

    assert violations == {}
