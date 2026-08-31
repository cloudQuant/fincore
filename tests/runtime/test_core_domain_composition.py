"""Contracts for the direct core-domain operation surface."""

from __future__ import annotations

import importlib

from fincore.runtime import OperationCatalog


def test_core_domain_roots_are_namespaces_without_leaf_reexports() -> None:
    for module_name in ("fincore.simulation", "fincore.optimization", "fincore.data"):
        module = importlib.import_module(module_name)

        assert module.__all__ == []


def test_core_domain_operations_resolve_to_their_direct_kernels() -> None:
    from fincore.data.operations import operations as data_operations
    from fincore.data.providers import get_provider
    from fincore.optimization.objectives import optimize
    from fincore.optimization.operations import operations as optimization_operations
    from fincore.simulation.operations import operations as simulation_operations
    from fincore.simulation.paths import geometric_brownian_motion

    catalog = OperationCatalog((*simulation_operations(), *optimization_operations(), *data_operations()))

    assert catalog.resolve("simulation.paths.geometric_brownian_motion").callable is geometric_brownian_motion
    assert catalog.resolve("optimization.objectives.optimize").callable is optimize
    assert catalog.resolve("data.providers.get_provider").callable is get_provider
    assert catalog.resolve("simulation.paths.geometric_brownian_motion").rng_policy == "explicit_seed_or_generator"
    assert catalog.resolve("simulation.base.compute_statistics").deterministic


def test_optimization_error_has_a_direct_domain_path() -> None:
    from fincore.optimization.exceptions import OptimizationError

    error = OptimizationError("solver did not converge", status=9, solver_message="iteration limit")

    assert error.status == 9
    assert error.solver_message == "iteration limit"
