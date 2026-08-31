"""Explicit canonical operation declarations for simulation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fincore.runtime import OperationSpec

from .base import annualize, compute_statistics, estimate_parameters, validate_returns
from .bootstrap import bootstrap, bootstrap_ci, bootstrap_summary
from .monte_carlo import MonteCarlo
from .paths import antithetic_variates, gbm_from_returns, geometric_brownian_motion, latin_hypercube_sampling
from .scenarios import generate_correlation_breakdown, scenario_table, stress_test

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = ["operations"]

_BINDINGS: tuple[tuple[str, Callable[..., Any], bool, str], ...] = (
    ("simulation.base.annualize", annualize, True, "none"),
    ("simulation.base.compute_statistics", compute_statistics, True, "none"),
    ("simulation.base.estimate_parameters", estimate_parameters, True, "none"),
    ("simulation.base.validate_returns", validate_returns, True, "none"),
    ("simulation.bootstrap.bootstrap", bootstrap, False, "explicit_seed"),
    ("simulation.bootstrap.bootstrap_ci", bootstrap_ci, False, "explicit_seed"),
    ("simulation.bootstrap.bootstrap_summary", bootstrap_summary, False, "explicit_seed"),
    ("simulation.monte_carlo.MonteCarlo.from_parameters", MonteCarlo.from_parameters, False, "explicit_seed"),
    ("simulation.paths.antithetic_variates", antithetic_variates, True, "none"),
    ("simulation.paths.gbm_from_returns", gbm_from_returns, False, "explicit_seed_or_generator"),
    ("simulation.paths.geometric_brownian_motion", geometric_brownian_motion, False, "explicit_seed_or_generator"),
    ("simulation.paths.latin_hypercube_sampling", latin_hypercube_sampling, False, "explicit_seed_or_generator"),
    ("simulation.scenarios.generate_correlation_breakdown", generate_correlation_breakdown, True, "none"),
    ("simulation.scenarios.scenario_table", scenario_table, True, "none"),
    ("simulation.scenarios.stress_test", stress_test, True, "none"),
)

_OPERATIONS = tuple(
    OperationSpec(
        operation_id=operation_id,
        capability_id=operation_id,
        domain="simulation",
        callable=callable_,
        deterministic=deterministic,
        rng_policy=rng_policy,
        provenance={"owner": "simulation", "kernel_module": callable_.__module__},
    )
    for operation_id, callable_, deterministic, rng_policy in _BINDINGS
)


def operations() -> tuple[OperationSpec, ...]:
    """Return immutable metadata for direct simulation operations."""

    return _OPERATIONS
