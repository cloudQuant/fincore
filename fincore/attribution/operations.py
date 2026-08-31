"""Explicit canonical operation declarations for attribution analytics."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fincore.runtime import OperationSpec

from .brinson import brinson_attribution, brinson_cumulative, brinson_results
from .fama_french import calculate_idiosyncratic_risk, fetch_ff_factors
from .performance import (
    align_and_warn,
    compute_exposures,
    compute_exposures_internal,
    create_perf_attrib_stats,
    cumulative_returns_less_costs,
    perf_attrib,
    perf_attrib_core,
)
from .style import (
    analyze_performance_by_style,
    calculate_regression_attribution,
    calculate_style_tilts,
    fetch_style_factors,
    style_analysis,
)

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = ["operations"]

_BINDINGS: tuple[tuple[str, Callable[..., Any]], ...] = (
    ("attribution.brinson.brinson_attribution", brinson_attribution),
    ("attribution.brinson.brinson_cumulative", brinson_cumulative),
    ("attribution.brinson.brinson_results", brinson_results),
    ("attribution.fama_french.calculate_idiosyncratic_risk", calculate_idiosyncratic_risk),
    ("attribution.fama_french.fetch_ff_factors", fetch_ff_factors),
    ("attribution.performance.align_and_warn", align_and_warn),
    ("attribution.performance.compute_exposures", compute_exposures),
    ("attribution.performance.compute_exposures_internal", compute_exposures_internal),
    ("attribution.performance.create_perf_attrib_stats", create_perf_attrib_stats),
    ("attribution.performance.cumulative_returns_less_costs", cumulative_returns_less_costs),
    ("attribution.performance.perf_attrib", perf_attrib),
    ("attribution.performance.perf_attrib_core", perf_attrib_core),
    ("attribution.style.analyze_performance_by_style", analyze_performance_by_style),
    ("attribution.style.calculate_regression_attribution", calculate_regression_attribution),
    ("attribution.style.calculate_style_tilts", calculate_style_tilts),
    ("attribution.style.fetch_style_factors", fetch_style_factors),
    ("attribution.style.style_analysis", style_analysis),
)

_OPERATIONS = tuple(
    OperationSpec(
        operation_id=operation_id,
        capability_id=operation_id,
        domain="attribution",
        callable=callable_,
        provenance={"owner": "attribution", "kernel_module": callable_.__module__},
    )
    for operation_id, callable_ in _BINDINGS
)


def operations() -> tuple[OperationSpec, ...]:
    """Return immutable metadata for direct attribution operations."""

    return _OPERATIONS
