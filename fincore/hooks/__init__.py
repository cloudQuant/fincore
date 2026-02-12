"""Event hooks and execution framework for fincore plugin system."""

from __future__ import annotations

from fincore.hooks.events import (
    AnalysisContext,
    OptimizationContext,
    ComputeContext,
    _EVENT_HOOKS,
    create_analysis_context,
    create_optimization_context,
    create_compute_context,
    execute_hooks,
    get_event_hooks,
    register_event_hook,
)

from fincore.hooks.registry import (
    _METRIC_REGISTRY,
    register_metric,
    register_viz_backend,
)

__all__ = [
    "AnalysisContext",
    "OptimizationContext",
    "ComputeContext",
    "_EVENT_HOOKS",
    "create_analysis_context",
    "create_optimization_context",
    "create_compute_context",
    "execute_hooks",
    "register_event_hook",
    "get_event_hooks",
    "register_metric",
    "register_viz_backend",
]