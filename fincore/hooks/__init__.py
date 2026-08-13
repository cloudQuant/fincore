"""Event hooks and execution framework for the fincore plugin system.

Hook storage is shared with the single extension registry
(:mod:`fincore.plugin.registry`): hooks registered through
:func:`fincore.plugin.register_hook` and through
:func:`fincore.hooks.register_event_hook` share one pipeline.
"""

from __future__ import annotations

import warnings
from typing import Any

from fincore.hooks import events

# Re-export events module contents
_EVENT_HOOKS = events._EVENT_HOOKS
AnalysisHookContext = events.AnalysisHookContext
ComputeContext = events.ComputeContext
OptimizationContext = events.OptimizationContext
create_analysis_context = events.create_analysis_context
create_compute_context = events.create_compute_context
create_optimization_context = events.create_optimization_context
execute_hooks = events.execute_hooks
get_event_hooks = events.get_event_hooks
register_event_hook = events.register_event_hook
list_events = events.list_events
clear_hooks = events.clear_hooks


def __getattr__(name: str) -> Any:
    """Deprecated hook-module attributes resolve lazily, with a warning."""
    if name == "AnalysisContext":
        warnings.warn(
            "fincore.hooks.AnalysisContext is deprecated; use AnalysisHookContext instead. "
            "(The name collided with fincore.core.context.AnalysisContext.)",
            DeprecationWarning,
            stacklevel=2,
        )
        return AnalysisHookContext
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "_EVENT_HOOKS",
    # Events (AnalysisContext is a deprecated alias resolved via __getattr__)
    "AnalysisContext",
    "AnalysisHookContext",
    "ComputeContext",
    "OptimizationContext",
    "clear_hooks",
    "create_analysis_context",
    "create_compute_context",
    "create_optimization_context",
    "execute_hooks",
    "get_event_hooks",
    "list_events",
    "register_event_hook",
]
