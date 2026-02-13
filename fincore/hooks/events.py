"""Event hooks and execution framework for fincore plugin system.

This module provides a hook system that allows users to register callbacks
that execute at specific points during analysis workflows.

Events
------
- "pre_analysis": Before AnalysisContext computes metrics
- "post_analysis": After AnalysisContext completes analysis
- "pre_compute": Before any metric computation
- "post_compute": After any metric computation
- "optimization": During portfolio optimization

Example
-------
>>> from fincore.hooks import register_hook, execute_hooks
>>>
>>> @register_hook("pre_analysis")
>>> def validate_data(returns, **kwargs):
...     # Remove outliers
...     return returns[returns < 3 * returns.std()]
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

# Event hook registry
# Maps event names to lists of registered hook functions
_EVENT_HOOKS: dict[str, list[Callable]] = {
    "pre_analysis": [],
    "post_analysis": [],
    "pre_compute": [],
    "post_compute": [],
    "optimization": [],
}


def register_event_hook(event: str, hook_func: Callable) -> None:
    """Register a hook function for an event.

    Parameters
    ----------
    event : str
        Event name. Valid events: "pre_analysis", "post_analysis",
        "pre_compute", "post_compute", "optimization".
    hook_func : Callable
        Function to call when event is triggered.

    Raises
    ------
    ValueError
        If event name is not recognized.

    Examples
    --------
    >>> from fincore.hooks import register_event_hook
    >>>
    >>> def my_validator(returns):
    ...     return returns.dropna()
    >>>
    >>> register_event_hook("pre_analysis", my_validator)
    """
    if event not in _EVENT_HOOKS:
        raise ValueError(f"Unknown event: {event}. Valid events: {list(_EVENT_HOOKS.keys())}")
    _EVENT_HOOKS[event].append(hook_func)


def get_event_hooks(event: str | None = None) -> dict[str, list[Callable]] | list[Callable]:
    """Get registered hooks for an event.

    Parameters
    ----------
    event : str, optional
        Event name to filter by. If None, returns all hooks.

    Returns
    -------
    dict or list
        If event is specified, returns the list of hooks for that event.
        If event is None, returns all hooks as a dict.
    """
    if event is None:
        return _EVENT_HOOKS.copy()
    return _EVENT_HOOKS.get(event, []).copy()


def execute_hooks(event: str, *args: Any, **kwargs: Any) -> None:
    """Execute all registered hooks for an event.

    Parameters
    ----------
    event : str
        Event name to execute hooks for.
    *args
        Positional arguments to pass to hook functions.
    **kwargs
        Keyword arguments to pass to hook functions.

    Examples
    --------
    >>> from fincore.hooks import execute_hooks
    >>> execute_hooks("pre_analysis", returns)
    """
    hooks = _EVENT_HOOKS.get(event, [])
    for hook_func in hooks:
        result = hook_func(*args, **kwargs)
        # Allow hooks to modify data by returning new values
        if result is not None and len(args) > 0:
            args = (result,) + args[1:]


def clear_hooks(event: str | None = None) -> None:
    """Clear registered hooks.

    Parameters
    ----------
    event : str, optional
        Event name to clear hooks for. If None, clears all hooks.
    """
    if event is None:
        for key in _EVENT_HOOKS:
            _EVENT_HOOKS[key].clear()
    elif event in _EVENT_HOOKS:
        _EVENT_HOOKS[event].clear()


def list_events() -> list[str]:
    """List all available event names.

    Returns
    -------
    list of str
        Available event names.
    """
    return list(_EVENT_HOOKS.keys())


# Convenience classes for context management


class AnalysisContext:
    """Context manager for analysis-related hooks.

    Examples
    --------
    >>> with AnalysisContext(returns):
    ...     # Hooks registered with "pre_analysis" will execute
    ...     pass
    """

    def __init__(self, returns):
        self.returns = returns

    def __enter__(self):
        execute_hooks("pre_analysis", self.returns)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        execute_hooks("post_analysis", self.returns)
        return False


class ComputeContext:
    """Context manager for computation-related hooks.

    Examples
    --------
    >>> with ComputeContext(data):
    ...     # Hooks registered with "pre_compute" will execute
    ...     result = compute_metric(data)
    ...     # "post_compute" hooks execute on exit
    """

    def __init__(self, data):
        self.data = data

    def __enter__(self):
        execute_hooks("pre_compute", self.data)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        execute_hooks("post_compute", self.data)
        return False


class OptimizationContext:
    """Context manager for optimization-related hooks.

    Examples
    --------
    >>> with OptimizationContext(returns):
    ...     # Hooks registered with "optimization" will execute
    ...     weights = optimize(returns)
    ...     pass
    """

    def __init__(self, returns):
        self.returns = returns

    def __enter__(self):
        execute_hooks("optimization", self.returns)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


def create_analysis_context(returns):
    """Create an analysis context.

    Factory function for creating AnalysisContext instances.
    """
    return AnalysisContext(returns)


def create_compute_context(data):
    """Create a compute context.

    Factory function for creating ComputeContext instances.
    """
    return ComputeContext(data)


def create_optimization_context(returns):
    """Create an optimization context.

    Factory function for creating OptimizationContext instances.
    """
    return OptimizationContext(returns)
