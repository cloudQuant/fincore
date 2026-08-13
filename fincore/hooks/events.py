"""Event hooks and execution framework for the fincore plugin system.

This module lets users register callbacks that execute at specific points
during analysis workflows.  Hook storage lives in the single
:class:`~fincore.plugin.registry.ExtensionRegistry` (the plugin registry),
so hooks registered through :func:`fincore.plugin.register_hook` and hooks
registered here share one pipeline and one set of semantics.

Events
------
- "pre_analysis": Before analysis
- "post_analysis": After analysis
- "pre_compute": Before metric computation
- "post_compute": After metric computation
- "optimization": During portfolio optimization

Unified return semantics: a hook that returns a value is a *transform*
hook — its result replaces the first positional argument for the remaining
hooks.  A hook that returns ``None`` is a *notification* hook.

Example
-------
>>> from fincore.hooks import register_event_hook
>>>
>>> def validate_data(returns, **kwargs):
...     # Remove outliers
...     return returns[returns < 3 * returns.std()]
>>>
>>> register_event_hook("pre_analysis", validate_data)
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

from fincore.plugin.registry import registry as _extension_registry
from fincore.plugin.specs import ExtensionKind, Scope

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = [
    "AnalysisContext",  # noqa: F822 -- deprecated alias resolved via module __getattr__
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

#: Canonical event names, in their historical order.
_KNOWN_EVENTS: tuple[str, ...] = (
    "pre_analysis",
    "post_analysis",
    "pre_compute",
    "post_compute",
    "optimization",
)

#: Shared hook storage (event -> priority-ordered registrations), backed by
#: the single extension registry in :mod:`fincore.plugin.registry`.
_EVENT_HOOKS = _extension_registry.hook_storage

#: Deprecated names kept for backward compatibility.
_DEPRECATED_ALIASES: dict[str, str] = {
    "AnalysisContext": "AnalysisHookContext",
}


def __getattr__(name: str) -> Any:
    """Resolve deprecated module attributes lazily, with a warning."""
    if name in _DEPRECATED_ALIASES:
        replacement = _DEPRECATED_ALIASES[name]
        warnings.warn(
            f"{__name__}.{name} is deprecated; use {replacement} instead. "
            f"(The name collided with fincore.core.context.AnalysisContext.)",
            DeprecationWarning,
            stacklevel=2,
        )
        return globals()[replacement]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def register_event_hook(event: str, hook_func: Callable) -> None:
    """Register a hook function for an event.

    Parameters
    ----------
    event : str
        Event name. Valid events: "pre_analysis", "post_analysis",
        "pre_compute", "post_compute", "optimization".
    hook_func : Callable
        Function to call when the event is triggered.

    Raises
    ------
    ValueError
        If the event name is not recognized.

    Examples
    --------
    >>> from fincore.hooks import register_event_hook
    >>>
    >>> def my_validator(returns):
    ...     return returns.dropna()
    >>>
    >>> register_event_hook("pre_analysis", my_validator)
    """
    if event not in _KNOWN_EVENTS:
        raise ValueError(f"Unknown event: {event}. Valid events: {list(_KNOWN_EVENTS)}")
    _extension_registry.register(
        ExtensionKind.HOOK,
        event,
        hook_func,
        priority=100,
        scope=Scope.GLOBAL,
    )


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
        If event is None, returns all hooks as a dict (copies, so callers
        cannot mutate the internal registry).
    """
    if event is None:
        return {name: [entry.target for entry in _extension_registry.hooks(name)] for name in list_events()}
    return [entry.target for entry in _extension_registry.hooks(event)]


def execute_hooks(event: str, *args: Any, **kwargs: Any) -> Any | None:
    """Execute all registered hooks for an event.

    Unified return semantics: each hook's non-``None`` return value
    replaces the first positional argument for the remaining hooks.  The
    (possibly transformed) first argument is returned, or ``None`` when no
    positional arguments were passed.

    Examples
    --------
    >>> from fincore.hooks import execute_hooks
    >>> execute_hooks("pre_analysis", returns)
    """
    return _extension_registry.execute_hooks(event, *args, **kwargs)


def clear_hooks(event: str | None = None) -> None:
    """Clear registered hooks.

    Parameters
    ----------
    event : str, optional
        Event name to clear hooks for. If None, clears all hooks.
    """
    if event is None:
        _extension_registry.clear(ExtensionKind.HOOK, include_builtins=True)
    else:
        _extension_registry.clear(ExtensionKind.HOOK, name=event, include_builtins=True)


def list_events() -> list[str]:
    """List all available event names (known events plus any registered extras)."""
    extras = [event for event in _EVENT_HOOKS if event not in _KNOWN_EVENTS]
    return [*_KNOWN_EVENTS, *extras]


# Convenience classes for context management


class AnalysisHookContext:
    """Context manager for analysis-related hooks.

    ``pre_analysis`` hooks run (and may transform the data) on entry;
    ``post_analysis`` hooks run on exit.

    This class was previously named ``AnalysisContext``; it was renamed to
    avoid colliding with :class:`fincore.core.context.AnalysisContext`.
    The old name remains available as a deprecated alias.

    Examples
    --------
    >>> with AnalysisHookContext(returns):
    ...     # Hooks registered with "pre_analysis" will execute
    ...     pass
    """

    def __init__(self, returns):
        self.returns = returns

    def __enter__(self):
        self.returns = execute_hooks("pre_analysis", self.returns)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.returns = execute_hooks("post_analysis", self.returns)
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
        self.data = execute_hooks("pre_compute", self.data)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.data = execute_hooks("post_compute", self.data)
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
        self.returns = execute_hooks("optimization", self.returns)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


def create_analysis_context(returns):
    """Create an analysis hook context.

    Factory function for creating AnalysisHookContext instances.
    """
    return AnalysisHookContext(returns)


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
