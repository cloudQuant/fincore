"""Plugin system for the extensible fincore framework.

Everything user-facing registers into ONE :class:`ExtensionRegistry`
(the process-wide singleton in :mod:`fincore.plugin.registry`):

- custom metrics (resolved by ``AnalysisContext.compute``),
- custom visualization backends (resolved by ``fincore.viz.base.get_backend``),
- event hooks (shared with :mod:`fincore.hooks.events`).

The registry defines the duplicate-registration policy, hook priority,
registration scope, a thread lock, and the :func:`isolated_registry`
test-isolation context manager.
"""

from __future__ import annotations

# Re-export everything from registry (single source of truth)
from fincore.plugin.registry import (
    ExtensionRegistry,
    clear_registry,
    execute_hooks,
    get_metric,
    get_registry,
    get_viz_backend,
    isolated_registry,
    list_hooks,
    list_metrics,
    list_viz_backends,
    register_hook,
    register_metric,
    register_viz_backend,
)
from fincore.plugin.specs import (
    DEFAULT_METRIC_FAMILY,
    ROLLING_FAMILY,
    DuplicatePolicy,
    DuplicateRegistrationError,
    ExtensionKind,
    Registration,
    Scope,
)

__all__ = [
    "DEFAULT_METRIC_FAMILY",
    "ROLLING_FAMILY",
    "DuplicatePolicy",
    "DuplicateRegistrationError",
    "ExtensionKind",
    "ExtensionRegistry",
    "Registration",
    "Scope",
    "clear_registry",
    "execute_hooks",
    "get_metric",
    "get_registry",
    "get_viz_backend",
    "isolated_registry",
    "list_hooks",
    "list_metrics",
    "list_viz_backends",
    "register_hook",
    "register_metric",
    "register_viz_backend",
]
