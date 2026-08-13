"""Unified extension registry for the fincore framework.

A single :class:`ExtensionRegistry` is the one source of truth for every
user-facing extension kind:

- custom metric functions (resolved by ``AnalysisContext.compute``),
- custom visualization backends (resolved by ``fincore.viz.base.get_backend``),
- event hooks (resolved by ``fincore.hooks.events`` and
  ``RollingEngine.available_metrics`` for the built-in rolling family).

The registry defines:

- a **duplicate registration policy** (overwrite by default; ``error`` and
  ``ignore`` are available),
- hook **priority** ordering (ascending; lower runs first),
- registration **scope** (builtin entries survive clearing),
- a **thread lock** (an ``RLock`` guards every mutation and snapshot), and
- a **test-isolation context manager** (:func:`isolated_registry`) that
  snapshots the registry and restores the exact prior state on exit.

Hook return semantics are unified: a hook that returns a value is a
*transform* hook and its result replaces the first positional argument for
the remaining hooks (and is returned to the caller); a hook that returns
``None`` is a *notification* hook and leaves the arguments untouched.
"""

from __future__ import annotations

import functools
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

from fincore.plugin.specs import (
    DEFAULT_METRIC_FAMILY,
    DuplicatePolicy,
    DuplicateRegistrationError,
    ExtensionKind,
    Registration,
    Scope,
)

__all__ = [
    "ExtensionRegistry",
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
    "registry",
]


# =============================================================================
# Registry internals
# =============================================================================


@dataclass(frozen=True)
class _RegistrySnapshot:
    """A deep enough copy of the registry state for isolation rollback."""

    metrics: dict[tuple[str, str], Registration]
    viz_backends: dict[str, Registration]
    hooks: dict[str, tuple[Registration, ...]]


def _is_cleared(entry: Registration, scope: Scope | None, include_builtins: bool) -> bool:
    """Whether ``entry`` should be removed by a clear with these options."""
    return not (entry.scope is Scope.BUILTIN and not include_builtins) and (scope is None or entry.scope is scope)


class ExtensionRegistry:
    """The single thread-safe registry for metrics, viz backends, and hooks.

    Users normally interact with the process-wide singleton ``registry``
    (returned by :func:`get_registry`) through the module-level convenience
    functions.  A fresh instance is useful for tests.
    """

    def __init__(self) -> None:
        self._metrics: dict[tuple[str, str], Registration] = {}
        self._viz_backends: dict[str, Registration] = {}
        self._hooks: dict[str, list[Registration]] = {}
        self._lock = threading.RLock()

    @property
    def hook_storage(self) -> dict[str, list[Registration]]:
        """The live hook storage: event name -> priority-ordered registrations.

        Exposed for :mod:`fincore.hooks.events` so both the plugin API and
        the hooks API share one backing store.  Prefer the public query
        methods unless you own the storage.
        """
        return self._hooks

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        kind: ExtensionKind,
        name: str,
        target: Any,
        *,
        family: str = DEFAULT_METRIC_FAMILY,
        priority: int = 100,
        scope: Scope = Scope.GLOBAL,
        duplicate: DuplicatePolicy = DuplicatePolicy.OVERWRITE,
    ) -> Registration:
        """Register a target under ``kind``/``name`` and return its entry.

        Raises
        ------
        DuplicateRegistrationError
            If ``duplicate=DuplicatePolicy.ERROR`` and the name is taken.
        ValueError
            If ``kind`` is not a supported :class:`ExtensionKind`.
        """
        entry = Registration(
            kind=kind,
            name=name,
            target=target,
            family=family,
            priority=priority,
            scope=scope,
        )
        with self._lock:
            if kind is ExtensionKind.HOOK:
                self._hooks.setdefault(name, [])
                self._hooks[name].append(entry)
                self._hooks[name].sort(key=lambda reg: reg.priority)
                return entry
            if kind is ExtensionKind.METRIC:
                return self._set_entry(self._metrics, (family, name), entry, duplicate)
            if kind is ExtensionKind.VIZ_BACKEND:
                return self._set_entry(self._viz_backends, name, entry, duplicate)
        raise ValueError(f"Unknown extension kind: {kind!r}")

    @staticmethod
    def _set_entry(
        storage: dict[Any, Registration],
        key: Any,
        entry: Registration,
        duplicate: DuplicatePolicy,
    ) -> Registration:
        existing = storage.get(key)
        if existing is not None:
            if duplicate is DuplicatePolicy.ERROR:
                raise DuplicateRegistrationError(f"{entry.kind.value} {entry.name!r} is already registered")
            if duplicate is DuplicatePolicy.IGNORE:
                return existing
        storage[key] = entry
        return entry

    def unregister(
        self,
        kind: ExtensionKind,
        name: str,
        *,
        family: str = DEFAULT_METRIC_FAMILY,
        raise_if_missing: bool = False,
    ) -> Registration | None:
        """Remove a registration and return it (or ``None`` if absent)."""
        with self._lock:
            if kind is ExtensionKind.METRIC:
                removed = self._metrics.pop((family, name), None)
            elif kind is ExtensionKind.VIZ_BACKEND:
                removed = self._viz_backends.pop(name, None)
            elif kind is ExtensionKind.HOOK:
                removed_list = self._hooks.pop(name, None)
                removed = removed_list[-1] if removed_list else None
            else:
                raise ValueError(f"Unknown extension kind: {kind!r}")
        if removed is None and raise_if_missing:
            raise KeyError(f"no {kind.value} named {name!r} is registered")
        return removed

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get(self, kind: ExtensionKind, name: str, *, family: str = DEFAULT_METRIC_FAMILY) -> Registration | None:
        """Return the registration for ``kind``/``name`` (``None`` if absent).

        Hooks are not addressable through ``get``; use :meth:`hooks`.
        """
        with self._lock:
            if kind is ExtensionKind.METRIC:
                return self._metrics.get((family, name))
            if kind is ExtensionKind.VIZ_BACKEND:
                return self._viz_backends.get(name)
        return None

    def metric_names(self, *, family: str = DEFAULT_METRIC_FAMILY) -> frozenset[str]:
        """All registered metric names in a family."""
        with self._lock:
            return frozenset(name for (fam, name) in self._metrics if fam == family)

    def metric_map(self, *, family: str = DEFAULT_METRIC_FAMILY) -> dict[str, Any]:
        """Registered metric name -> target for a family."""
        with self._lock:
            return {name: entry.target for (fam, name), entry in self._metrics.items() if fam == family}

    def viz_backend_map(self) -> dict[str, Any]:
        """Registered viz backend name -> target."""
        with self._lock:
            return {name: entry.target for name, entry in self._viz_backends.items()}

    def hooks(self, event: str) -> list[Registration]:
        """A copy of the priority-ordered registrations for ``event``."""
        with self._lock:
            return list(self._hooks.get(event, []))

    def hook_map(self) -> dict[str, list[Any]]:
        """Event name -> hook targets in priority order (only events with hooks)."""
        with self._lock:
            return {event: [entry.target for entry in entries] for event, entries in self._hooks.items()}

    def execute_hooks(self, event: str, *args: Any, **kwargs: Any) -> Any | None:
        """Run the event's hooks in priority order with unified semantics.

        Transform hooks (non-``None`` return) replace the first positional
        argument for the remaining hooks; notification hooks (``None``
        return) leave the arguments untouched.  Returns the (possibly
        transformed) first argument, or ``None`` when no positional
        arguments were passed.
        """
        hook_args = args
        for entry in self.hooks(event):
            result = entry.target(*hook_args, **kwargs)
            if result is not None and len(hook_args) > 0:
                hook_args = (result, *hook_args[1:])
        if len(hook_args) == 0:
            return None
        return hook_args[0]

    def clear(
        self,
        kind: ExtensionKind | None = None,
        *,
        family: str | None = None,
        name: str | None = None,
        scope: Scope | None = None,
        include_builtins: bool = False,
    ) -> None:
        """Remove registrations.

        Parameters
        ----------
        kind : ExtensionKind, optional
            Restrict to one kind; ``None`` clears every kind.
        family : str, optional
            Restrict metrics to one family.
        name : str, optional
            Restrict to one name (or hook event).
        scope : Scope, optional
            Restrict to one scope; ``None`` clears every non-builtin scope.
        include_builtins : bool
            Whether :class:`Scope.BUILTIN` entries are also removed.
        """
        with self._lock:
            kinds = list(ExtensionKind) if kind is None else [kind]
            if ExtensionKind.METRIC in kinds:
                for metric_key in list(self._metrics):
                    entry = self._metrics[metric_key]
                    if family is not None and metric_key[0] != family:
                        continue
                    if name is not None and metric_key[1] != name:
                        continue
                    if _is_cleared(entry, scope, include_builtins):
                        del self._metrics[metric_key]
            if ExtensionKind.VIZ_BACKEND in kinds:
                for backend_name in list(self._viz_backends):
                    if name is not None and backend_name != name:
                        continue
                    if _is_cleared(self._viz_backends[backend_name], scope, include_builtins):
                        del self._viz_backends[backend_name]
            if ExtensionKind.HOOK in kinds:
                for event, entries in list(self._hooks.items()):
                    if name is not None and event != name:
                        continue
                    kept = [entry for entry in entries if not _is_cleared(entry, scope, include_builtins)]
                    if kept:
                        entries[:] = kept
                    else:
                        del self._hooks[event]

    # ------------------------------------------------------------------
    # Test isolation
    # ------------------------------------------------------------------

    def _snapshot(self) -> _RegistrySnapshot:
        return _RegistrySnapshot(
            metrics=dict(self._metrics),
            viz_backends=dict(self._viz_backends),
            hooks={event: tuple(entries) for event, entries in self._hooks.items()},
        )

    def _restore(self, snapshot: _RegistrySnapshot) -> None:
        self._metrics.clear()
        self._metrics.update(snapshot.metrics)
        self._viz_backends.clear()
        self._viz_backends.update(snapshot.viz_backends)
        self._hooks.clear()
        for event, entries in snapshot.hooks.items():
            self._hooks[event] = list(entries)

    @contextmanager
    def isolated(self) -> Iterator[ExtensionRegistry]:
        """Snapshot the registry, yield, then restore the exact prior state.

        Every registration made inside the block (regardless of scope) is
        rolled back on exit, even when the block raises.
        """
        with self._lock:
            snapshot = self._snapshot()
        try:
            yield self
        finally:
            with self._lock:
                self._restore(snapshot)


# =============================================================================
# Process-wide singleton
# =============================================================================

#: The process-wide extension registry (the single source of truth).
registry = ExtensionRegistry()


def get_registry() -> ExtensionRegistry:
    """Return the process-wide :class:`ExtensionRegistry` singleton."""
    return registry


# =============================================================================
# Decorators (module-level convenience API)
# =============================================================================


def register_metric(
    name: str | None = None,
    *,
    family: str = DEFAULT_METRIC_FAMILY,
    scope: Scope = Scope.GLOBAL,
    duplicate: DuplicatePolicy = DuplicatePolicy.OVERWRITE,
) -> Callable:
    """Decorator to register a custom metric function.

    The decorated function should:
    - Take returns (pd.Series or np.ndarray) as first argument
    - Return a single scalar value

    Registered metrics in the default family are callable through
    :meth:`fincore.core.context.AnalysisContext.compute`.

    Example::
        @register_metric("custom_ratio")
        def custom_ratio(returns, period=252):
            mean = np.mean(returns)
            std = np.std(returns, ddof=1)
            return mean / std * np.sqrt(period)
    """

    def decorator(func: Callable) -> Callable:
        metric_name = name or func.__name__
        registry.register(
            ExtensionKind.METRIC,
            metric_name,
            func,
            family=family,
            scope=scope,
            duplicate=duplicate,
        )

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        wrapper.__name__ = metric_name
        return wrapper

    return decorator


def register_viz_backend(
    name: str,
    *,
    scope: Scope = Scope.GLOBAL,
    duplicate: DuplicatePolicy = DuplicatePolicy.OVERWRITE,
) -> Callable[[type], type]:
    """Decorator to register a custom visualization backend class.

    The registered class must be instantiable without arguments and should
    either implement the :class:`~fincore.viz.base.VizBackend` protocol
    (``plot_*`` methods) or a ``render(model, **kwargs)`` method that
    returns a :class:`~fincore.report.artifacts.ReportArtifacts`.
    """

    def decorator(cls: type) -> type:
        backend_name = name or cls.__name__
        registry.register(
            ExtensionKind.VIZ_BACKEND,
            backend_name,
            cls,
            scope=scope,
            duplicate=duplicate,
        )
        return cls

    return decorator


def register_hook(
    event: str,
    priority: int = 100,
    *,
    scope: Scope = Scope.GLOBAL,
) -> Callable:
    """Decorator to register an event hook.

    Hooks run in ascending priority order.  A hook that returns a value is
    a transform hook: its result replaces the first positional argument for
    the remaining hooks.  A hook that returns ``None`` is a notification
    hook.
    """

    def decorator(func: Callable) -> Callable:
        event_name = event or func.__name__
        registry.register(
            ExtensionKind.HOOK,
            event_name,
            func,
            priority=priority,
            scope=scope,
        )

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        wrapper.__name__ = event_name
        return wrapper

    return decorator


# =============================================================================
# Query functions (module-level convenience API)
# =============================================================================


def list_metrics(*, family: str = DEFAULT_METRIC_FAMILY) -> dict[str, Callable]:
    """All registered metrics in a family as ``{name: function}``."""
    return registry.metric_map(family=family)


def list_viz_backends() -> dict[str, type]:
    """All registered visualization backends as ``{name: class}``."""
    return registry.viz_backend_map()


def list_hooks(event: str | None = None) -> dict[str, list[Callable]]:
    """Registered hooks for an event (in priority order).

    With ``event=None`` returns every event that has at least one hook as
    ``{event: [functions]}``.
    """
    if event is None:
        return registry.hook_map()
    return {event: [entry.target for entry in registry.hooks(event)]}


def get_metric(name: str, *, family: str = DEFAULT_METRIC_FAMILY) -> Callable | None:
    """Get a registered metric function by name (``None`` if absent)."""
    entry = registry.get(ExtensionKind.METRIC, name, family=family)
    return None if entry is None else entry.target


def get_viz_backend(name: str) -> type | None:
    """Get a registered visualization backend class by name (``None`` if absent)."""
    entry = registry.get(ExtensionKind.VIZ_BACKEND, name)
    return None if entry is None else entry.target


def execute_hooks(event: str, *args: Any, **kwargs: Any) -> Any | None:
    """Execute all hooks for an event with unified transform semantics.

    See :meth:`ExtensionRegistry.execute_hooks`.
    """
    return registry.execute_hooks(event, *args, **kwargs)


def clear_registry(
    registry_type: str | None = None,
    *,
    family: str | None = None,
    name: str | None = None,
    scope: Scope | None = None,
    include_builtins: bool = False,
) -> None:
    """Clear user registrations (builtin-scoped entries survive).

    Parameters
    ----------
    registry_type : str, optional
        ``'metrics'``, ``'hooks'``, ``'viz_backends'``, or ``None`` for all.
    include_builtins : bool
        Also remove :class:`Scope.BUILTIN` entries.
    """
    kind_map = {
        "metrics": ExtensionKind.METRIC,
        "hooks": ExtensionKind.HOOK,
        "viz_backends": ExtensionKind.VIZ_BACKEND,
    }
    if registry_type is None:
        kind = None
    else:
        kind = kind_map.get(registry_type)
        if kind is None:
            raise ValueError(f"Unknown registry type {registry_type!r}. Valid: 'metrics', 'hooks', 'viz_backends'")
    registry.clear(
        kind,
        family=family,
        name=name,
        scope=scope,
        include_builtins=include_builtins,
    )


@contextmanager
def isolated_registry() -> Iterator[None]:
    """Test-isolation context manager: snapshot and restore the whole registry."""
    with registry.isolated():
        yield
