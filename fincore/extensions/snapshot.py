"""Immutable extension snapshots for one runtime/catalog lifetime."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Mapping

from fincore.runtime.specs import OperationSpec

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = ["ExtensionHook", "ExtensionSnapshot", "RendererRegistration"]


def _identifier(value: str, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _callable_fingerprint(value: Callable[..., Any]) -> str:
    if not callable(value):
        raise TypeError("extension target must be callable")
    module = getattr(value, "__module__", None)
    qualname = getattr(value, "__qualname__", None)
    if not isinstance(module, str) or not isinstance(qualname, str):
        raise TypeError("extension target must expose __module__ and __qualname__")
    return f"{module}:{qualname}"


@dataclass(frozen=True, slots=True)
class ExtensionHook:
    """One priority-ordered, snapshot-local transform or notification hook."""

    event: str
    callable: Callable[..., Any]
    priority: int = 100
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _identifier(self.event, "event")
        _callable_fingerprint(self.callable)
        if not isinstance(self.priority, int) or isinstance(self.priority, bool):
            raise TypeError("priority must be an int")
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))

    @property
    def fingerprint(self) -> str:
        """Return the stable identity used for sorting and digesting a hook."""

        return _callable_fingerprint(self.callable)


@dataclass(frozen=True, slots=True)
class RendererRegistration:
    """One named renderer/backend attached to a snapshot, not global state."""

    name: str
    renderer: Callable[..., Any]
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _identifier(self.name, "renderer name")
        _callable_fingerprint(self.renderer)
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))

    @property
    def fingerprint(self) -> str:
        """Return the stable renderer identity used by snapshot provenance."""

        return _callable_fingerprint(self.renderer)


@dataclass(frozen=True, slots=True)
class ExtensionSnapshot:
    """A fully immutable collection of extension operations, hooks, and renderers.

    Build a new snapshot with :meth:`with_operation`, :meth:`with_hook`, or
    :meth:`with_renderer`; no call mutates an existing snapshot or process-wide
    singleton. A runtime catalog incorporates this snapshot's digest, allowing
    sessions to pin one exact extension configuration.
    """

    operations: tuple[OperationSpec, ...] = ()
    hooks: tuple[ExtensionHook, ...] = ()
    renderers: tuple[RendererRegistration, ...] = ()
    _hooks_by_event: Mapping[str, tuple[ExtensionHook, ...]] = field(init=False, repr=False, compare=False)
    _renderers_by_name: Mapping[str, RendererRegistration] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        operations = tuple(self.operations)
        hooks = tuple(self.hooks)
        renderers = tuple(self.renderers)
        operation_ids: set[str] = set()
        for operation in operations:
            if not isinstance(operation, OperationSpec):
                raise TypeError("operations must contain OperationSpec instances")
            if not operation.operation_id.startswith("extension.") or operation.operation_id.count(".") < 2:
                raise ValueError("extension operation_id must use the extension namespace")
            if not operation.capability_id.startswith("extension.") or operation.capability_id.count(".") < 2:
                raise ValueError("extension capability_id must use the extension namespace")
            if operation.domain != "extensions":
                raise ValueError("extension operations must use domain='extensions'")
            if operation.operation_id in operation_ids:
                raise ValueError(f"duplicate extension operation_id: {operation.operation_id}")
            operation_ids.add(operation.operation_id)

        hooks_by_event: dict[str, list[ExtensionHook]] = {}
        for hook in hooks:
            if not isinstance(hook, ExtensionHook):
                raise TypeError("hooks must contain ExtensionHook instances")
            hooks_by_event.setdefault(hook.event, []).append(hook)
        ordered_hooks = {
            event: tuple(sorted(hooks_by_event[event], key=lambda item: (item.priority, item.fingerprint)))
            for event in sorted(hooks_by_event)
        }

        renderers_by_name: dict[str, RendererRegistration] = {}
        for renderer in renderers:
            if not isinstance(renderer, RendererRegistration):
                raise TypeError("renderers must contain RendererRegistration instances")
            if renderer.name in renderers_by_name:
                raise ValueError(f"duplicate renderer registration: {renderer.name}")
            renderers_by_name[renderer.name] = renderer

        object.__setattr__(self, "operations", tuple(sorted(operations, key=lambda item: item.operation_id)))
        object.__setattr__(self, "hooks", tuple(hook for entries in ordered_hooks.values() for hook in entries))
        object.__setattr__(self, "renderers", tuple(sorted(renderers_by_name.values(), key=lambda item: item.name)))
        object.__setattr__(self, "_hooks_by_event", MappingProxyType(ordered_hooks))
        object.__setattr__(self, "_renderers_by_name", MappingProxyType(renderers_by_name))

    @property
    def digest(self) -> str:
        """Return a stable content digest excluding process-local callable identities."""

        payload = {
            "operations": [
                {
                    "operation_id": operation.operation_id,
                    "capability_id": operation.capability_id,
                    "domain": operation.domain,
                    "implementation_fingerprint": operation.implementation_fingerprint,
                    "optional_extra": operation.optional_extra,
                    "deterministic": operation.deterministic,
                    "rng_policy": operation.rng_policy,
                    "semantic_mode": operation.semantic_mode,
                    "mode_approval": operation.mode_approval,
                }
                for operation in self.operations
            ],
            "hooks": [
                {"event": hook.event, "fingerprint": hook.fingerprint, "priority": hook.priority} for hook in self.hooks
            ],
            "renderers": [{"name": renderer.name, "fingerprint": renderer.fingerprint} for renderer in self.renderers],
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def with_operation(self, operation: OperationSpec) -> ExtensionSnapshot:
        """Return a new snapshot containing one additional extension operation."""

        return ExtensionSnapshot(operations=(*self.operations, operation), hooks=self.hooks, renderers=self.renderers)

    def with_hook(self, hook: ExtensionHook) -> ExtensionSnapshot:
        """Return a new snapshot containing one additional hook."""

        return ExtensionSnapshot(operations=self.operations, hooks=(*self.hooks, hook), renderers=self.renderers)

    def with_renderer(self, renderer: RendererRegistration) -> ExtensionSnapshot:
        """Return a new snapshot containing one additional renderer."""

        return ExtensionSnapshot(operations=self.operations, hooks=self.hooks, renderers=(*self.renderers, renderer))

    def hooks_for(self, event: str) -> tuple[ExtensionHook, ...]:
        """Return the immutable priority-ordered hooks for one event."""

        return self._hooks_by_event.get(event, ())

    def execute_hooks(self, event: str, *args: Any, **kwargs: Any) -> Any | None:
        """Run snapshot-local hooks with transform semantics for the first argument."""

        if not args:
            for hook in self.hooks_for(event):
                hook.callable(**kwargs)
            return None
        current_args = list(args)
        for hook in self.hooks_for(event):
            transformed = hook.callable(*current_args, **kwargs)
            if transformed is not None:
                current_args[0] = transformed
        return current_args[0]

    def renderer(self, name: str) -> Callable[..., Any] | None:
        """Return one snapshot-local renderer by name, if present."""

        registration = self._renderers_by_name.get(name)
        return None if registration is None else registration.renderer
