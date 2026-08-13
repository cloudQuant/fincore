"""Specification types shared by the unified extension registry.

These types define the contract of every registration in
:class:`~fincore.plugin.registry.ExtensionRegistry`: what kind of extension
it is (metric, viz backend, or hook), its name, the registered target, its
priority, its scope, and how duplicate registrations are resolved.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

__all__ = [
    "DEFAULT_METRIC_FAMILY",
    "ROLLING_FAMILY",
    "DuplicatePolicy",
    "DuplicateRegistrationError",
    "ExtensionKind",
    "Registration",
    "Scope",
]

#: Family of user-facing metrics (what ``AnalysisContext.compute`` resolves).
DEFAULT_METRIC_FAMILY = "default"

#: Family of built-in rolling metrics exposed by ``RollingEngine``.
ROLLING_FAMILY = "rolling"


class ExtensionKind(StrEnum):
    """The kind of extension a registration describes."""

    METRIC = "metric"
    VIZ_BACKEND = "viz_backend"
    HOOK = "hook"


class Scope(StrEnum):
    """Lifetime of a registration.

    - ``BUILTIN``: part of fincore itself; survives ``clear_registry()``
      unless ``include_builtins=True`` is passed.
    - ``GLOBAL``: a user registration; persists until explicitly cleared.
    - ``LOCAL``: a test-local registration; rolled back when the enclosing
      :func:`~fincore.plugin.registry.isolated_registry` block exits (in
      fact, isolation rolls back *every* registration made inside the block).
    """

    BUILTIN = "builtin"
    GLOBAL = "global"
    LOCAL = "local"


class DuplicatePolicy(StrEnum):
    """What happens when a name that is already registered is registered again.

    - ``OVERWRITE``: the new target replaces the old one (default).
    - ``ERROR``: raise :class:`DuplicateRegistrationError`.
    - ``IGNORE``: keep the existing registration and discard the new one.
    """

    OVERWRITE = "overwrite"
    ERROR = "error"
    IGNORE = "ignore"


class DuplicateRegistrationError(ValueError):
    """Raised when ``DuplicatePolicy.ERROR`` is in effect and a name is re-registered."""


@dataclass(frozen=True, slots=True)
class Registration:
    """One immutable registry entry.

    Attributes
    ----------
    kind : ExtensionKind
        What the entry is registered as.
    name : str
        Public name of the extension (metric name, backend name, event name).
    target : Any
        The registered callable (function or class).
    family : str
        Metric family; the default family is what user-facing consumers see.
    priority : int
        Hook execution order (ascending); lower runs first.  Ignored for
        metrics and viz backends.
    scope : Scope
        Registration lifetime, see :class:`Scope`.
    """

    kind: ExtensionKind
    name: str
    target: Any
    family: str = DEFAULT_METRIC_FAMILY
    priority: int = 100
    scope: Scope = Scope.GLOBAL
