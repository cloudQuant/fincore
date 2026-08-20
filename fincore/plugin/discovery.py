"""Explicit plugin discovery via entry-point groups.

Discovery is *opt-in*: ``import fincore`` never runs third-party code.
``discover_plugins()`` reads distribution metadata only (entry points) and
returns immutable records; loading the referenced object is a separate,
explicit step the caller performs.
"""

from __future__ import annotations

import importlib.metadata
from dataclasses import dataclass

__all__ = [
    "ENTRY_POINT_GROUPS",
    "DiscoveredPlugin",
    "discover_plugins",
    "load_plugin",
]

#: The documented entry-point groups fincore consumes.
ENTRY_POINT_GROUPS = (
    "fincore.metrics",
    "fincore.providers",
    "fincore.renderers",
    "fincore.exporters",
)


@dataclass(frozen=True)
class DiscoveredPlugin:
    """An immutable record of a discovered entry point."""

    name: str
    group: str
    distribution: str
    value: str


def _dist_name(entry_point: importlib.metadata.EntryPoint) -> str:
    dist = getattr(entry_point, "dist", None)
    if dist is not None:
        return str(dist.name)
    return str(entry_point.value.split(":", 1)[0])


def discover_plugins(groups: tuple[str, ...] = ENTRY_POINT_GROUPS) -> tuple[DiscoveredPlugin, ...]:
    """Return discovered plugin entry points without importing them."""
    all_entries = importlib.metadata.entry_points()
    discovered = [
        DiscoveredPlugin(
            name=entry_point.name,
            group=entry_point.group,
            distribution=_dist_name(entry_point),
            value=entry_point.value,
        )
        for group in groups
        for entry_point in all_entries.select(group=group)
    ]
    return tuple(discovered)


def load_plugin(plugin: DiscoveredPlugin) -> object:
    """Import and return the object referenced by a discovered entry point."""
    return importlib.metadata.EntryPoint(
        name=plugin.name,
        value=plugin.value,
        group=plugin.group,
    ).load()
