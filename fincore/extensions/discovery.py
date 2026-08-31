"""Explicit discovery records for extension entry points."""

from __future__ import annotations

import importlib.metadata
from dataclasses import dataclass

__all__ = ["ENTRY_POINT_GROUPS", "DiscoveredExtension", "discover_extensions", "load_extension"]

ENTRY_POINT_GROUPS: tuple[str, ...] = ("fincore.extensions",)


@dataclass(frozen=True, slots=True)
class DiscoveredExtension:
    """One unexecuted extension entry point with distribution provenance."""

    name: str
    group: str
    distribution: str
    value: str


def _distribution_name(entry_point: importlib.metadata.EntryPoint) -> str:
    distribution = getattr(entry_point, "dist", None)
    return str(distribution.name) if distribution is not None else entry_point.value.split(":", 1)[0]


def discover_extensions(groups: tuple[str, ...] = ENTRY_POINT_GROUPS) -> tuple[DiscoveredExtension, ...]:
    """Read matching entry-point metadata without importing third-party code."""

    entries = importlib.metadata.entry_points()
    discovered = (
        DiscoveredExtension(
            name=entry_point.name,
            group=entry_point.group,
            distribution=_distribution_name(entry_point),
            value=entry_point.value,
        )
        for group in groups
        for entry_point in entries.select(group=group)
    )
    return tuple(sorted(discovered, key=lambda item: (item.group, item.name, item.distribution, item.value)))


def load_extension(extension: DiscoveredExtension) -> object:
    """Explicitly load one already-discovered extension object."""

    return importlib.metadata.EntryPoint(
        name=extension.name,
        value=extension.value,
        group=extension.group,
    ).load()
