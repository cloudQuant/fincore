"""Builtin extension-domain operation declarations.

Extensions provide immutable overlay metadata rather than builtin analytical
operations. Third parties add their own namespaced operations through an
``ExtensionSnapshot``; this explicit empty provider keeps the composition root
auditable without creating a second registry.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fincore.runtime import OperationSpec

__all__ = ["operations"]


def operations() -> tuple[OperationSpec, ...]:
    """Return no builtin extension operations; overlays own their declarations."""

    return ()
