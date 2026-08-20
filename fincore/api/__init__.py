"""Operation catalog package.

``fincore.api`` exposes the immutable semantic authority that unifies the
enhanced layer: operation definitions (canonical semantics) and public
bindings (where each operation is exposed).  The builtin catalog is a read-only
projection of the frozen registries.
"""

from __future__ import annotations

from fincore.api.builtins import build_builtin_catalog
from fincore.api.catalog import OperationCatalog
from fincore.api.specs import OperationDefinition, PublicBinding

__all__ = [
    "OperationCatalog",
    "OperationDefinition",
    "PublicBinding",
    "build_builtin_catalog",
]
