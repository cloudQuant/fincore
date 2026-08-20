"""Operation catalog entities.

``OperationDefinition`` records *what* an operation means (canonical semantics),
and ``PublicBinding`` records *where* it is exposed.  A logical operation is
defined once and bound onto many public paths, so contract, kernel provenance,
and stability facts are never duplicated per surface.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["OperationDefinition", "PublicBinding"]


@dataclass(frozen=True)
class OperationDefinition:
    """One logical operation with canonical, versioned semantics."""

    operation_id: str
    semantic_profile: str
    domain: str
    canonical_name: str
    aliases: tuple[str, ...] = ()
    stability: str = "experimental"
    input_contract: str = ""
    output_contract: str = ""
    kernel_ref: str = ""
    optional_extra: str | None = None
    deterministic: bool = False
    rng_policy: str = "none"
    provenance: str = ""
    semantic_version: str = "1.0"


@dataclass(frozen=True)
class PublicBinding:
    """A public path bound to exactly one operation under one profile."""

    binding_id: str
    operation_id: str
    semantic_profile: str
    public_path: str
    surface: str
    signature: str = ""
    adapter_ref: str = ""
    result_projection: str = "identity"
    typing_contract_ref: str = ""
    overloads: tuple[str, ...] = ()
    introduced_in: str = "0.3.0"
    deprecated_in: str | None = None
    remove_in: str | None = None
