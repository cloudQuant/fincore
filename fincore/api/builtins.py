"""Build the builtin OperationCatalog from the frozen registries.

This is a read-only projection: ``METRIC_REGISTRY`` and ``WORKFLOW_REGISTRY``
remain the source of truth during the migration, and the catalog is rebuilt
from them.  The mapping collapses each registry entry into one ``PublicBinding``
and merges bindings of the same logical operation into one
``OperationDefinition``.
"""

from __future__ import annotations

from fincore._registry import METRIC_REGISTRY
from fincore.api.catalog import OperationCatalog
from fincore.api.specs import OperationDefinition, PublicBinding
from fincore.contracts.workflows import WORKFLOW_REGISTRY

__all__ = ["SURFACE_DOMAIN", "SURFACE_PATH", "VARIANT_PROFILE", "build_builtin_catalog"]

VARIANT_PROFILE = {
    "strict-0.6.0": "strict_empyrical_0_6_0",
    "strict-0.9.6": "strict_pyfolio_0_9_6",
    "stateful-enhanced": "enhanced_v1",
    "enhanced": "enhanced_v1",
    "enhanced-0.3.x": "enhanced_v1",
    "cached-property": "enhanced_v1",
}

SURFACE_PATH = {
    "empyrical_module": "fincore.empyrical",
    "empyrical_class": "fincore.empyrical.Empyrical",
    "metrics": "fincore.metrics",
    "fincore_flat": "fincore",
    "context": "fincore.core.context.AnalysisContext",
    "pyfolio_module": "fincore.pyfolio",
    "pyfolio_class": "fincore.pyfolio.Pyfolio",
}

SURFACE_DOMAIN = {
    "empyrical_module": "metrics",
    "empyrical_class": "metrics",
    "metrics": "metrics",
    "fincore_flat": "metrics",
    "context": "metrics",
    "pyfolio_module": "report",
    "pyfolio_class": "report",
}

_STRICT_PROFILES = frozenset({"strict_empyrical_0_6_0", "strict_pyfolio_0_9_6"})


def _stability(semantic_profile: str) -> str:
    return "stable" if semantic_profile in _STRICT_PROFILES else "experimental"


def build_builtin_catalog() -> OperationCatalog:
    """Build the immutable builtin catalog from the frozen registries."""
    definitions: list[OperationDefinition] = []
    bindings: list[PublicBinding] = []
    seen_definitions: set[tuple[str, str]] = set()

    for (surface, public_name, variant), spec in METRIC_REGISTRY.items():
        profile = VARIANT_PROFILE[variant]
        domain = SURFACE_DOMAIN[surface]
        key = (public_name, profile)
        if key not in seen_definitions:
            definitions.append(
                OperationDefinition(
                    operation_id=public_name,
                    semantic_profile=profile,
                    domain=domain,
                    canonical_name=public_name,
                    kernel_ref=spec.kernel_ref,
                    stability=_stability(profile),
                    deterministic=variant in ("strict-0.6.0",),
                    provenance=f"surface={surface} variant={variant}",
                )
            )
            seen_definitions.add(key)
        bindings.append(
            PublicBinding(
                binding_id=f"{surface}.{public_name}.{variant}",
                operation_id=public_name,
                semantic_profile=profile,
                public_path=f"{SURFACE_PATH[surface]}.{public_name}",
                surface=surface,
                signature=spec.signature_manifest_key or "",
                adapter_ref=spec.adapter_ref,
                result_projection=spec.result_projection,
                introduced_in="0.3.0",
            )
        )

    for (wsurface, wname, wvariant), wspec in WORKFLOW_REGISTRY.items():
        profile = VARIANT_PROFILE[wvariant]
        domain = SURFACE_DOMAIN[wsurface]
        key = (wname, profile)
        if key not in seen_definitions:
            definitions.append(
                OperationDefinition(
                    operation_id=wname,
                    semantic_profile=profile,
                    domain=domain,
                    canonical_name=wname,
                    kernel_ref=wspec.workflow_ref,
                    stability=_stability(profile),
                    deterministic=False,
                    provenance=f"surface={wsurface} variant={wvariant}",
                )
            )
            seen_definitions.add(key)
        bindings.append(
            PublicBinding(
                binding_id=f"{wsurface}.{wname}.{wvariant}",
                operation_id=wname,
                semantic_profile=profile,
                public_path=f"{SURFACE_PATH[wsurface]}.{wname}",
                surface=wsurface,
                signature=wspec.signature_manifest_key or "",
                adapter_ref=wspec.adapter_ref,
                result_projection=wspec.result_projection,
                introduced_in="0.3.0",
            )
        )

    return OperationCatalog(tuple(definitions), tuple(bindings))
