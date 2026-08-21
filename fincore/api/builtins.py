"""Build the builtin OperationCatalog from registries and explicit enhanced APIs.

This is a read-only projection: ``METRIC_REGISTRY`` and ``WORKFLOW_REGISTRY``
remain the source of truth for their surfaces.  Enhanced APIs that do not have
a legacy registry receive an explicit catalog specification here, so every
public operation remains discoverable by profile, provenance, and stability.
The mapping collapses bindings of the same logical operation into one
``OperationDefinition``.
"""

from __future__ import annotations

from fincore._registry import METRIC_REGISTRY
from fincore.api.catalog import OperationCatalog
from fincore.api.specs import OperationDefinition, PublicBinding
from fincore.contracts.workflows import WORKFLOW_REGISTRY

__all__ = [
    "PERFORMANCE_OPERATION_SPECS",
    "SURFACE_DOMAIN",
    "SURFACE_PATH",
    "VARIANT_PROFILE",
    "build_builtin_catalog",
]

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

# This package is enhanced-only and intentionally not routed through the
# frozen metric registry.  Keep a source-level specification rather than
# importing it: catalog construction must remain light and deterministic.
PERFORMANCE_OPERATION_SPECS = (
    (
        "cashflow_adjusted_returns",
        "fincore.performance.cashflows:cashflow_adjusted_returns",
        "(valuations, cashflows=None, *, fees=None, timing='end', cashflow_timings=None, "
        "fee_treatment='net', cashflow_currency=None, reporting_currency='USD', fx_rates=None)",
        "series",
    ),
    (
        "cashflow_adjusted_twr",
        "fincore.performance.cashflows:cashflow_adjusted_twr",
        "(valuations, cashflows=None, *, fees=None, timing='end', cashflow_timings=None, "
        "fee_treatment='net', cashflow_currency=None, reporting_currency='USD', fx_rates=None)",
        "scalar",
    ),
    ("mwr", "fincore.performance.returns:mwr", "(cashflows, periods=None)", "scalar"),
    ("render_disclosure", "fincore.performance.disclosures:render_disclosure", "(context)", "scalar"),
    (
        "sharpe_confidence_interval",
        "fincore.performance.inference:sharpe_confidence_interval",
        "(returns, risk_free=0.0, *, z=1.96)",
        "legacy_tuple",
    ),
    (
        "sharpe_standard_error",
        "fincore.performance.inference:sharpe_standard_error",
        "(returns, risk_free=0.0)",
        "scalar",
    ),
    ("standard_error_of_mean", "fincore.performance.inference:standard_error_of_mean", "(values)", "scalar"),
    ("twr", "fincore.performance.returns:twr", "(returns)", "scalar"),
    ("xirr", "fincore.performance.returns:xirr", "(cashflows, dates)", "scalar"),
)


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

    for public_name, kernel_ref, signature, result_projection in PERFORMANCE_OPERATION_SPECS:
        definitions.append(
            OperationDefinition(
                operation_id=public_name,
                semantic_profile="enhanced_v1",
                domain="performance",
                canonical_name=public_name,
                kernel_ref=kernel_ref,
                stability="experimental",
                deterministic=True,
                provenance="surface=fincore.performance enhanced-v1",
            )
        )
        bindings.append(
            PublicBinding(
                binding_id=f"performance.{public_name}.enhanced",
                operation_id=public_name,
                semantic_profile="enhanced_v1",
                public_path=f"fincore.performance.{public_name}",
                surface="performance",
                signature=signature,
                adapter_ref="fincore.performance",
                result_projection=result_projection,
                introduced_in="0.4.0.dev0",
            )
        )

    return OperationCatalog(tuple(definitions), tuple(bindings))
