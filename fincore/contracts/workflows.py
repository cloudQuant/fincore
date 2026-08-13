"""Registry and lazy resolver for multi-step report workflows."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Literal

WorkflowSurface = Literal["pyfolio_module", "pyfolio_class"]
WorkflowValidationProfile = Literal["legacy_pyfolio", "enhanced"]
WorkflowProjection = Literal["identity", "legacy_none"]


@dataclass(frozen=True)
class WorkflowSpec:
    """One public workflow contract on an independently versioned surface."""

    surface: WorkflowSurface
    public_name: str
    variant: str
    signature_manifest_key: str
    workflow_ref: str
    adapter_ref: str
    validation_profile: WorkflowValidationProfile
    result_contract_key: str
    result_projection: WorkflowProjection


_FACTOR_PARTITIONS = {
    "style": ["momentum", "size", "value", "reversal_short_term", "volatility"],
    "sector": [
        "basic_materials",
        "consumer_cyclical",
        "financial_services",
        "real_estate",
        "consumer_defensive",
        "health_care",
        "utilities",
        "communication_services",
        "energy",
        "industrials",
        "technology",
    ],
}

PYFOLIO_SIGNATURES = {
    "create_bayesian_tear_sheet": (
        "(returns, benchmark_rets=None, live_start_date=None, samples=2000, "
        "run_flask_app=False, stoch_vol=False, progressbar=True)"
    ),
    "create_capacity_tear_sheet": (
        "(returns, positions, transactions, market_data, liquidation_daily_vol_limit=0.2, "
        "trade_daily_vol_limit=0.05, last_n_days=126, days_to_liquidate_limit=1, "
        "estimate_intraday='infer', run_flask_app=False)"
    ),
    "create_full_tear_sheet": (
        "(returns, positions=None, transactions=None, market_data=None, benchmark_rets=None, "
        "slippage=None, live_start_date=None, sector_mappings=None, bayesian=False, round_trips=False, "
        "estimate_intraday='infer', hide_positions=False, cone_std=(1.0, 1.5, 2.0), bootstrap=False, "
        "unadjusted_returns=None, style_factor_panel=None, sectors=None, caps=None, shares_held=None, "
        "volumes=None, percentile=None, turnover_denom='AGB', set_context=True, factor_returns=None, "
        "factor_loadings=None, pos_in_dollars=True, header_rows=None, "
        f"factor_partitions={_FACTOR_PARTITIONS!r})"
    ),
    "create_interesting_times_tear_sheet": ("(returns, benchmark_rets=None, legend_loc='best', run_flask_app=False)"),
    "create_perf_attrib_tear_sheet": (
        "(returns, positions, factor_returns, factor_loadings, transactions=None, pos_in_dollars=True, "
        f"run_flask_app=False, factor_partitions={_FACTOR_PARTITIONS!r})"
    ),
    "create_position_tear_sheet": (
        "(returns, positions, show_and_plot_top_pos=2, hide_positions=False, run_flask_app=False, "
        "sector_mappings=None, transactions=None, estimate_intraday='infer')"
    ),
    "create_returns_tear_sheet": (
        "(returns, positions=None, transactions=None, live_start_date=None, cone_std=(1.0, 1.5, 2.0), "
        "benchmark_rets=None, bootstrap=False, turnover_denom='AGB', header_rows=None, run_flask_app=False)"
    ),
    "create_risk_tear_sheet": (
        "(positions, style_factor_panel=None, sectors=None, caps=None, shares_held=None, volumes=None, "
        "percentile=None, returns=None, transactions=None, estimate_intraday='infer', run_flask_app=False)"
    ),
    "create_round_trip_tear_sheet": (
        "(returns, positions, transactions, sector_mappings=None, estimate_intraday='infer', run_flask_app=False)"
    ),
    "create_simple_tear_sheet": (
        "(returns, positions=None, transactions=None, benchmark_rets=None, slippage=None, "
        "estimate_intraday='infer', live_start_date=None, turnover_denom='AGB', header_rows=None)"
    ),
    "create_txn_tear_sheet": (
        "(returns, positions, transactions, unadjusted_returns=None, estimate_intraday='infer', run_flask_app=False)"
    ),
}
PYFOLIO_SIGNATURE_MANIFEST = {
    f"pyfolio-0.9.6:{name}": (name, signature) for name, signature in PYFOLIO_SIGNATURES.items()
}

WORKFLOW_REGISTRY: dict[tuple[WorkflowSurface, str, str], WorkflowSpec] = {}
for _name in PYFOLIO_SIGNATURES:
    _projection: WorkflowProjection = (
        "legacy_none"
        if _name
        in {
            "create_full_tear_sheet",
            "create_simple_tear_sheet",
        }
        else "identity"
    )
    _spec = WorkflowSpec(
        surface="pyfolio_module",
        public_name=_name,
        variant="strict-0.9.6",
        signature_manifest_key=f"pyfolio-0.9.6:{_name}",
        workflow_ref="fincore._pyfolio_impl:run_workflow",
        adapter_ref="fincore.contracts.workflows:strict_pyfolio_adapter",
        validation_profile="legacy_pyfolio",
        result_contract_key=f"pyfolio-0.9.6:{_name}",
        result_projection=_projection,
    )
    WORKFLOW_REGISTRY[(_spec.surface, _spec.public_name, _spec.variant)] = _spec


def resolve_ref(reference: str) -> Any:
    """Resolve a lazy ``module:attribute`` reference."""

    module_name, attribute = reference.split(":", 1)
    return getattr(importlib.import_module(module_name), attribute)


def strict_pyfolio_adapter(
    workflow: Any,
    spec: WorkflowSpec,
    arguments: dict[str, Any],
) -> Any:
    """Invoke a strict workflow and apply its pinned return projection."""

    from fincore._dispatch import _raw_kernel_execution

    # Pyfolio workflows compose many metric modules.  Keep that complete
    # nested call on the frozen compatibility path rather than allowing an
    # enhanced module wrapper reached deep in the tear sheet to validate it.
    with _raw_kernel_execution():
        result = workflow(spec.public_name, arguments)
    if spec.result_projection == "legacy_none":
        return None
    return result


def invoke_workflow(spec: WorkflowSpec, arguments: dict[str, Any]) -> Any:
    """Resolve and invoke a workflow only at the first public call."""

    try:
        workflow = resolve_ref(spec.workflow_ref)
        adapter = resolve_ref(spec.adapter_ref)
        return adapter(workflow, spec, arguments)
    except ModuleNotFoundError as exc:
        optional_roots = {"IPython", "matplotlib", "pymc", "scipy", "seaborn", "statsmodels"}
        missing_root = (exc.name or "").split(".", 1)[0]
        if missing_root not in optional_roots:
            raise
        extras = "viz,bayesian" if missing_root == "pymc" else "viz"
        raise ImportError(
            f"{spec.public_name} requires the Pyfolio plotting dependencies; "
            f"install them with `pip install fincore[{extras}]`."
        ) from exc


def get_workflow_spec(surface: WorkflowSurface, public_name: str, variant: str) -> WorkflowSpec:
    """Return one exact public workflow contract."""

    return WORKFLOW_REGISTRY[(surface, public_name, variant)]


__all__ = [
    "PYFOLIO_SIGNATURES",
    "PYFOLIO_SIGNATURE_MANIFEST",
    "WORKFLOW_REGISTRY",
    "WorkflowSpec",
    "get_workflow_spec",
    "invoke_workflow",
    "resolve_ref",
    "strict_pyfolio_adapter",
]
