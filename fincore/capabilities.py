"""Machine-readable capability inventory for fincore.

A single declarative registry distinguishes the four actionable public states
(stable, experimental, provider_required, not_implemented) so users can tell
what is safe to depend on without reading source or triggering an exception.

The registry is import-light (no heavy optional dependencies are imported at
module load) and rendered into ``docs/quality/capability-inventory.md`` by
``scripts/render_capability_inventory.py``.  Every row must carry a non-empty
``docs_path``; the renderer rejects undocumented public rows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

STATUSES: tuple[str, ...] = ("stable", "experimental", "provider_required", "not_implemented")

STATUS_STABLE = "stable"
STATUS_EXPERIMENTAL = "experimental"
STATUS_PROVIDER_REQUIRED = "provider_required"
STATUS_NOT_IMPLEMENTED = "not_implemented"


@dataclass(frozen=True)
class Capability:
    """An immutable, documented public capability."""

    id: str
    public_path: str
    domain: str
    status: str
    input_contract: str
    output_contract: str
    docs_path: str
    rationale: str


_CAPABILITIES: tuple[Capability, ...] = (
    # --- risk: EVT / GARCH -------------------------------------------------
    Capability(
        id="risk.evt",
        public_path="fincore.risk.evt",
        domain="risk",
        status=STATUS_STABLE,
        input_contract="A returns Series/array (excess or simple) with optional tail probability.",
        output_contract="Tail-risk estimates (VaR/CVaR) with the fitted EVT parameters attached.",
        docs_path="api/risk.md",
        rationale="Frozen EVT tail-risk kernels with pinned numerical fixtures.",
    ),
    Capability(
        id="risk.garch",
        public_path="fincore.risk.garch",
        domain="risk",
        status=STATUS_EXPERIMENTAL,
        input_contract="A returns Series/array and model order parameters (p, q) — only (1, 1) is implemented.",
        output_contract="Fitted conditional-volatility model with forecast and conditional VaR/ES.",
        docs_path="api/risk.md",
        rationale="GARCH/EGARCH/GJR-GARCH (1,1) kernels with corrected forecast recursions and convergence checks; higher orders unsupported until fully verified.",
    ),
    Capability(
        id="risk.walk_forward_validation",
        public_path="fincore.risk.build_risk_validation_report",
        domain="risk",
        status=STATUS_EXPERIMENTAL,
        input_contract=(
            "A RiskModelSpec plus finite, sorted returns passed through walk_forward_var; only one-step, "
            "lower-tail VaR is currently supported."
        ),
        output_contract=(
            "A JSON-serializable event ledger containing every out-of-sample forecast, exception, refit, "
            "fit parameters, and input/backtest digests."
        ),
        docs_path="guide/risk-validation.md",
        rationale=(
            "The walk-forward VaR boundary has independent calibration and no-look-ahead checks, but it is "
            "experimental until broader risk-model validation and release gates are complete."
        ),
    ),
    # --- strict compatibility façades --------------------------------------
    Capability(
        id="compat.empyrical",
        public_path="fincore.empyrical",
        domain="compat",
        status=STATUS_STABLE,
        input_contract="empyrical 0.6.0 call signatures and return shapes.",
        output_contract="Numerically verified empyrical 0.6.0 results (C0-C3 gates).",
        docs_path="development/compatibility.md",
        rationale="Frozen empyrical 0.6.0 surface pinned by tests/compat fixtures.",
    ),
    Capability(
        id="compat.pyfolio",
        public_path="fincore.pyfolio",
        domain="compat",
        status=STATUS_STABLE,
        input_contract="pyfolio 0.9.6 tear-sheet workflow signatures.",
        output_contract="pyfolio 0.9.6-profile tear-sheet workflows (C1/C4 gates).",
        docs_path="development/compatibility.md",
        rationale="Frozen pyfolio 0.9.6 profile behind the fincore[pyfolio] extra.",
    ),
    Capability(
        id="compat.alphalens",
        public_path="fincore.alphalens",
        domain="compat",
        status=STATUS_EXPERIMENTAL,
        input_contract="alphalens 0.4.0 source-shaped call signatures.",
        output_contract="Source-shaped alphalens strict façade (Beta integration).",
        docs_path="development/compatibility.md",
        rationale="Beta alphalens migration; use the tested APIs in the migration guide.",
    ),
    # --- attribution -------------------------------------------------------
    Capability(
        id="attribution.brinson",
        public_path="fincore.attribution.brinson_attribution",
        domain="attribution",
        status=STATUS_STABLE,
        input_contract="Portfolio/benchmark returns and weights of matching shape.",
        output_contract="Allocation, selection and interaction effects summing to active return.",
        docs_path="api/attribution.md",
        rationale="BHB decomposition with residual accounting; numerically verified.",
    ),
    Capability(
        id="attribution.brinson_hood",
        public_path="fincore.attribution.BrinsonAttribution.calculate",
        domain="attribution",
        status=STATUS_NOT_IMPLEMENTED,
        input_contract="(unavailable) would be portfolio/benchmark returns and weights.",
        output_contract="(unavailable) would be Brinson-Hood-Faber attribution by period.",
        docs_path="api/attribution.md",
        rationale="Public option that raises NotImplementedError until a verified implementation ships.",
    ),
    Capability(
        id="attribution.fama_french_model",
        public_path="fincore.attribution.FamaFrenchModel",
        domain="attribution",
        status=STATUS_STABLE,
        input_contract="Asset/portfolio returns plus factor returns (3/4/5 factor).",
        output_contract="Fitted factor exposures, idiosyncratic risk and R-squared.",
        docs_path="api/attribution.md",
        rationale="Local multi-factor estimation; no network access required for the model itself.",
    ),
    Capability(
        id="attribution.ff_factor_provider",
        public_path="fincore.attribution.fetch_ff_factors",
        domain="attribution",
        status=STATUS_PROVIDER_REQUIRED,
        input_contract="Factor names and a date range; an injected provider is required.",
        output_contract="Fama-French factor returns for the requested interval.",
        docs_path="api/attribution.md",
        rationale="Requires an external provider; no default network fetcher is bundled.",
    ),
    Capability(
        id="attribution.style_analysis",
        public_path="fincore.attribution.style_analysis",
        domain="attribution",
        status=STATUS_STABLE,
        input_contract="Portfolio returns and optional style factor returns.",
        output_contract="Style tilts and regression-based attribution.",
        docs_path="api/attribution.md",
        rationale="Constrained style regression with documented coefficient semantics.",
    ),
    Capability(
        id="attribution.style_factor_provider",
        public_path="fincore.attribution.fetch_style_factors",
        domain="attribution",
        status=STATUS_PROVIDER_REQUIRED,
        input_contract="Style factor identifiers and a date range; an injected provider is required.",
        output_contract="Style factor returns for the requested interval.",
        docs_path="api/attribution.md",
        rationale="Requires an external provider; no default network fetcher is bundled.",
    ),
    # --- data providers ----------------------------------------------------
    Capability(
        id="data.yahoo",
        public_path="fincore.data.YahooFinanceProvider",
        domain="data",
        status=STATUS_PROVIDER_REQUIRED,
        input_contract="Symbol(s), date range; yfinance extra and a working transport.",
        output_contract="Price history DataFrame with the provider's price-adjustment convention.",
        docs_path="api/fincore.md",
        rationale="Requires fincore[data-yahoo] and a reachable Yahoo Finance service.",
    ),
    Capability(
        id="data.alphavantage",
        public_path="fincore.data.AlphaVantageProvider",
        domain="data",
        status=STATUS_PROVIDER_REQUIRED,
        input_contract="Symbol(s), date range and an API key; requests extra.",
        output_contract="Price history DataFrame.",
        docs_path="api/fincore.md",
        rationale="Requires fincore[data-alphavantage] and an API key.",
    ),
    Capability(
        id="data.tushare",
        public_path="fincore.data.TushareProvider",
        domain="data",
        status=STATUS_PROVIDER_REQUIRED,
        input_contract="Symbol(s), date range and a Tushare token.",
        output_contract="Chinese A-share price history DataFrame.",
        docs_path="api/fincore.md",
        rationale="Requires fincore[data-cn] and a Tushare token.",
    ),
    Capability(
        id="data.akshare",
        public_path="fincore.data.AkShareProvider",
        domain="data",
        status=STATUS_PROVIDER_REQUIRED,
        input_contract="Symbol(s) and date range; akshare extra.",
        output_contract="Chinese market price history DataFrame.",
        docs_path="api/fincore.md",
        rationale="Requires fincore[data-cn] and a working akshare install.",
    ),
    # --- report ------------------------------------------------------------
    Capability(
        id="report.strategy_report",
        public_path="fincore.report.create_strategy_report",
        domain="report",
        status=STATUS_STABLE,
        input_contract=(
            "Validated periodic returns plus optional benchmark/positions/transactions/trades; "
            "an enhanced DisclosureContext declares cashflow, fee and unit semantics."
        ),
        output_contract=(
            "An HTML or PDF strategy report with calculation, units, sample and data-quality disclosure; "
            "optional provenance sidecar."
        ),
        docs_path="api/report.md",
        rationale="Compute-once/render-many report pipeline with deterministic sections and explicit performance context.",
    ),
    # --- factor analysis ---------------------------------------------------
    Capability(
        id="factor_analysis.prepare",
        public_path="fincore.factor_analysis.prepare_factor_data",
        domain="factor_analysis",
        status=STATUS_EXPERIMENTAL,
        input_contract="Factor, prices and optional groupby/quantile configuration.",
        output_contract="A PreparedFactorData container with documented loss accounting.",
        docs_path="api/factor-analysis.md",
        rationale="Enhanced prepare/analyze/render workflow; Beta integration.",
    ),
    Capability(
        id="factor_analysis.prepare_by_horizon",
        public_path="fincore.factor_analysis.prepare_factor_data_by_horizon",
        domain="factor_analysis",
        status=STATUS_EXPERIMENTAL,
        input_contract=(
            "Factor, prices, and unique forward periods; full-sample filter_zscore is rejected on this causal "
            "enhanced route."
        ),
        output_contract=(
            "A MultiHorizonPreparedFactorData mapping whose per-period PreparedFactorData and loss report retain "
            "observations available for that horizon."
        ),
        docs_path="concepts/factor-research-protocol.md",
        rationale=(
            "Per-horizon availability prevents long-horizon missing returns from removing short-horizon evidence, "
            "but costs, capacity, corporate-action/calendar provenance, and full workflow integration remain unsealed."
        ),
    ),
    Capability(
        id="factor_analysis.costs",
        public_path="fincore.factor_analysis.apply_factor_costs",
        domain="factor_analysis",
        status=STATUS_EXPERIMENTAL,
        input_contract=(
            "Gross-normalized enhanced factor weights, same-currency dollar-volume panel, simple gross returns, "
            "an explicit FactorCostModel, and complete borrow ledgers whenever weights are short."
        ),
        output_contract=(
            "An immutable gross-to-net cost ledger with entry/rebalance trade weights, spread/impact/borrow costs, "
            "participation, and the binding hard capacity inequality."
        ),
        docs_path="concepts/factor-research-protocol.md",
        rationale=(
            "A labelled arithmetic accounting boundary with independent reconciliation fixtures; it is not an "
            "execution simulator, calibrated market-impact model, or complete research-trial workflow."
        ),
    ),
    Capability(
        id="factor_analysis.pit_prepare",
        public_path="fincore.factor_analysis.prepare_pit_factor_data",
        domain="factor_analysis",
        status=STATUS_EXPERIMENTAL,
        input_contract=(
            "A point-in-time factor ledger with asset/as_of/known_at/effective_from/value/in_universe, "
            "prices, and sorted evaluation dates."
        ),
        output_contract=(
            "PreparedFactorData from revisions known and effective at each evaluation date; "
            "full-sample filter_zscore is rejected."
        ),
        docs_path="concepts/factor-research-protocol.md",
        rationale=(
            "Causal PIT materialization is independently timeline-tested, but corporate-action/calendar provenance, "
            "liquidity/borrow provenance, execution calibration, and complete workflow integration remain unsealed."
        ),
    ),
    Capability(
        id="factor_analysis.inference",
        public_path="fincore.factor_analysis.factor_model_inference",
        domain="factor_analysis",
        status=STATUS_EXPERIMENTAL,
        input_contract=(
            "An enhanced FactorAnalysisModel plus an FDR alpha; uses its stored aggregate date-by-period IC snapshot."
        ),
        output_contract=(
            "ICInferenceResult with sample counts, two-sided Student-t p-values, BH q-values, "
            "discoveries, and an explicit untestable-period marker."
        ),
        docs_path="concepts/factor-research-protocol.md",
        rationale=(
            "Statsmodels/SciPy-oracle-tested IC/FDR post-analysis; IC tests remain i.i.d., while the separate "
            "Fama-MacBeth helper supports Newey-West. Trial registry, clustered inference, and report integration remain incomplete."
        ),
    ),
    Capability(
        id="factor_analysis.fama_macbeth",
        public_path="fincore.factor_analysis.fama_macbeth",
        domain="factor_analysis",
        status=STATUS_EXPERIMENTAL,
        input_contract=(
            "Labelled return and exposure panels with at least two assets; optional Newey-West requires a "
            "chronologically ordered returns index and an explicit lag count."
        ),
        output_contract=(
            "Cross-sectional intercept/exposure means, standard errors and t-statistics; DataFrame attrs disclose "
            "the i.i.d. or Newey-West covariance profile, lags, and fitted cross-section count."
        ),
        docs_path="concepts/factor-research-protocol.md",
        rationale=(
            "Asset-label alignment and Bartlett Newey-West standard errors are independently statsmodels-tested, "
            "but multi-factor, clustered, trial-registry, and report integration remain incomplete."
        ),
    ),
    Capability(
        id="factor_analysis.analyze",
        public_path="fincore.factor_analysis.analyze_factor",
        domain="factor_analysis",
        status=STATUS_EXPERIMENTAL,
        input_contract="A PreparedFactorData container.",
        output_contract="FactorAnalysisModel with IC, quantile returns and turnover.",
        docs_path="api/factor-analysis.md",
        rationale="Enhanced compute-only factor analysis; Beta integration.",
    ),
    Capability(
        id="factor_analysis.render",
        public_path="fincore.factor_analysis.tears",
        domain="factor_analysis",
        status=STATUS_EXPERIMENTAL,
        input_contract="A FactorAnalysisModel or EventAnalysisModel.",
        output_contract="Rendered tear-sheet figures with explicit close_owned_figures ownership.",
        docs_path="api/factor-analysis.md",
        rationale="Enhanced rendering; requires fincore[alphalens] extra.",
    ),
)


def list_capabilities() -> list[Capability]:
    """Return every registered capability in declaration order."""
    return list(_CAPABILITIES)


def get_capability(capability_id: str) -> Capability:
    """Return the capability with the given id, or raise KeyError."""
    for capability in _CAPABILITIES:
        if capability.id == capability_id:
            return capability
    raise KeyError(f"unknown capability id: {capability_id}")


__all__ = [
    "STATUSES",
    "STATUS_EXPERIMENTAL",
    "STATUS_NOT_IMPLEMENTED",
    "STATUS_PROVIDER_REQUIRED",
    "STATUS_STABLE",
    "Capability",
    "get_capability",
    "list_capabilities",
]
