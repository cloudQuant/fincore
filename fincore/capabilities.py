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
        status=STATUS_STABLE,
        input_contract="A returns Series/array and model order parameters (p, q).",
        output_contract="Fitted conditional-volatility model with forecast and conditional VaR.",
        docs_path="api/risk.md",
        rationale="Frozen GARCH/EGARCH/GJR-GARCH kernels with pinned numerical fixtures.",
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
        input_contract="Returns Series (required) plus optional benchmark/positions/transactions/trades.",
        output_contract="An HTML or PDF strategy report at the caller-selected path.",
        docs_path="api/report.md",
        rationale="Compute-once/render-many report pipeline with deterministic sections.",
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
