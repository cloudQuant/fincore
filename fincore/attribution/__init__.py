"""Advanced performance attribution module.

Provides Brinson attribution, Fama-French multi-factor models,
style analysis, and timing attribution.

Public capability states are declared in :mod:`fincore.capabilities` and
rendered into ``docs/quality/capability-inventory.md``.  The historical
``BrinsonAttribution.calculate(method="brinson_hood")`` spelling is a verified
alias for the standard Brinson--Hood--Beebower (BHB) decomposition.
"""

from __future__ import annotations

from fincore.attribution.brinson import (
    BrinsonAttribution,
    brinson_attribution,
    brinson_cumulative,
    brinson_results,
)
from fincore.attribution.fama_french import FamaFrenchModel
from fincore.attribution.style import (
    StyleResult,
    analyze_performance_by_style,
    calculate_regression_attribution,
    calculate_style_tilts,
    style_analysis,
)

__all__ = [
    "BrinsonAttribution",
    "FamaFrenchModel",
    "StyleResult",
    "analyze_performance_by_style",
    "brinson_attribution",
    "brinson_cumulative",
    "brinson_results",
    "calculate_regression_attribution",
    "calculate_style_tilts",
    "style_analysis",
]
