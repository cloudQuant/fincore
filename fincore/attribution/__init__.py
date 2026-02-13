"""Advanced performance attribution module.

Provides Brinson attribution, Fama-French multi-factor models,
style analysis, and timing attribution.
"""

from __future__ import annotations

from fincore.attribution.brinson import brinson_attribution, brinson_results
from fincore.attribution.fama_french import FamaFrenchModel
from fincore.attribution.style import (
    StyleResult,
    analyze_performance_by_style,
    calculate_regression_attribution,
    calculate_style_tilts,
    style_analysis,
)

__all__ = [
    "brinson_attribution",
    "brinson_results",
    "FamaFrenchModel",
    "style_analysis",
    "StyleResult",
    "calculate_style_tilts",
    "calculate_regression_attribution",
    "analyze_performance_by_style",
]
