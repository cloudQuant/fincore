"""Advanced performance attribution module.

Provides Brinson attribution, Fama-French multi-factor models,
style analysis, and timing attribution.
"""

from __future__ import annotations

from fincore.attribution.brinson import brinson_attribution, brinson_results
from fincore.attribution.fama_french import FamaFrenchModel
from fincore.attribution.style import style_analysis, StyleResult

__all__ = [
    "brinson_attribution",
    "brinson_results",
    "FamaFrenchModel",
    "style_analysis",
    "StyleResult",
]
