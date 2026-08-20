"""Semantic profile identifiers.

A profile names one frozen behavior contract.  Strict profiles pin upstream
observable behavior; ``enhanced_v1`` is fincore's own versioned enhanced
semantics; ``plugin_v1`` covers extension points.
"""

from __future__ import annotations

STRICT_EMPYRICAL_0_6_0 = "strict_empyrical_0_6_0"
STRICT_PYFOLIO_0_9_6 = "strict_pyfolio_0_9_6"
STRICT_ALPHALENS_CLOUDQUANT_0_4_0 = "strict_alphalens_cloudquant_0_4_0"
ENHANCED_V1 = "enhanced_v1"
PLUGIN_V1 = "plugin_v1"

STRICT_PROFILES = frozenset(
    {
        STRICT_EMPYRICAL_0_6_0,
        STRICT_PYFOLIO_0_9_6,
        STRICT_ALPHALENS_CLOUDQUANT_0_4_0,
    }
)

ALL_PROFILES = frozenset(
    {
        STRICT_EMPYRICAL_0_6_0,
        STRICT_PYFOLIO_0_9_6,
        STRICT_ALPHALENS_CLOUDQUANT_0_4_0,
        ENHANCED_V1,
        PLUGIN_V1,
    }
)

__all__ = [
    "ALL_PROFILES",
    "ENHANCED_V1",
    "PLUGIN_V1",
    "STRICT_ALPHALENS_CLOUDQUANT_0_4_0",
    "STRICT_EMPYRICAL_0_6_0",
    "STRICT_PROFILES",
    "STRICT_PYFOLIO_0_9_6",
]
