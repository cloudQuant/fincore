"""Metrics domain namespace.

Leaf functions have one public implementation path under their focused module,
for example ``fincore.metrics.ratios.sharpe_ratio``.  The package namespace
does not install dynamic module aliases, flat functions, or compatibility
surfaces.
"""

from __future__ import annotations

__all__: list[str] = []
