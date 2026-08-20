"""Strict compatibility adapter layer.

The ``_compat`` package isolates the strict Empyrical / Pyfolio / Alphalens
execution paths from the enhanced layer.  Strict adapters bypass enhanced
validation and never construct enhanced stateful classes; they call the frozen
raw kernels directly.
"""

from __future__ import annotations

__all__: list[str] = []
