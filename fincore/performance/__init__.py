"""Performance domain namespace.

Leaf functions and models live at their single implementation paths, such as
``fincore.performance.returns.twr`` and
``fincore.performance.cashflows.cashflow_adjusted_returns``.  This namespace
intentionally does not duplicate them as package-level aliases.
"""

from __future__ import annotations

__all__: list[str] = []
