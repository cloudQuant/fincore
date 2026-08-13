"""C0/C1-only facade for the pinned Alphalens utility API."""

from __future__ import annotations

from fincore.alphalens._compat import export_deferred_functions


class NonMatchingTimezoneError(Exception):
    """Legacy timezone-mismatch exception identity reserved for Task 3."""


class MaxLossExceededError(Exception):
    """Legacy cleaning-loss exception identity reserved for Task 3."""


__all__ = (
    "MaxLossExceededError",
    "NonMatchingTimezoneError",
    *export_deferred_functions(globals(), "utils"),
)

del export_deferred_functions
