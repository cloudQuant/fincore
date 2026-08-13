"""C0/C1-only facade for the pinned Alphalens performance API."""

from __future__ import annotations

from fincore.alphalens._compat import export_deferred_functions

__all__ = export_deferred_functions(globals(), "performance")

del export_deferred_functions
