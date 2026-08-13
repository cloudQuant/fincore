"""C0/C1-only facade for the pinned Alphalens plotting API.

No plotting package is imported here.  Actual renderer work belongs to Task 8.
"""

from __future__ import annotations

from fincore.alphalens._compat import export_deferred_functions

__all__ = export_deferred_functions(globals(), "plotting")

del export_deferred_functions
