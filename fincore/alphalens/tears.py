"""C0/C1-only facade for pinned Alphalens tear-sheet entry points."""

from __future__ import annotations

from fincore.alphalens._compat import export_deferred_functions


class GridFigure:
    """Resolve the pinned public constructor without eagerly importing Matplotlib."""

    def __init__(self, rows, cols):
        raise NotImplementedError(
            "Legacy Alphalens symbol 'GridFigure' is available for C0/C1 compatibility, "
            "but its rendering kernel is not implemented yet."
        )


__all__ = ("GridFigure", *export_deferred_functions(globals(), "tears"))

del export_deferred_functions
