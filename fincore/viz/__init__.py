"""Visualization backends for fincore.

This package provides a pluggable visualization layer decoupled from
domain computation. Renderers consume explicit domain or report models and
can receive an immutable extension snapshot for custom backends.

Available backends:
- 'html': Self-contained HTML (no external dependencies)
- 'matplotlib': Matplotlib static plots (requires matplotlib)
- 'plotly': Interactive Plotly plots (requires plotly)
- 'bokeh': Interactive Bokeh plots (requires bokeh)
"""

from fincore.viz.base import VizBackend, get_backend

__all__ = ["VizBackend", "get_backend"]
