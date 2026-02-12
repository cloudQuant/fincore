"""Interactive visualization backends for fincore.

Provides plot-based interactive visualizations using Plotly and Bokeh:
- PlotlyBackend: Web-based interactive plots
- BokehBackend: Bokeh server-compatible plots
"""

from __future__ import annotations

from fincore.viz.interactive.plotly_backend import PlotlyBackend

__all__ = ["PlotlyBackend"]

# Try to import BokehBackend if bokeh is available
try:
    from fincore.viz.interactive.bokeh_backend import BokehBackend

    __all__.append("BokehBackend")
except ImportError:
    pass
