"""Visualization backend protocol and explicit snapshot resolution.

Defines the :class:`VizBackend` protocol that all visualization backends
must satisfy, a :class:`RenderModel` (the structured inputs passed to a
custom backend's ``render`` method), plus a helper :func:`get_backend` to
resolve a backend by name. Custom backends are supplied through one immutable
extension snapshot rather than a mutable process-wide registry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from fincore.metrics.frequencies import DAILY
from fincore.viz.contracts import VizBackend

if TYPE_CHECKING:
    import pandas as pd

__all__ = ["RenderModel", "VizBackend", "get_backend"]


@dataclass(frozen=True)
class RenderModel:
    """Structured, side-effect-free inputs passed to a custom backend's ``render``.

    Custom viz backends may define ``render(model, **kwargs)`` instead of
    the ``plot_*`` protocol; an explicit caller builds this model and passes
    it through the selected snapshot.
    """

    returns: pd.Series
    factor_returns: pd.Series | None = None
    period: str = DAILY


def get_backend(name: str = "matplotlib", *, extension_snapshot: object | None = None) -> VizBackend:
    """Resolve a visualization backend by name.

    A renderer from ``extension_snapshot`` takes precedence over built-in
    backends. The snapshot protocol is intentionally small: it must provide a
    callable ``renderer(name)`` method, and the returned renderer must be
    instantiable without arguments.

    Parameters
    ----------
    name : str
        Backend identifier.  Built-in backends:
        - ``'matplotlib'``: Static Matplotlib plots (requires matplotlib)
        - ``'html'``: Self-contained HTML reports
        - ``'plotly'``: Interactive Plotly plots (requires plotly)
        - ``'bokeh'``: Interactive Bokeh plots (requires bokeh)

    Returns
    -------
    VizBackend
        An instance satisfying the :class:`VizBackend` protocol.

    Raises
    ------
    ValueError
        If the requested backend is not recognized.
    ImportError
        If the backend's dependencies are not installed.
    """
    name = name.lower().strip()

    if extension_snapshot is not None:
        resolve = getattr(extension_snapshot, "renderer", None)
        if not callable(resolve):
            raise TypeError("extension_snapshot must provide renderer(name)")
        registered = resolve(name)
        if registered is not None:
            backend: Any = registered() if callable(registered) else registered
            return cast("VizBackend", backend)

    if name == "matplotlib":
        from fincore.viz.matplotlib_backend import MatplotlibBackend

        return MatplotlibBackend()

    if name == "html":
        from fincore.viz.html_backend import HtmlReportBuilder

        return HtmlReportBuilder()

    if name == "plotly":
        from fincore.viz.interactive.plotly_backend import PlotlyBackend

        return PlotlyBackend()

    if name == "bokeh":
        from fincore.viz.interactive.bokeh_backend import BokehBackend

        return BokehBackend()

    raise ValueError(f"Unknown viz backend {name!r}. Available: 'matplotlib', 'html', 'plotly', 'bokeh'")
