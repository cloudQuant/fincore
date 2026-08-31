"""Lazy Plotly and Bokeh projections of report series."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fincore.exceptions import DependencyError
from fincore.runtime import ArtifactBundle

if TYPE_CHECKING:
    from fincore.report.models import ReportDocument

__all__ = ["render_bokeh", "render_plotly"]


def render_plotly(document: ReportDocument) -> ArtifactBundle:
    """Create one Plotly figure from already computed report series."""

    try:
        import plotly.graph_objects as graph_objects
    except ImportError as error:
        raise DependencyError(
            "optional_dependency_missing: plotly is required for interactive report rendering",
            dependency="plotly",
            extra="visualization",
        ) from error
    figure = graph_objects.Figure()
    for section in document.sections:
        for name, values in section.series.items():
            figure.add_scatter(
                x=values.index,
                y=values.to_numpy(),
                mode="lines",
                name=section.legends.get(name, f"{section.title}: {name}"),
            )
    figure.update_layout(title=document.title)
    bundle = ArtifactBundle(metadata={"backend": "plotly", "report_digest": document.semantic_digest})
    bundle.add(figure, owned=False, name="figure")
    return bundle


def render_bokeh(document: ReportDocument) -> ArtifactBundle:
    """Create one Bokeh figure from already computed report series."""

    try:
        from bokeh.plotting import figure
    except ImportError as error:
        raise DependencyError(
            "optional_dependency_missing: bokeh is required for interactive report rendering",
            dependency="bokeh",
            extra="visualization",
        ) from error
    chart = figure(title=document.title, x_axis_type="datetime")
    for section in document.sections:
        for name, values in section.series.items():
            chart.line(
                values.index,
                values.to_numpy(),
                legend_label=section.legends.get(name, f"{section.title}: {name}"),
            )
    bundle = ArtifactBundle(metadata={"backend": "bokeh", "report_digest": document.semantic_digest})
    bundle.add(chart, owned=False, name="figure")
    return bundle
