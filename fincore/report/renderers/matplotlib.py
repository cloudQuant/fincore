"""Matplotlib projection of a report document with explicit ownership."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any

import pandas as pd

from fincore.report.models import ReportDocument
from fincore.runtime import ArtifactBundle
from fincore.runtime.validation import load_optional_module

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = ["render_matplotlib"]


_matplotlib = partial(
    load_optional_module,
    "matplotlib.pyplot",
    dependency="matplotlib",
    extra="visualization",
    message="optional_dependency_missing: matplotlib is required for static report rendering",
)


def render_matplotlib(document: ReportDocument, *, axes: Mapping[str, Any] | None = None) -> ArtifactBundle:
    """Render document series to axes; only newly created figures are owned."""

    if not isinstance(document, ReportDocument):
        raise TypeError("document must be a ReportDocument")
    supplied_axes = dict(axes or {})
    plt = _matplotlib()
    bundle = ArtifactBundle(metadata={"backend": "matplotlib", "report_digest": document.semantic_digest})
    for section in document.sections:
        for name, values in section.series.items():
            key = f"{section.key}.{name}"
            axis = supplied_axes.get(key)
            closer: Any = None
            if axis is None:
                figure, axis = plt.subplots(figsize=(10, 3.5))

                def closer(figure=figure) -> None:
                    plt.close(figure)

                owned = True
            else:
                closer = None
                owned = False
            label = section.legends.get(name, name)
            x_values = values.index if isinstance(values.index, pd.DatetimeIndex) else range(len(values))
            axis.plot(x_values, values.to_numpy(), label=label)
            axis.set_title(f"{section.title}: {label}")
            axis.set_ylabel(section.units.get(name, ""))
            axis.grid(True, alpha=0.3)
            if label:
                axis.legend(loc="best")
            bundle.add(axis, owned=owned, closer=closer, name=f"axis:{key}")
    return bundle
