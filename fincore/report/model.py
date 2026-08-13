"""Structured report model: compute once, render with many backends.

``compute.compute_sections`` returns a :class:`ReportModel`, a dict-compatible
container whose values are restricted to structured data only: scalar metrics,
tables, series, and short text.  Renderers (HTML, PDF, matplotlib, plotly,
bokeh) consume the model and are otherwise side-effect free; they never
compute statistics themselves.

``SectionModel`` provides the typed view used by renderers that want
shape-aware access: numeric metric blocks, DataFrames, Series, and metadata.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

__all__ = ["ReportModel", "SectionModel", "classify_sections"]


def _is_scalar(value: Any) -> bool:
    """Whether a value is a structured scalar (number or short text)."""
    return isinstance(value, (bool, int, float, complex, str, np.generic, np.number)) or value is None


@dataclass(frozen=True)
class SectionModel:
    """Typed, structured view of one report section.

    Contains only structured numbers, tables, and series (plus short
    metadata text).  No figure objects, file handles, or renderer state.
    """

    name: str
    metrics: dict[str, Any] = field(default_factory=dict)
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    series: dict[str, pd.Series] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def is_empty(self) -> bool:
        """Whether the section carries no structured content at all."""
        return not (self.metrics or self.tables or self.series or self.meta)


def _classify_mapping(name: str, values: Mapping[str, Any]) -> SectionModel:
    model = SectionModel(name=name)
    for key, value in values.items():
        if isinstance(value, pd.DataFrame):
            model.tables[key] = value
        elif isinstance(value, pd.Series):
            model.series[key] = value
        elif isinstance(value, Mapping):
            sub = _classify_mapping(f"{name}.{key}", value)
            model.metrics.update({f"{key}.{k}": v for k, v in sub.metrics.items()})
            model.tables.update(sub.tables)
            model.series.update(sub.series)
            model.meta.update(sub.meta)
        elif _is_scalar(value):
            model.metrics[key] = value
        else:  # pragma: no cover - defensive; anything else is not structured
            model.meta[key] = value
    return model


def classify_sections(sections: Mapping[str, Any]) -> dict[str, SectionModel]:
    """Classify raw report entries into typed :class:`SectionModel` views.

    - ``pd.DataFrame`` values become tables,
    - ``pd.Series`` values become series,
    - mappings of scalars (e.g. ``perf_stats``) become metric blocks,
    - scalar / short-text values (``period``, ``summary_text``, ...) become meta.
    """
    models: dict[str, SectionModel] = {}
    for name, value in sections.items():
        if isinstance(value, pd.DataFrame):
            models[name] = SectionModel(name=name, tables={name: value})
        elif isinstance(value, pd.Series):
            models[name] = SectionModel(name=name, series={name: value})
        elif isinstance(value, Mapping):
            models[name] = _classify_mapping(name, value)
        elif _is_scalar(value):
            models[name] = SectionModel(name=name, meta={name: value})
        else:  # tuples like date_range, or unknown containers
            models[name] = SectionModel(name=name, meta={name: value})
    return models


class ReportModel(dict):
    """Dict-compatible result of the report computation stage.

    Subclasses ``dict`` so existing mapping-style consumers (``s["key"]``,
    ``"key" in s``, ``s.get(...)``) keep working unchanged, while adding the
    structured :class:`SectionModel` view for shape-aware renderers and a
    ``title`` carried alongside the data.
    """

    def __init__(self, sections: Mapping[str, Any] | None = None, *, title: str = "Strategy Report") -> None:
        super().__init__(sections or {})
        self.title = title

    @property
    def section_models(self) -> dict[str, SectionModel]:
        """Typed views of every section, classified by value shape."""
        return classify_sections(self)

    def to_dict(self) -> dict[str, Any]:
        """A plain-dict copy of the computed sections."""
        return dict(self)
