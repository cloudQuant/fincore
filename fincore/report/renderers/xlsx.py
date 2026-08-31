"""XLSX projection of report sections; spreadsheet support remains lazy."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from fincore.exceptions import DependencyError
from fincore.runtime import ArtifactBundle

if TYPE_CHECKING:
    from fincore.report.models import ReportDocument

__all__ = ["write_xlsx"]


def write_xlsx(document: ReportDocument, target: str | Path) -> ArtifactBundle:
    """Write precomputed metrics/tables/series to an XLSX workbook."""

    try:
        import openpyxl
    except ImportError as error:
        raise DependencyError(
            "optional_dependency_missing: openpyxl is required for XLSX report rendering",
            dependency="openpyxl",
            extra="report-xlsx",
        ) from error
    output = Path(target)
    output.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for section in document.sections:
            sheet = section.key[:31]
            metrics = pd.Series(section.metrics, name="value")
            metrics.to_frame().to_excel(writer, sheet_name=sheet, startrow=0)
            row = len(metrics) + 3
            for name, table in section.tables.items():
                pd.DataFrame({name: []}).to_excel(writer, sheet_name=sheet, startrow=row, index=False)
                table.to_excel(writer, sheet_name=sheet, startrow=row + 1)
                row += len(table) + 4
            for name, values in section.series.items():
                values.to_frame(name=name).to_excel(writer, sheet_name=sheet, startrow=row)
                row += len(values) + 3
    bundle = ArtifactBundle(metadata={"backend": "xlsx", "report_digest": document.semantic_digest})
    bundle.add(output, owned=False, name="file")
    return bundle
