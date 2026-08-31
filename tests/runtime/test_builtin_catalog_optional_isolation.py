"""Builtin composition must not import optional report renderer dependencies."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_builtin_catalog_registers_report_compute_without_loading_optional_renderers() -> None:
    root = Path(__file__).parents[2]
    probe = """
import json
import sys
from fincore.runtime.builtins import builtin_catalog

catalog = builtin_catalog()
forbidden = ("bokeh", "matplotlib", "openpyxl", "playwright", "plotly")
print(json.dumps({
    "report_operation": "report.portfolio.build_portfolio_report" in catalog.operation_ids,
    "loaded": sorted(name for name in sys.modules if name.split('.', 1)[0] in forbidden),
}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)

    assert payload == {"report_operation": True, "loaded": []}
