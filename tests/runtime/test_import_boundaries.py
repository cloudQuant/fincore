"""Architecture checks for the standalone canonical runtime package."""

from __future__ import annotations

import ast
import os
from pathlib import Path

_FORBIDDEN_PREFIXES = (
    "fincore._dispatch",
    "fincore._registry",
    "fincore.alphalens",
    "fincore.api",
    "fincore.core",
    "fincore.empyrical",
    "fincore.pyfolio",
    "fincore.report",
    "fincore.results",
    "fincore.viz",
)
_FORBIDDEN_OPTIONAL_RENDERERS = {"bokeh", "matplotlib", "plotly"}


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.add(node.module)
    return imported


def test_runtime_modules_do_not_depend_on_legacy_facades_domains_or_renderers() -> None:
    runtime_root = (
        Path(os.environ.get("FINCORE_0042R2_SOURCE_ROOT", Path(__file__).parents[2])).resolve() / "fincore" / "runtime"
    )
    violations: dict[str, list[str]] = {}
    for path in sorted(runtime_root.glob("*.py")):
        imports = _imports(path)
        forbidden = sorted(
            imported
            for imported in imports
            if imported.startswith(_FORBIDDEN_PREFIXES)
            or imported.split(".", maxsplit=1)[0] in _FORBIDDEN_OPTIONAL_RENDERERS
        )
        if forbidden:
            violations[str(path.relative_to(runtime_root))] = forbidden

    assert violations == {}


def test_builtin_composition_source_never_scans_loaded_modules() -> None:
    builtins_source = (
        Path(os.environ.get("FINCORE_0042R2_SOURCE_ROOT", Path(__file__).parents[2])).resolve()
        / "fincore"
        / "runtime"
        / "builtins.py"
    ).read_text(encoding="utf-8")

    assert "sys.modules" not in builtins_source
