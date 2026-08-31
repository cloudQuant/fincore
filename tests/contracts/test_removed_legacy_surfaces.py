"""The 0.5 source tree must not resolve compatibility-era modules or exports."""

from __future__ import annotations

import json
import subprocess
import sys
import sysconfig
from pathlib import Path

LEGACY_MODULES = (
    "fincore.empyrical",
    "fincore.pyfolio",
    "fincore.alphalens",
    "fincore._registry",
    "fincore._dispatch",
    "fincore._compat",
    "fincore.api",
    "fincore.backends",
    "fincore.capabilities",
    "fincore.constants",
    "fincore.contracts",
    "fincore.core",
    "fincore.hooks",
    "fincore.plugin",
    "fincore.results",
    "fincore.tearsheets",
    "fincore.utils",
    "fincore.validation",
    "fincore._types",
    "fincore.report.artifacts",
    "fincore.report.compute",
    "fincore.report.format",
    "fincore.report.model",
    "fincore.report.provenance",
    "fincore.report.render_html",
    "fincore.report.render_pdf",
)
LEGACY_ROOT_EXPORTS = (
    "Empyrical",
    "Pyfolio",
    "alphalens",
    "analyze",
    "create_strategy_report",
    "sharpe_ratio",
    "cum_returns",
    "max_drawdown",
)


def _probe_source_tree(source_root: Path) -> dict[str, object]:
    base_site = Path(sysconfig.get_paths()["purelib"])
    probe = """
import importlib.util
import json
import sys
from pathlib import Path

source_root = Path(sys.argv[1]).resolve()
base_site = Path(sys.argv[2]).resolve()
sys.path[:0] = [str(source_root), str(base_site)]

import fincore

legacy_modules = tuple(json.loads(sys.argv[3]))
legacy_exports = tuple(json.loads(sys.argv[4]))
print(json.dumps({
    "module_file": str(Path(fincore.__file__).resolve()),
    "root_legacy_exports": [name for name in legacy_exports if hasattr(fincore, name)],
    "legacy_specs": [name for name in legacy_modules if importlib.util.find_spec(name) is not None],
}, sort_keys=True))
"""
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            "-E",
            "-c",
            probe,
            str(source_root),
            str(base_site),
            json.dumps(LEGACY_MODULES),
            json.dumps(LEGACY_ROOT_EXPORTS),
        ],
        cwd=source_root,
        capture_output=True,
        check=False,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_source_tree_has_no_legacy_modules_or_root_aliases() -> None:
    source_root = Path(__file__).parents[2]
    payload = _probe_source_tree(source_root)

    assert Path(str(payload["module_file"])).is_relative_to(source_root)
    assert payload["legacy_specs"] == []
    assert payload["root_legacy_exports"] == []
