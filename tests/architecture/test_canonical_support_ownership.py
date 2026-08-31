"""Keep canonical domains independent from pre-cutover support packages."""

from __future__ import annotations

import ast
import os
from pathlib import Path

CANONICAL_DOMAINS = (
    "attribution",
    "data",
    "extensions",
    "factor_analysis",
    "metrics",
    "optimization",
    "performance",
    "portfolio",
    "report",
    "risk",
    "simulation",
    "viz",
)
LEGACY_SUPPORT_PREFIXES = (
    "fincore.constants",
    "fincore.contracts",
    "fincore.core",
    "fincore.utils",
)


def _absolute_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imports.add(node.module)
    return imports


def test_canonical_domains_do_not_depend_on_pre_cutover_support_packages() -> None:
    package_root = Path(os.environ.get("FINCORE_0042R2_SOURCE_ROOT", Path(__file__).parents[2])).resolve() / "fincore"
    violations: list[str] = []

    for domain in CANONICAL_DOMAINS:
        for path in sorted((package_root / domain).rglob("*.py")):
            violations.extend(
                f"{path.relative_to(package_root)} -> {target}"
                for target in _absolute_imports(path)
                if target.startswith(LEGACY_SUPPORT_PREFIXES)
            )

    assert not violations, "\\n".join(violations)


def test_shared_time_series_and_rolling_kernels_have_canonical_owners() -> None:
    from fincore.metrics._rolling_moments import roll_alpha_beta_vectorized
    from fincore.runtime.time_series import align_time_series

    assert callable(align_time_series)
    assert callable(roll_alpha_beta_vectorized)
