#!/usr/bin/env python
"""PR risk classifier for fincore (iteration 0041 / M3).

Maps changed file paths to the R0-R3 levels defined in
``docs/governance/branch-governance.md``. The classification is a *floor*:
multi-path diffs take the highest matching level, maintainers may upgrade
with a structured reason and must never downgrade. Output is advisory for
triage; the ``PR Governance / policy`` workflow consumes it and CI remains
the merge authority.

Exit codes: ``0`` on any successful classification.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import sys
from pathlib import Path, PurePosixPath

R0 = "R0"
R1 = "R1"
R2 = "R2"
R3 = "R3"
LEVELS = (R0, R1, R2, R3)

_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        R3,
        (
            ".github/workflows/",
            ".github/CODEOWNERS",
            ".github/governance/",
            ".github/dependabot.yml",
            ".github/pull_request_template.md",
            "scripts/ci/",
            "pyproject.toml",
            "setup.py",
            "setup.cfg",
            "requirements.txt",
            "requirements-test.txt",
            "constraints/",
            "MANIFEST.in",
            "LICENSE",
            "NOTICE",
            "THIRD_PARTY_NOTICES.md",
            "THIRD_PARTY_LICENSES/",
            "SECURITY.md",
            "GOVERNANCE.md",
            "MAINTAINERS.md",
        ),
    ),
    (
        R2,
        (
            "fincore/empyrical.py",
            "fincore/pyfolio.py",
            "fincore/_dispatch.py",
            "fincore/_registry.py",
            "fincore/core/",
            "fincore/contracts/",
            "fincore/metrics/",
            "fincore/alphalens/",
            "fincore/factor_analysis/",
            "tests/compat/fixtures/",
        ),
    ),
    (
        R0,
        (
            "*.md",
            "*.rst",
            "docs/",
            "mkdocs_docs/",
            "examples/",
            ".github/ISSUE_TEMPLATE/",
            ".gitignore",
            "CITATION.cff",
        ),
    ),
)


def _match(path: str) -> str | None:
    normalized = PurePosixPath(path.strip().replace("\\", "/")).as_posix()
    if not normalized:
        return None
    name = PurePosixPath(normalized).name
    for level, patterns in _RULES:
        for pattern in patterns:
            if pattern.endswith("/"):
                if normalized.startswith(pattern) or normalized == pattern.rstrip("/"):
                    return level
            elif "*" in pattern:
                if fnmatch.fnmatch(name, pattern):
                    return level
            elif normalized == pattern:
                return level
    return None


def classify(paths: list[str], upgrade_to: str | None = None, reason: str | None = None) -> dict:
    matched_rules = []
    highest = None
    for path in paths:
        level = _match(path)
        if level is None:
            continue
        matched_rules.append({"path": path, "level": level})
        if highest is None or LEVELS.index(level) > LEVELS.index(highest):
            highest = level
    if highest is None:
        highest = R1

    result = {
        "risk_level": highest,
        "matched_rules": matched_rules,
        "upgraded": False,
        "reason": reason,
        "warning": "",
    }

    if upgrade_to is not None:
        if upgrade_to not in LEVELS:
            result["warning"] = f"invalid upgrade target {upgrade_to!r}; keeping {highest}"
        elif LEVELS.index(upgrade_to) > LEVELS.index(highest):
            if not (reason or "").strip():
                result["warning"] = "upgrade requires a structured reason; ignoring"
            else:
                result["risk_level"] = upgrade_to
                result["upgraded"] = True
        else:
            result["warning"] = f"downgrade request ignored: {highest} cannot be lowered to {upgrade_to}"
    return result


def _read_paths_file(path: str | None) -> list[str]:
    if not path:
        return []
    return [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--paths", nargs="+", help="changed file paths")
    group.add_argument(
        "--paths-file",
        help="newline-delimited file of changed paths (e.g. from git diff --name-only)",
    )
    parser.add_argument("--upgrade-to", choices=list(LEVELS), help="maintainer override; upgrades only")
    parser.add_argument("--reason", help="structured reason required for upgrades")
    parser.add_argument("--output", help="also write JSON report to this path")
    args = parser.parse_args(argv)

    paths = args.paths if args.paths is not None else _read_paths_file(args.paths_file)
    report = classify(paths, upgrade_to=args.upgrade_to, reason=args.reason)
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())
