#!/usr/bin/env python3
"""Fail-closed integrity gate for GitHub Actions workflow files.

Checks every ``.github/workflows/*.yml`` for:

1. Duplicate mapping keys (PyYAML's ``safe_load`` silently keeps the last one,
   so this checker walks the compose tree where duplicates are preserved).
2. ``jobs.<id>.needs`` entries that reference a job id that does not exist in
   the same workflow.

Exit code 1 on any violation; 0 otherwise.  This is the local, deterministic
half of the release gate (``actionlint`` runs separately in CI where it is
installed).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import yaml
from yaml.nodes import MappingNode, SequenceNode

ROOT = Path(__file__).resolve().parent.parent
WORKFLOWS_DIR = ROOT / ".github" / "workflows"


def _duplicate_keys(node: object, path: str = "") -> list[str]:
    """Return the dotted paths of duplicate mapping keys in a compose tree."""
    dups: list[str] = []
    if isinstance(node, MappingNode):
        seen: set[str] = set()
        for key_node, value_node in node.value:
            key = str(key_node.value)
            keypath = f"{path}.{key}" if path else key
            if key in seen:
                dups.append(keypath)
            seen.add(key)
            dups.extend(_duplicate_keys(value_node, keypath))
    elif isinstance(node, SequenceNode):
        for i, item in enumerate(node.value):
            dups.extend(_duplicate_keys(item, f"{path}[{i}]"))
    return dups


def _missing_needs(doc: object, workflow_name: str) -> list[str]:
    """Return ``jobs.<id>`` needs entries that reference a missing job id."""
    if not isinstance(doc, dict):
        return []
    jobs = doc.get("jobs") or {}
    if not isinstance(jobs, dict):
        return []
    missing: list[str] = []
    for job_id, job in jobs.items():
        if not isinstance(job, dict):
            continue
        needs = job.get("needs") or []
        if isinstance(needs, str):
            needs = [needs]
        missing.extend(
            f"{workflow_name}: jobs.{job_id}.needs references missing job '{needed}'"
            for needed in needs
            if needed not in jobs
        )
    return missing


def _unsupported_gh_download_options(text: str, workflow_name: str) -> list[str]:
    """Reject GitHub CLI options that cannot select a candidate artifact.

    ``gh run download`` accepts a run id (or an interactive choice), not a
    ``--workflow`` selector.  A release workflow must resolve an exact CI run
    id first; otherwise it either fails at runtime or downloads an unrelated
    candidate.  Join shell continuations before scanning so multiline steps
    are handled the same way as one-line commands.
    """
    normalized = text.replace("\\\n", " ")
    pattern = re.compile(r"\bgh\s+run\s+download\b[^\n]*\s--workflow(?:=|\s)")
    if pattern.search(normalized):
        return [f"{workflow_name}: gh run download does not support --workflow; resolve an exact run id first"]
    return []


def check_workflow(path: Path) -> list[str]:
    """Return violations for a single workflow file (empty means valid)."""
    text = path.read_text(encoding="utf-8")
    violations: list[str] = []

    # 1. Duplicate keys via the compose tree.
    compose_tree = yaml.compose(text)
    if compose_tree is not None:
        violations.extend(f"{path.name}: duplicate mapping key '{dup}'" for dup in _duplicate_keys(compose_tree))

    # 2. Missing needs references.
    doc = yaml.safe_load(text)
    violations.extend(_missing_needs(doc, path.name))

    # 3. Shell-level candidate-artifact misuse that YAML schema validation
    # cannot see.
    violations.extend(_unsupported_gh_download_options(text, path.name))

    return violations


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workflows", default=str(WORKFLOWS_DIR), help="workflow directory")
    args = parser.parse_args(argv)

    workflows_dir = Path(args.workflows)
    files = sorted(workflows_dir.glob("*.yml"))
    if not files:
        print(f"no workflow files found under {workflows_dir}", file=sys.stderr)
        return 1

    all_violations: list[str] = []
    for path in files:
        all_violations.extend(check_workflow(path))

    for violation in all_violations:
        print(f"FAIL: {violation}", file=sys.stderr)
    if all_violations:
        return 1
    print(f"workflow integrity is valid ({len(files)} files).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
