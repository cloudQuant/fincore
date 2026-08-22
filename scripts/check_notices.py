#!/usr/bin/env python3
"""Validate the third-party notice inventory.

Parses the machine-readable JSON block in ``THIRD_PARTY_NOTICES.md`` and fails
when the inventory is missing, malformed, or records an adapted component
without a pinned source commit and an explicit review status.  This is evidence
gathering, not legal self-certification: unresolved human review remains a
release blocker.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tomllib
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parent.parent
NOTICES_PATH = ROOT / "THIRD_PARTY_NOTICES.md"

_JSON_BLOCK_RE = re.compile(r"```json\n(.*?)\n```", re.DOTALL)
_SCHEMA_VERSION = 2
_REQUIRED_COMPONENTS = ("empyrical", "pyfolio", "alphalens", "echarts")
_VALID_REVIEW_STATUSES = ("pending-human-review", "approved")
_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def load_notices(path: str | Path | None = None) -> dict[str, Any]:
    """Parse the JSON inventory block out of the notices Markdown file."""
    target = Path(path) if path is not None else NOTICES_PATH
    text = target.read_text(encoding="utf-8")
    matches = _JSON_BLOCK_RE.findall(text)
    if not matches:
        raise ValueError(f"no machine-readable JSON block found in {target}")
    payload = json.loads(matches[0])
    if not isinstance(payload, dict):
        raise ValueError(f"machine-readable inventory in {target} must be an object")
    return cast("dict[str, Any]", payload)


def _project_version() -> str:
    with (ROOT / "pyproject.toml").open("rb") as file:
        return str(tomllib.load(file)["project"]["version"])


def _validate_project(notices: dict[str, Any], violations: list[str]) -> None:
    if notices.get("schema_version") != _SCHEMA_VERSION:
        violations.append(f"schema_version must be {_SCHEMA_VERSION}")
    project = notices.get("project")
    if not isinstance(project, dict):
        violations.append("project must be an object")
        return
    if project.get("name") != "fincore":
        violations.append("project.name must be fincore")
    if project.get("version") != _project_version():
        violations.append("project.version must equal pyproject.toml")
    if project.get("license") != "MIT":
        violations.append("project.license must be MIT")


def _validate_vendored_asset(component: str, record: dict[str, Any], violations: list[str]) -> None:
    source_reference = record.get("source_reference")
    if not isinstance(source_reference, str) or not source_reference:
        violations.append(f"{component}: vendored asset requires a source_reference")
    source_sha256 = record.get("source_sha256")
    if not isinstance(source_sha256, str) or not _SHA256_RE.match(source_sha256):
        violations.append(f"{component}: source_sha256 must be a 64-hex digest")
    vendored_path = record.get("vendored_path")
    if not isinstance(vendored_path, str) or not vendored_path:
        violations.append(f"{component}: vendored asset requires a vendored_path")
        return
    target = (ROOT / vendored_path).resolve()
    if not target.is_relative_to(ROOT) or not target.is_file():
        violations.append(f"{component}: vendored_path must name a repository file")
        return
    if isinstance(source_sha256, str) and _SHA256_RE.match(source_sha256):
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        if digest != source_sha256:
            violations.append(f"{component}: vendored asset digest does not match source_sha256")
    embedded_attributions = record.get("embedded_attributions")
    if not isinstance(embedded_attributions, list) or not all(
        isinstance(attribution, str) and attribution for attribution in embedded_attributions
    ):
        violations.append(f"{component}: embedded_attributions must be a list of non-empty strings")


def check_notices(notices: dict[str, Any], *, require_approved: bool = False) -> list[str]:
    """Return the list of notice violations (empty means valid).

    When ``require_approved`` is set, every adapted component must carry an
    ``approved`` review status (fail closed for the release profile); ordinary
    PR checks may leave records in ``pending-human-review``.
    """
    violations: list[str] = []
    _validate_project(notices, violations)
    for component in _REQUIRED_COMPONENTS:
        record = notices.get(component)
        if not isinstance(record, dict):
            violations.append(f"missing notice for {component}")
            continue
        if not isinstance(record.get("upstream_version"), str) or not record["upstream_version"]:
            violations.append(f"{component}: upstream_version must be a non-empty string")
        if not isinstance(record.get("license"), str) or not record["license"]:
            violations.append(f"{component}: license must be a non-empty string")
        if not isinstance(record.get("license_header"), str) or not record["license_header"]:
            violations.append(f"{component}: license_header must be a non-empty string")
        adapted = record.get("adapted")
        if not isinstance(adapted, bool):
            violations.append(f"{component}: adapted must be a boolean")
        elif adapted:
            source_commit = record.get("source_commit")
            if not isinstance(source_commit, str) or not _GIT_SHA_RE.match(source_commit):
                violations.append(f"{component}: source_commit must be a 40-hex Git SHA")
        else:
            _validate_vendored_asset(component, record, violations)
        review_status = record.get("review_status")
        if review_status not in _VALID_REVIEW_STATUSES:
            violations.append(f"{component}: review_status must be pending-human-review or approved")
        if require_approved and review_status != "approved":
            violations.append(f"{component}: review_status must be approved (release profile)")
        if review_status == "approved":
            if not isinstance(record.get("reviewer"), str) or not record["reviewer"]:
                violations.append(f"{component}: approved notice requires a reviewer")
            if not record.get("reviewed_at"):
                violations.append(f"{component}: approved notice requires reviewed_at")
    return violations


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notices", default=str(NOTICES_PATH), help="path to THIRD_PARTY_NOTICES.md")
    parser.add_argument(
        "--require-approved",
        action="store_true",
        help="Fail unless every adapted component has an approved review status (release profile).",
    )
    args = parser.parse_args(argv)
    notices = load_notices(args.notices)
    violations = check_notices(notices, require_approved=args.require_approved)
    for violation in violations:
        print(f"FAIL: {violation}", file=sys.stderr)
    if violations:
        return 1
    print("third-party notice inventory is valid.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
