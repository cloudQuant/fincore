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
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NOTICES_PATH = ROOT / "THIRD_PARTY_NOTICES.md"

_JSON_BLOCK_RE = re.compile(r"```json\n(.*?)\n```", re.DOTALL)
_REQUIRED_COMPONENTS = ("empyrical", "pyfolio", "alphalens")
_VALID_REVIEW_STATUSES = ("pending-human-review", "approved")
_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def load_notices(path: str | Path | None = None) -> dict:
    """Parse the JSON inventory block out of the notices Markdown file."""
    target = Path(path) if path is not None else NOTICES_PATH
    text = target.read_text(encoding="utf-8")
    matches = _JSON_BLOCK_RE.findall(text)
    if not matches:
        raise ValueError(f"no machine-readable JSON block found in {target}")
    return json.loads(matches[0])


def check_notices(notices: dict, *, require_approved: bool = False) -> list[str]:
    """Return the list of notice violations (empty means valid).

    When ``require_approved`` is set, every adapted component must carry an
    ``approved`` review status (fail closed for the release profile); ordinary
    PR checks may leave records in ``pending-human-review``.
    """
    violations: list[str] = []
    for component in _REQUIRED_COMPONENTS:
        record = notices.get(component)
        if not isinstance(record, dict):
            violations.append(f"missing notice for {component}")
            continue
        source_commit = record.get("source_commit")
        if not isinstance(source_commit, str) or not _GIT_SHA_RE.match(source_commit):
            violations.append(f"{component}: source_commit must be a 40-hex Git SHA")
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
