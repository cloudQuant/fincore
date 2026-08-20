#!/usr/bin/env python3
"""Compare two public API snapshots and report breaking changes.

Reads two ``public-api-*.json`` snapshots (produced by
``scripts/snapshot_public_api.py``) and reports added, removed, and changed
public paths.  A removed or re-profiled path is a breaking change.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _surface_map(snapshot: dict) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for surface, data in snapshot.get("surfaces", {}).items():
        for name, entry in data.get("entries", {}).items():
            result[f"{surface}.{name}"] = entry
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("old", help="path to the previous snapshot")
    parser.add_argument("new", help="path to the current snapshot")
    args = parser.parse_args(argv)

    old = _surface_map(json.loads(Path(args.old).read_text(encoding="utf-8")))
    new = _surface_map(json.loads(Path(args.new).read_text(encoding="utf-8")))

    added = sorted(set(new) - set(old))
    removed = sorted(set(old) - set(new))
    changed = sorted(path for path in set(old) & set(new) if old[path] != new[path])

    for path in removed:
        print(f"REMOVED (breaking): {path}")
    for path in changed:
        print(f"CHANGED (potentially breaking): {path}")
    for path in added:
        print(f"ADDED: {path}")

    if removed or changed:
        print(
            f"\n{len(removed)} removed, {len(changed)} changed, {len(added)} added — breaking changes require an ADR."
        )
        return 1
    print(f"{len(added)} added, no removed/changed paths.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
