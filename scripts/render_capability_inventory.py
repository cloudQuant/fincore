#!/usr/bin/env python3
"""Render the capability inventory Markdown from the declarative registry.

The checked-in file is generated; never hand-edit it.  ``--check`` fails when
the file does not match the registry, and a registry row with an empty
``docs_path`` is rejected so undocumented public surfaces cannot ship.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from fincore.capabilities import list_capabilities

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "docs" / "quality" / "capability-inventory.md"


def render() -> str:
    capabilities = list_capabilities()
    undocumented = [cap.id for cap in capabilities if not cap.docs_path]
    if undocumented:
        raise ValueError(f"undocumented public capability rows: {', '.join(undocumented)}")
    lines = [
        "# Capability Inventory",
        "",
        "> Machine-generated from `fincore.capabilities`. Do not edit by hand.",
        "",
        "| id | domain | status | public path | input contract | output contract | docs |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    lines.extend(
        f"| `{cap.id}` | {cap.domain} | `{cap.status}` | `{cap.public_path}` | "
        f"{cap.input_contract} | {cap.output_contract} | `{cap.docs_path}` |"
        for cap in capabilities
    )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if the checked-in inventory is stale or undocumented.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output Markdown path.")
    args = parser.parse_args(argv)
    rendered = render()
    if args.check:
        current = args.output.read_text(encoding="utf-8") if args.output.exists() else ""
        if current != rendered:
            print(f"FAIL: {args.output.name} is stale; regenerate it with {Path(__file__).name}")
            return 1
        print("OK: capability inventory is up to date.")
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
