"""Minimal complete-surface inputs shared by frozen capture contracts."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path


def canonical_sha256(value: object) -> str:
    """Hash JSON with the same canonical encoding as the frozen checker."""
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_provenance() -> dict[str, Any]:
    return {"commit": "a" * 40, "tree": "b" * 40, "clean": True}


def write_minimal_complete_surface_inputs(paths: dict[str, Path]) -> None:
    """Write one valid complete-inventory input trio into a temporary source tree."""
    provenance = _source_provenance()
    legacy_entry = {
        "entry_id": "legacy:metrics.annual_return",
        "source_id": "legacy_manifest",
        "source_kind": "compat_manifest",
        "source_locator": {"artifact_path": "fixtures/legacy.json", "locator": "entries[0]"},
    }
    legacy = {
        "schema_version": 1,
        "artifact_type": "legacy_surface_discovery",
        "discovery_status": "partial",
        "not_for_d0": True,
        "source": provenance,
        "entries": [legacy_entry],
    }
    _write_json(paths["legacy_discovery"], legacy)
    kinds = (
        "public_definition",
        "registry",
        "manifest",
        "documentation",
        "example",
        "benchmark",
        "extra",
        "wheel_content",
    )
    union_entries = [
        {
            "entry_id": f"{kind}:fixtures/{kind}.json",
            "source_kind": kind,
            "source": {
                "artifact_path": f"fixtures/{kind}.json",
                "artifact_sha256": "d" * 64,
                "locator": f"{kind}:{index}",
            },
        }
        for index, kind in enumerate(kinds)
    ]
    union = {
        "schema_version": 1,
        "artifact_type": "surface_union_facts_discovery",
        "discovery_status": "complete",
        "not_for_d0": True,
        "does_not_assert": ["D-TECH", "D0", "installed_wheel_behavior", "legacy_zero"],
        "source_provenance": provenance,
        "wheel": {"filename": "fincore.whl", "sha256": "c" * 64, "member_count": 1},
        "entry_count": len(union_entries),
        "kind_counts": dict.fromkeys(kinds, 1),
        "entries": union_entries,
        "canonical_entries_sha256": canonical_sha256(union_entries),
    }
    _write_json(paths["surface_union"], union)
    supporting_entries = [
        {
            "inventory_entry_id": f"surface_union:{raw_entry['entry_id']}",
            "source_id": "surface_union",
            "raw_entry_id": raw_entry["entry_id"],
            "raw_entry_sha256": canonical_sha256(raw_entry),
            "owner": "platform",
            "disposition": "supporting",
            "capability_ids": ["platform.surface"],
            "target_operation_id": "platform.surface",
            "scenario_ids": ["platform.surface"],
            "source_nodeids": [],
            "wheel_nodeids": [],
            "evidence": {"oracle_reference": "reviewed supporting surface record"},
            "completion_gate": "D0",
            "rationale": "The raw union record is supporting evidence for the complete surface decision.",
            "rule_id": "supporting-surface",
        }
        for raw_entry in union_entries
    ]
    inventory = {
        "schema_version": 1,
        "artifact_type": "complete_surface_inventory",
        "scope": "complete_legacy_surface_union",
        "decision_status": "complete",
        "does_not_assert": ["D-TECH", "D0", "installed_wheel_behavior", "legacy_zero"],
        "owners": ["metrics", "platform"],
        "source_artifacts": [
            {
                "source_id": "legacy_surface_discovery",
                "path": paths["legacy_discovery"].name,
                "sha256": _sha256(paths["legacy_discovery"]),
                "entry_count": 1,
                "source_provenance": provenance,
            },
            {
                "source_id": "surface_union",
                "path": paths["surface_union"].name,
                "sha256": _sha256(paths["surface_union"]),
                "entry_count": len(union_entries),
                "source_provenance": provenance,
            },
        ],
        "entries": [
            {
                "inventory_entry_id": "legacy_surface_discovery:legacy:metrics.annual_return",
                "source_id": "legacy_surface_discovery",
                "raw_entry_id": legacy_entry["entry_id"],
                "raw_entry_sha256": canonical_sha256(legacy_entry),
                "owner": "metrics",
                "disposition": "required",
                "capability_ids": ["metrics.annual_return"],
                "target_operation_id": "metrics.annual_return",
                "scenario_ids": ["ordinary_returns"],
                "source_nodeids": ["tests/legacy/test_metrics.py::test_annual_return"],
                "wheel_nodeids": ["tests/parity/test_metrics.py::test_annual_return"],
                "evidence": {"golden_path": "annual-return.json"},
                "completion_gate": "D-DOMAIN",
                "rationale": "The canonical metrics operation remains a required analytical capability.",
                "rule_id": "unified-operation",
            },
            *supporting_entries,
        ],
    }
    _write_json(paths["inventory"], inventory)
