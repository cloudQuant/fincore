"""Small, self-consistent repository-surface inputs for capture tests."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_minimal_repository_surface_inputs(source_root: Path) -> dict[str, Path]:
    """Create a valid scoped, explicitly non-D0 facts/disposition pair."""
    source_script = source_root / "scripts" / "check_quality.py"
    source_script.parent.mkdir()
    source_script.write_text("# fixture-owned quality command\n", encoding="utf-8")
    facts_path = source_root / "repository-surface-facts.json"
    disposition_path = source_root / "repository-surface-disposition.json"
    boundaries = {
        "included": ["classified repository paths"],
        "excluded": "runtime execution and installed distributions",
    }
    source_provenance = {"commit": "a" * 40, "tree": "b" * 40, "clean": True}
    source_blob_sha256 = hashlib.sha256(source_script.read_bytes()).hexdigest()
    facts: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": "repository_surface_facts_discovery",
        "discovery_status": "partial",
        "not_for_d0": True,
        "boundaries": boundaries,
        "source_provenance": source_provenance,
        "record_count": 1,
        "records": [
            {
                "path": "scripts/check_quality.py",
                "git_mode": "100644",
                "blob_sha256": source_blob_sha256,
                "kind": "script_candidate",
                "category_tags": ["script_candidate"],
                "token_facts": {"content_kind": "text"},
            }
        ],
    }
    _write_json(facts_path, facts)
    disposition: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": "repository_surface_disposition",
        "scope": "classified_repository_surface_only",
        "decision_status": "scoped",
        "not_for_d0": True,
        "does_not_assert": ["D-TECH", "D0", "installed_wheel_behavior", "legacy_zero"],
        "owners": ["quality"],
        "source_facts": {
            "path": facts_path.name,
            "sha256": hashlib.sha256(facts_path.read_bytes()).hexdigest(),
            "record_count": 1,
            "source_provenance": source_provenance,
        },
        "source_contract": {
            "raw_artifact_type": "repository_surface_facts_discovery",
            "required_raw_status": "partial",
            "required_raw_not_for_d0": True,
            "boundaries_sha256": _canonical_sha256(boundaries),
        },
        "entries": [
            {
                "path": "scripts/check_quality.py",
                "source": {
                    "git_mode": "100644",
                    "blob_sha256": source_blob_sha256,
                    "kind": "script_candidate",
                    "category_tags": ["script_candidate"],
                    "content_kind": "text",
                },
                "owner": "quality",
                "lifecycle": "maintained",
                "disposition": "retain",
                "completion_gate": "D0",
                "target": {
                    "path": "scripts/check_quality.py",
                    "contract_ids": [],
                    "capability_ids": [],
                },
                "legacy_reference_policy": "none",
                "rationale": "Fixture quality command remains maintained.",
                "rule_id": "quality-command",
            }
        ],
        "historical_provenance_allowlist": [],
    }
    _write_json(disposition_path, disposition)
    return {
        "repository_surface_facts": facts_path,
        "repository_surface_disposition": disposition_path,
    }
