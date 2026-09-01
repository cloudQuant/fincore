"""Fail-closed contracts for a complete 0042-R2 surface inventory."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from typing import Any


REPOSITORY_ROOT = Path(os.environ.get("FINCORE_0042R2_SOURCE_ROOT", Path(__file__).parents[2])).resolve()
SCRIPT = REPOSITORY_ROOT / "scripts" / "check_0042_r2_complete_surface_inventory.py"


def _load_checker() -> Any:
    module_name = "fincore_0042_r2_complete_surface_inventory_test"
    specification = importlib.util.spec_from_file_location(module_name, SCRIPT)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    specification.loader.exec_module(module)
    return module


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _provenance() -> dict[str, object]:
    return {"commit": "a" * 40, "tree": "b" * 40, "clean": True}


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _legacy_discovery() -> dict[str, object]:
    return {
        "schema_version": 1,
        "artifact_type": "legacy_surface_discovery",
        "discovery_status": "partial",
        "not_for_d0": True,
        "source": _provenance(),
        "entries": [
            {
                "entry_id": "legacy:metrics.annual_return",
                "source_id": "legacy_manifest",
                "source_kind": "compat_manifest",
                "source_locator": {"artifact_path": "fixtures/legacy.json", "locator": "entries[0]"},
            }
        ],
    }


def _surface_union() -> dict[str, object]:
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
    entries = [
        {
            "entry_id": (
                "public_definition:fincore/metrics.py:function:annual_return:1"
                if kind == "public_definition"
                else f"{kind}:fixtures/{kind}.json"
            ),
            "source_kind": kind,
            "source": {
                "artifact_path": "fincore/metrics.py",
                "artifact_sha256": "d" * 64,
                "locator": f"{kind}:{index}",
            },
        }
        for index, kind in enumerate(kinds)
    ]
    return {
        "schema_version": 1,
        "artifact_type": "surface_union_facts_discovery",
        "discovery_status": "complete",
        "not_for_d0": True,
        "does_not_assert": ["D-TECH", "D0", "installed_wheel_behavior", "legacy_zero"],
        "source_provenance": _provenance(),
        "wheel": {"filename": "fincore.whl", "sha256": "c" * 64, "member_count": 1},
        "entry_count": len(entries),
        "kind_counts": dict.fromkeys(kinds, 1),
        "entries": entries,
        "canonical_entries_sha256": _canonical_sha256(entries),
    }


def _source_record(path: Path, artifact: dict[str, object], source_id: str) -> dict[str, object]:
    provenance = artifact.get("source") or artifact.get("source_provenance")
    assert isinstance(provenance, dict)
    entries = artifact["entries"]
    assert isinstance(entries, list)
    return {
        "source_id": source_id,
        "path": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "entry_count": len(entries),
        "source_provenance": provenance,
    }


def _inventory(legacy_path: Path, union_path: Path) -> dict[str, object]:
    legacy = _legacy_discovery()
    union = _surface_union()
    legacy_entry = legacy["entries"][0]
    assert isinstance(legacy_entry, dict)
    checker = _load_checker()
    union_entries = union["entries"]
    assert isinstance(union_entries, list)
    supporting_entries = [
        {
            "inventory_entry_id": f"surface_union:{union_entry['entry_id']}",
            "source_id": "surface_union",
            "raw_entry_id": union_entry["entry_id"],
            "raw_entry_sha256": checker.canonical_sha256(union_entry),
            "owner": "platform",
            "disposition": "supporting",
            "capability_ids": ["metrics.annual_return"],
            "target_operation_id": "metrics.annual_return",
            "scenario_ids": ["ordinary_returns"],
            "source_nodeids": [],
            "wheel_nodeids": [],
            "evidence": {"oracle_reference": "same capability binding"},
            "completion_gate": "D-CUTOVER",
            "rationale": "The source surface is absorbed into the canonical metrics operation.",
            "rule_id": "canonical-source-rewrite",
        }
        for union_entry in union_entries
        if isinstance(union_entry, dict)
    ]
    return {
        "schema_version": 1,
        "artifact_type": "complete_surface_inventory",
        "scope": "complete_legacy_surface_union",
        "decision_status": "complete",
        "does_not_assert": ["D-TECH", "D0", "installed_wheel_behavior", "legacy_zero"],
        "owners": ["metrics", "platform"],
        "source_artifacts": [
            _source_record(legacy_path, legacy, "legacy_surface_discovery"),
            _source_record(union_path, union, "surface_union"),
        ],
        "entries": [
            {
                "inventory_entry_id": "legacy_surface_discovery:legacy:metrics.annual_return",
                "source_id": "legacy_surface_discovery",
                "raw_entry_id": legacy_entry["entry_id"],
                "raw_entry_sha256": checker.canonical_sha256(legacy_entry),
                "owner": "metrics",
                "disposition": "required",
                "capability_ids": ["metrics.annual_return"],
                "target_operation_id": "metrics.annual_return",
                "scenario_ids": ["ordinary_returns"],
                "source_nodeids": ["tests/test_metrics.py::test_annual_return"],
                "wheel_nodeids": ["tests/parity/test_metrics.py::test_annual_return"],
                "evidence": {"golden_path": "metrics/annual_return.json"},
                "completion_gate": "D-DOMAIN",
                "rationale": "The numerical return operation remains required in the unified metrics domain.",
                "rule_id": "unified-operation",
            },
            *supporting_entries,
        ],
    }


def _write_pair(tmp_path: Path) -> tuple[Path, Path, Path]:
    legacy_path = tmp_path / "legacy-discovery.json"
    union_path = tmp_path / "surface-union.json"
    _write_json(legacy_path, _legacy_discovery())
    _write_json(union_path, _surface_union())
    inventory_path = tmp_path / "complete-inventory.json"
    _write_json(inventory_path, _inventory(legacy_path, union_path))
    return legacy_path, union_path, inventory_path


def test_complete_inventory_binds_every_source_fact_exactly_once(tmp_path: Path) -> None:
    legacy_path, union_path, inventory_path = _write_pair(tmp_path)
    checker = _load_checker()

    result = checker.validate_complete_inventory(legacy_path, union_path, inventory_path)

    assert result["artifact_type"] == "complete_surface_inventory_validation"
    assert result["entry_count"] == 9
    assert result["unmapped_entries"] == []
    assert result["duplicate_entries"] == []
    assert result["not_a_d0_verdict"] is True


def test_complete_inventory_reader_preserves_regular_file_identity_without_no_follow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    legacy_path, union_path, inventory_path = _write_pair(tmp_path)
    checker = _load_checker()
    monkeypatch.delattr(checker.os, "O_NOFOLLOW", raising=False)

    result = checker.validate_complete_inventory(legacy_path, union_path, inventory_path)

    assert result["entry_count"] == 9

    alias = tmp_path / "inventory-alias.json"
    alias.symlink_to(inventory_path)
    with pytest.raises(checker.CompleteInventoryValidationError, match="symbolic link"):
        checker._read_regular_file(alias, "complete inventory")


def test_rejects_inventory_which_omits_one_union_fact(tmp_path: Path) -> None:
    legacy_path, union_path, inventory_path = _write_pair(tmp_path)
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    inventory["entries"].pop()
    _write_json(inventory_path, inventory)
    checker = _load_checker()

    with pytest.raises(checker.CompleteInventoryValidationError, match="unmapped"):
        checker.validate_complete_inventory(legacy_path, union_path, inventory_path)


def test_rejects_raw_provenance_drift_between_discovery_inputs(tmp_path: Path) -> None:
    legacy_path, union_path, inventory_path = _write_pair(tmp_path)
    union = _surface_union()
    provenance = union["source_provenance"]
    assert isinstance(provenance, dict)
    provenance["tree"] = "f" * 40
    _write_json(union_path, union)
    checker = _load_checker()

    with pytest.raises(checker.CompleteInventoryValidationError, match="provenance"):
        checker.validate_complete_inventory(legacy_path, union_path, inventory_path)


def test_rejects_scoped_or_not_for_d0_inventory_header(tmp_path: Path) -> None:
    legacy_path, union_path, inventory_path = _write_pair(tmp_path)
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    inventory["decision_status"] = "scoped"
    inventory["not_for_d0"] = True
    _write_json(inventory_path, inventory)
    checker = _load_checker()

    with pytest.raises(checker.CompleteInventoryValidationError, match="complete"):
        checker.validate_complete_inventory(legacy_path, union_path, inventory_path)


def test_rejects_required_entry_without_real_source_and_wheel_nodeids(tmp_path: Path) -> None:
    legacy_path, union_path, inventory_path = _write_pair(tmp_path)
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    required = inventory["entries"][0]
    assert isinstance(required, dict)
    required["source_nodeids"] = []
    required["wheel_nodeids"] = []
    _write_json(inventory_path, inventory)
    checker = _load_checker()

    with pytest.raises(checker.CompleteInventoryValidationError, match="source_nodeids and wheel_nodeids"):
        checker.validate_complete_inventory(legacy_path, union_path, inventory_path)


def test_rejects_duplicate_policy_keys(tmp_path: Path) -> None:
    legacy_path, union_path, inventory_path = _write_pair(tmp_path)
    payload = inventory_path.read_bytes().replace(
        b'"decision_status": "complete",',
        b'"decision_status": "complete", "decision_status": "complete",',
        1,
    )
    inventory_path.write_bytes(payload)
    checker = _load_checker()

    with pytest.raises(checker.CompleteInventoryValidationError, match="duplicate"):
        checker.validate_complete_inventory(legacy_path, union_path, inventory_path)
