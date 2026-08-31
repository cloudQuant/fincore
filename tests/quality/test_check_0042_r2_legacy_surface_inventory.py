"""Fail-closed contracts for the scoped 0042-R2 legacy-surface inventory.

The inventory in this slice is deliberately limited to the raw legacy
discovery artifact.  It proves that every discovered raw row has one reviewed
decision, without claiming D0 coverage for the still-uncollected repository
surfaces.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

REPOSITORY_ROOT = Path(os.environ.get("FINCORE_0042R2_SOURCE_ROOT", Path(__file__).parents[2])).resolve()
SCRIPT = REPOSITORY_ROOT / "scripts" / "check_0042_r2_legacy_surface_inventory.py"
COMMITTED_RAW_DISCOVERY = REPOSITORY_ROOT / "tests" / "parity" / "fixtures" / "legacy-surface-discovery-0042-r2.json"
COMMITTED_INVENTORY = REPOSITORY_ROOT / "tests" / "parity" / "fixtures" / "legacy-surface-inventory-0042-r2.json"


def _load_checker() -> Any:
    module_name = "fincore_0042_r2_legacy_surface_inventory_test"
    specification = importlib.util.spec_from_file_location(module_name, SCRIPT)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    specification.loader.exec_module(module)
    return module


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _minimal_raw_discovery() -> dict[str, object]:
    return {
        "schema_version": 1,
        "artifact_type": "legacy_surface_discovery",
        "discovery_status": "partial",
        "not_for_d0": True,
        "partial_reason": "Docs, examples, benchmarks, wheels, and test nodes are intentionally absent.",
        "required_source_kinds": ["metric_registry", "workflow_registry"],
        "source": {"commit": "a" * 40, "tree": "b" * 40, "clean": True},
        "source_artifacts": [
            {
                "source_id": "metric_registry",
                "source_kind": "metric_registry",
                "path": "fincore/_registry.py",
                "sha256": "c" * 64,
            },
            {
                "source_id": "workflow_registry",
                "source_kind": "workflow_registry",
                "path": "fincore/contracts/workflows.py",
                "sha256": "d" * 64,
            },
        ],
        "entries": [
            {
                "entry_id": "metric_registry:annual_return",
                "source_id": "metric_registry",
                "source_kind": "metric_registry",
                "source_locator": {
                    "artifact_path": "fincore/_registry.py",
                    "artifact_sha256": "c" * 64,
                    "locator": "METRIC_REGISTRY['annual_return']",
                },
            },
            {
                "entry_id": "workflow_registry:full_tear_sheet",
                "source_id": "workflow_registry",
                "source_kind": "workflow_registry",
                "source_locator": {
                    "artifact_path": "fincore/contracts/workflows.py",
                    "artifact_sha256": "d" * 64,
                    "locator": "WORKFLOW_REGISTRY['full_tear_sheet']",
                },
            },
        ],
    }


def _minimal_inventory(raw_path: Path) -> dict[str, object]:
    checker = _load_checker()
    raw = _minimal_raw_discovery()
    entries = []
    for raw_entry in raw["entries"]:
        source_locator = raw_entry["source_locator"]
        entries.append(
            {
                "legacy_entry_id": raw_entry["entry_id"],
                "source": {
                    "source_id": raw_entry["source_id"],
                    "source_kind": raw_entry["source_kind"],
                    "artifact_path": source_locator["artifact_path"],
                    "artifact_sha256": source_locator["artifact_sha256"],
                    "locator": source_locator["locator"],
                    "raw_entry_sha256": checker.canonical_sha256(raw_entry),
                },
                "owner": "performance",
                "disposition": "required",
                "target_operation_id": "returns.annual_return",
                "rationale": "The raw legacy surface is covered by a unified operation.",
                "rule_id": "unified-operation",
                "completion_gate": "D-RUNTIME",
                "capability_ids": ["returns.metrics"],
            }
        )
    return {
        "schema_version": 1,
        "artifact_type": "legacy_surface_inventory",
        "scope": "raw_legacy_surface_only",
        "decision_status": "scoped",
        "not_for_d0": True,
        "does_not_assert": ["D-TECH", "D0", "installed_wheel_behavior", "legacy_zero"],
        "source_contract": {
            "raw_artifact_type": "legacy_surface_discovery",
            "required_raw_status": "partial",
            "required_raw_not_for_d0": True,
        },
        "source_discovery": {
            "path": raw_path.name,
            "sha256": "TO_BE_FILLED_BY_TEST",
            "entry_count": 2,
            "source_provenance": {"commit": "a" * 40, "tree": "b" * 40, "clean": True},
        },
        "owners": ["performance"],
        "entries": entries,
    }


def _write_minimal_pair(tmp_path: Path) -> tuple[Path, Path]:
    raw_path = tmp_path / "legacy-surface-discovery.json"
    _write_json(raw_path, _minimal_raw_discovery())
    inventory_path = tmp_path / "legacy-surface-inventory.json"
    inventory = _minimal_inventory(raw_path)
    inventory["source_discovery"]["sha256"] = _load_checker().sha256_file(raw_path)
    _write_json(inventory_path, inventory)
    return raw_path, inventory_path


def test_byte_contract_matches_the_protected_path_wrapper(tmp_path: Path) -> None:
    checker = _load_checker()
    raw_path, inventory_path = _write_minimal_pair(tmp_path)

    from_paths = checker.validate_legacy_surface_inventory(raw_path, inventory_path)
    from_payloads = checker.validate_legacy_surface_inventory_payloads(
        raw_path.read_bytes(),
        inventory_path.read_bytes(),
        raw_filename=raw_path.name,
    )

    assert from_payloads == from_paths
    assert from_payloads["record_count"] == 2
    assert from_payloads["unmapped_entry_ids"] == []
    assert from_payloads["duplicate_entry_ids"] == []
    assert from_payloads["not_for_d0"] is True
    assert from_payloads["raw_discovery_sha256"] == checker.sha256_file(raw_path)
    assert from_payloads["inventory_sha256"] == checker.sha256_file(inventory_path)


def test_rejects_missing_raw_entry_mapping(tmp_path: Path) -> None:
    checker = _load_checker()
    raw_path, inventory_path = _write_minimal_pair(tmp_path)
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    inventory["entries"].pop()
    _write_json(inventory_path, inventory)

    with pytest.raises(checker.LegacySurfaceInventoryValidationError, match="unmapped"):
        checker.validate_legacy_surface_inventory(raw_path, inventory_path)


def test_rejects_duplicate_and_unknown_raw_entry_mappings(tmp_path: Path) -> None:
    checker = _load_checker()
    raw_path, inventory_path = _write_minimal_pair(tmp_path)
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    inventory["entries"].append(dict(inventory["entries"][0]))
    _write_json(inventory_path, inventory)

    with pytest.raises(checker.LegacySurfaceInventoryValidationError, match="duplicate"):
        checker.validate_legacy_surface_inventory(raw_path, inventory_path)

    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    inventory["entries"] = _minimal_inventory(raw_path)["entries"]
    inventory["entries"][1]["legacy_entry_id"] = "workflow_registry:unknown"
    _write_json(inventory_path, inventory)

    with pytest.raises(checker.LegacySurfaceInventoryValidationError, match="unknown"):
        checker.validate_legacy_surface_inventory(raw_path, inventory_path)


def test_rejects_source_locator_or_canonical_raw_entry_drift(tmp_path: Path) -> None:
    checker = _load_checker()
    raw_path, inventory_path = _write_minimal_pair(tmp_path)
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    inventory["entries"][0]["source"]["artifact_path"] = "fincore/not-the-registry.py"
    _write_json(inventory_path, inventory)

    with pytest.raises(checker.LegacySurfaceInventoryValidationError, match="artifact_path"):
        checker.validate_legacy_surface_inventory(raw_path, inventory_path)

    inventory = _minimal_inventory(raw_path)
    inventory["source_discovery"]["sha256"] = checker.sha256_file(raw_path)
    inventory["entries"][0]["source"]["raw_entry_sha256"] = "e" * 64
    _write_json(inventory_path, inventory)

    with pytest.raises(checker.LegacySurfaceInventoryValidationError, match="raw_entry_sha256"):
        checker.validate_legacy_surface_inventory(raw_path, inventory_path)


def test_rejects_unsupported_d0_or_unified_target_claims(tmp_path: Path) -> None:
    checker = _load_checker()
    raw_path, inventory_path = _write_minimal_pair(tmp_path)
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    raw["not_for_d0"] = False
    _write_json(raw_path, raw)
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    inventory["source_discovery"]["sha256"] = checker.sha256_file(raw_path)
    _write_json(inventory_path, inventory)

    with pytest.raises(checker.LegacySurfaceInventoryValidationError, match="partial not_for_d0"):
        checker.validate_legacy_surface_inventory(raw_path, inventory_path)

    raw_path, inventory_path = _write_minimal_pair(tmp_path)
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    inventory["entries"][0]["target_operation_id"] = None
    _write_json(inventory_path, inventory)

    with pytest.raises(checker.LegacySurfaceInventoryValidationError, match="target_operation_id"):
        checker.validate_legacy_surface_inventory(raw_path, inventory_path)


def test_accepts_dot_separated_capability_ids_from_the_existing_registry(tmp_path: Path) -> None:
    checker = _load_checker()
    raw_path, inventory_path = _write_minimal_pair(tmp_path)
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    inventory["entries"][0]["capability_ids"] = ["risk.evt"]
    _write_json(inventory_path, inventory)

    result = checker.validate_legacy_surface_inventory(raw_path, inventory_path)

    assert result["record_count"] == 2


def test_preparatory_inventory_refuses_alias_only_until_full_independent_evidence_exists(tmp_path: Path) -> None:
    checker = _load_checker()
    raw_path, inventory_path = _write_minimal_pair(tmp_path)
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    alias = inventory["entries"][1]
    alias["disposition"] = "alias_only"
    alias["equivalent_to_legacy_entry_id"] = inventory["entries"][0]["legacy_entry_id"]
    alias["equivalence_evidence"] = {
        "kind": "same_unified_operation",
        "reference": "returns.annual_return shared canonical implementation",
    }
    _write_json(inventory_path, inventory)

    with pytest.raises(checker.LegacySurfaceInventoryValidationError, match="only supports required"):
        checker.validate_legacy_surface_inventory(raw_path, inventory_path)


def test_preparatory_inventory_refuses_legacy_quirk_until_full_external_approval_exists(tmp_path: Path) -> None:
    checker = _load_checker()
    raw_path, inventory_path = _write_minimal_pair(tmp_path)
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    quirk = inventory["entries"][1]
    quirk.update(disposition="legacy_quirk", target_operation_id=None, capability_ids=[])
    quirk["nonfunctional_quirk_evidence"] = {
        "kind": "compatibility_shape_only",
        "reference": "No independent financial observable remains after the unified operation migration.",
    }
    inventory["owners"] = ["performance", "product", "reviewer"]
    quirk["retirement_approval"] = {
        "product_owner": "product",
        "independent_reviewer": "reviewer",
        "approved": True,
        "approval_reference": "R2-quirk-review-001",
    }
    _write_json(inventory_path, inventory)

    with pytest.raises(checker.LegacySurfaceInventoryValidationError, match="only supports required"):
        checker.validate_legacy_surface_inventory(raw_path, inventory_path)


def test_rejects_legacy_zero_or_final_verdict_claims_in_a_scoped_inventory(tmp_path: Path) -> None:
    checker = _load_checker()
    raw_path, inventory_path = _write_minimal_pair(tmp_path)
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    inventory["legacy_zero"] = True
    _write_json(inventory_path, inventory)

    with pytest.raises(checker.LegacySurfaceInventoryValidationError, match="assertion"):
        checker.validate_legacy_surface_inventory(raw_path, inventory_path)


def test_rejects_a_decision_bearing_raw_discovery(tmp_path: Path) -> None:
    checker = _load_checker()
    raw_path, inventory_path = _write_minimal_pair(tmp_path)
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    raw["entries"][0]["owner"] = "performance"
    _write_json(raw_path, raw)
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    inventory["source_discovery"]["sha256"] = checker.sha256_file(raw_path)
    inventory["entries"][0]["source"]["raw_entry_sha256"] = checker.canonical_sha256(raw["entries"][0])
    _write_json(inventory_path, inventory)

    with pytest.raises(checker.LegacySurfaceInventoryValidationError, match="decision fields"):
        checker.validate_legacy_surface_inventory(raw_path, inventory_path)


def test_rejects_normalized_d0_claims_in_raw_discovery_and_duplicate_json_keys(tmp_path: Path) -> None:
    checker = _load_checker()
    raw_path, inventory_path = _write_minimal_pair(tmp_path)
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    raw["D0"] = {"status": "passed"}
    _write_json(raw_path, raw)
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    inventory["source_discovery"]["sha256"] = checker.sha256_file(raw_path)
    _write_json(inventory_path, inventory)

    with pytest.raises(checker.LegacySurfaceInventoryValidationError, match="assertion"):
        checker.validate_legacy_surface_inventory(raw_path, inventory_path)

    raw_path, inventory_path = _write_minimal_pair(tmp_path)
    duplicate_key_payload = raw_path.read_bytes().replace(
        b'"not_for_d0": true,',
        b'"not_for_d0": true, "not_for_d0": true,',
        1,
    )
    with pytest.raises(checker.LegacySurfaceInventoryValidationError, match="duplicate JSON key"):
        checker.validate_legacy_surface_inventory_payloads(
            duplicate_key_payload,
            inventory_path.read_bytes(),
            raw_filename=raw_path.name,
        )


def test_path_wrapper_binds_parse_and_hash_to_the_same_raw_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checker = _load_checker()
    raw_path, inventory_path = _write_minimal_pair(tmp_path)
    original_reader = checker._read_regular_file
    original_raw = raw_path.read_bytes()
    swapped = False

    def read_then_swap(path: Path, label: str) -> bytes:
        nonlocal swapped
        payload = original_reader(path, label)
        if path == raw_path and not swapped:
            swapped = True
            raw_path.write_text('{"unexpected": true}\n', encoding="utf-8")
        return payload

    monkeypatch.setattr(checker, "_read_regular_file", read_then_swap)

    result = checker.validate_legacy_surface_inventory(raw_path, inventory_path)

    assert swapped is True
    assert result["record_count"] == 2
    assert result["raw_discovery_sha256"] == hashlib.sha256(original_raw).hexdigest()
    assert original_raw != raw_path.read_bytes()


def test_protected_reader_fails_closed_when_the_platform_lacks_no_follow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checker = _load_checker()
    raw_path, inventory_path = _write_minimal_pair(tmp_path)
    monkeypatch.delattr(checker.os, "O_NOFOLLOW", raising=False)

    with pytest.raises(checker.LegacySurfaceInventoryValidationError, match="O_NOFOLLOW"):
        checker.validate_legacy_surface_inventory(raw_path, inventory_path)


def test_cli_emits_a_scoped_success_summary(tmp_path: Path) -> None:
    _load_checker()
    raw_path, inventory_path = _write_minimal_pair(tmp_path)

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--raw-discovery", str(raw_path), "--inventory", str(inventory_path)],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    assert summary["scope"] == "raw_legacy_surface_only"
    assert summary["not_for_d0"] is True


def test_cli_is_fail_closed_for_an_invalid_inventory(tmp_path: Path) -> None:
    _load_checker()
    raw_path, inventory_path = _write_minimal_pair(tmp_path)
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    inventory["source_discovery"]["sha256"] = "0" * 64
    _write_json(inventory_path, inventory)

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--raw-discovery", str(raw_path), "--inventory", str(inventory_path)],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode != 0
    assert "sha256" in result.stderr


def test_committed_inventory_maps_each_discovered_raw_row_exactly_once() -> None:
    assert COMMITTED_RAW_DISCOVERY.is_file(), "committed raw legacy-surface discovery fixture is missing"
    assert COMMITTED_INVENTORY.is_file(), "committed legacy-surface inventory fixture is missing"
    checker = _load_checker()

    result = checker.validate_legacy_surface_inventory(COMMITTED_RAW_DISCOVERY, COMMITTED_INVENTORY)

    raw = json.loads(COMMITTED_RAW_DISCOVERY.read_text(encoding="utf-8"))
    expected_count = len(raw["entries"])
    assert result["record_count"] == expected_count
    assert result["unmapped_entry_ids"] == []
    assert result["duplicate_entry_ids"] == []
    assert result["not_for_d0"] is True
    assert result["scope"] == "raw_legacy_surface_only"
    assert result["disposition_counts"] == {"required": expected_count}
    assert result["raw_discovery_sha256"] == checker.sha256_file(COMMITTED_RAW_DISCOVERY)
    assert result["inventory_sha256"] == checker.sha256_file(COMMITTED_INVENTORY)
