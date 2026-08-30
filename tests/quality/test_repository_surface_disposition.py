"""Fail-closed contracts for 0042-R2 repository-surface dispositions.

The raw repository-surface collector deliberately records facts rather than
lifecycle decisions.  This contract verifies that a separate, reviewable
fixture maps every one of those facts exactly once without promoting the raw
artifact to D0 evidence.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

REPOSITORY_ROOT = Path(__file__).parents[2]
SCRIPT = REPOSITORY_ROOT / "scripts" / "check_0042_r2_repository_surface_disposition.py"
FACTS = REPOSITORY_ROOT / "tests" / "parity" / "fixtures" / "repository-surface-facts-discovery-0042-r2.json"
DISPOSITION = REPOSITORY_ROOT / "tests" / "parity" / "fixtures" / "repository-surface-disposition-0042-r2.json"


def _load_checker() -> Any:
    module_name = "fincore_0042_r2_repository_surface_disposition_test"
    specification = importlib.util.spec_from_file_location(module_name, SCRIPT)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    specification.loader.exec_module(module)
    return module


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _minimal_facts() -> dict[str, object]:
    return {
        "schema_version": 1,
        "artifact_type": "repository_surface_facts_discovery",
        "discovery_status": "partial",
        "not_for_d0": True,
        "boundaries": {
            "included": ["classified repository paths"],
            "excluded": "runtime execution and installed distributions",
        },
        "source_provenance": {"commit": "a" * 40, "tree": "b" * 40, "clean": True},
        "record_count": 2,
        "records": [
            {
                "path": "docs/architecture/adr/0042.md",
                "git_mode": "100644",
                "blob_sha256": "c" * 64,
                "kind": "historical_provenance_candidate",
                "category_tags": ["historical_provenance_candidate"],
                "token_facts": {"content_kind": "text"},
            },
            {
                "path": "scripts/check_quality.py",
                "git_mode": "100755",
                "blob_sha256": "d" * 64,
                "kind": "script_candidate",
                "category_tags": ["script_candidate"],
                "token_facts": {"content_kind": "text"},
            },
        ],
    }


def _minimal_disposition(facts_path: Path) -> dict[str, object]:
    checker = _load_checker()
    return {
        "schema_version": 1,
        "artifact_type": "repository_surface_disposition",
        "scope": "classified_repository_surface_only",
        "decision_status": "scoped",
        "not_for_d0": True,
        "does_not_assert": ["D-TECH", "D0", "installed_wheel_behavior", "legacy_zero"],
        "owners": ["architecture", "quality"],
        "source_facts": {
            "path": facts_path.name,
            "sha256": "TO_BE_FILLED_BY_TEST",
            "record_count": 2,
            "source_provenance": {"commit": "a" * 40, "tree": "b" * 40, "clean": True},
        },
        "source_contract": {
            "raw_artifact_type": "repository_surface_facts_discovery",
            "required_raw_status": "partial",
            "required_raw_not_for_d0": True,
            "boundaries_sha256": checker.canonical_sha256(_minimal_facts()["boundaries"]),
        },
        "entries": [
            {
                "path": "docs/architecture/adr/0042.md",
                "source": {
                    "git_mode": "100644",
                    "blob_sha256": "c" * 64,
                    "kind": "historical_provenance_candidate",
                    "category_tags": ["historical_provenance_candidate"],
                    "content_kind": "text",
                },
                "owner": "architecture",
                "lifecycle": "historical_provenance",
                "disposition": "allowlist",
                "completion_gate": "D0",
                "target": None,
                "legacy_reference_policy": "text_only_allowlist",
                "rationale": "Historical ADR remains an immutable provenance record.",
                "rule_id": "historical-adr",
            },
            {
                "path": "scripts/check_quality.py",
                "source": {
                    "git_mode": "100755",
                    "blob_sha256": "d" * 64,
                    "kind": "script_candidate",
                    "category_tags": ["script_candidate"],
                    "content_kind": "text",
                },
                "owner": "quality",
                "lifecycle": "maintained",
                "disposition": "retarget",
                "completion_gate": "D-RUNTIME",
                "target": {
                    "path": "scripts/check_quality.py",
                    "contract_ids": ["quality.contract"],
                    "capability_ids": [],
                },
                "legacy_reference_policy": "none",
                "rationale": "Retain the quality command while updating its 0.5 contract.",
                "rule_id": "quality-command",
            },
        ],
        "historical_provenance_allowlist": [
            {
                "path": "docs/architecture/adr/0042.md",
                "blob_sha256": "c" * 64,
                "reason": "Historical ADR remains an immutable provenance record.",
                "text_only": True,
            }
        ],
    }


def _write_minimal_pair(tmp_path: Path) -> tuple[Path, Path]:
    facts_path = tmp_path / "facts.json"
    _write_json(facts_path, _minimal_facts())
    disposition_path = tmp_path / "disposition.json"
    disposition = _minimal_disposition(facts_path)
    disposition["source_facts"]["sha256"] = _load_checker().sha256_file(facts_path)
    _write_json(disposition_path, disposition)
    return facts_path, disposition_path


def test_committed_disposition_maps_each_frozen_fact_exactly_once() -> None:
    checker = _load_checker()

    result = checker.validate_disposition(FACTS, DISPOSITION)

    assert result["record_count"] == 319
    assert result["unmapped_paths"] == []
    assert result["duplicate_paths"] == []
    assert result["not_for_d0"] is True


def test_byte_contract_matches_the_protected_path_wrapper() -> None:
    checker = _load_checker()

    from_paths = checker.validate_disposition(FACTS, DISPOSITION)
    from_payloads = checker.validate_disposition_payloads(
        FACTS.read_bytes(),
        DISPOSITION.read_bytes(),
        facts_filename=FACTS.name,
    )

    assert from_payloads == from_paths
    assert from_payloads["facts_sha256"] == checker.sha256_file(FACTS)
    assert from_payloads["disposition_sha256"] == checker.sha256_file(DISPOSITION)


def test_rejects_duplicate_json_policy_keys(tmp_path: Path) -> None:
    checker = _load_checker()
    facts_path, disposition_path = _write_minimal_pair(tmp_path)
    duplicate_key_payload = facts_path.read_bytes().replace(
        b'"not_for_d0": true,',
        b'"not_for_d0": true, "not_for_d0": true,',
        1,
    )

    with pytest.raises(checker.DispositionValidationError, match="duplicate JSON key"):
        checker.validate_disposition_payloads(
            duplicate_key_payload,
            disposition_path.read_bytes(),
            facts_filename=facts_path.name,
        )


def test_rejects_normalized_d0_assertion_claims(tmp_path: Path) -> None:
    checker = _load_checker()
    facts_path, disposition_path = _write_minimal_pair(tmp_path)
    facts = json.loads(facts_path.read_text(encoding="utf-8"))
    facts["D0"] = {"status": "passed"}
    _write_json(facts_path, facts)
    disposition = json.loads(disposition_path.read_text(encoding="utf-8"))
    disposition["source_facts"]["sha256"] = checker.sha256_file(facts_path)
    _write_json(disposition_path, disposition)

    with pytest.raises(checker.DispositionValidationError, match="assertion"):
        checker.validate_disposition(facts_path, disposition_path)


def test_rejects_missing_record_mapping_before_claiming_success(tmp_path: Path) -> None:
    checker = _load_checker()
    facts_path, disposition_path = _write_minimal_pair(tmp_path)
    disposition = json.loads(disposition_path.read_text(encoding="utf-8"))
    disposition["entries"].pop()
    _write_json(disposition_path, disposition)

    with pytest.raises(checker.DispositionValidationError, match="unmapped"):
        checker.validate_disposition(facts_path, disposition_path)


def test_rejects_source_fact_digest_or_category_drift(tmp_path: Path) -> None:
    checker = _load_checker()
    facts_path, disposition_path = _write_minimal_pair(tmp_path)
    disposition = json.loads(disposition_path.read_text(encoding="utf-8"))
    disposition["entries"][1]["source"]["category_tags"] = ["active_workflow"]
    _write_json(disposition_path, disposition)

    with pytest.raises(checker.DispositionValidationError, match="category_tags"):
        checker.validate_disposition(facts_path, disposition_path)


def test_rejects_historical_record_without_immutable_allowlist(tmp_path: Path) -> None:
    checker = _load_checker()
    facts_path, disposition_path = _write_minimal_pair(tmp_path)
    disposition = json.loads(disposition_path.read_text(encoding="utf-8"))
    disposition["entries"][0]["disposition"] = "retarget"
    disposition["entries"][0]["target"] = {
        "path": "docs/architecture/adr/0042.md",
        "contract_ids": [],
        "capability_ids": [],
    }
    _write_json(disposition_path, disposition)

    with pytest.raises(checker.DispositionValidationError, match="historical_provenance"):
        checker.validate_disposition(facts_path, disposition_path)


def test_rejects_historical_candidate_reclassified_as_a_transition(tmp_path: Path) -> None:
    checker = _load_checker()
    facts_path, disposition_path = _write_minimal_pair(tmp_path)
    disposition = json.loads(disposition_path.read_text(encoding="utf-8"))
    entry = disposition["entries"][0]
    entry.update(
        lifecycle="transition",
        disposition="remove",
        completion_gate="D-CUTOVER",
        legacy_reference_policy="retarget_before_cutover",
        target=None,
    )
    disposition["historical_provenance_allowlist"] = []
    _write_json(disposition_path, disposition)

    with pytest.raises(checker.DispositionValidationError, match="historical source requires historical_provenance"):
        checker.validate_disposition(facts_path, disposition_path)


def test_rejects_binary_historical_artifact_claimed_as_text_only(tmp_path: Path) -> None:
    checker = _load_checker()
    facts_path, disposition_path = _write_minimal_pair(tmp_path)
    facts = json.loads(facts_path.read_text(encoding="utf-8"))
    facts["records"][0]["token_facts"]["content_kind"] = "binary"
    _write_json(facts_path, facts)
    disposition = json.loads(disposition_path.read_text(encoding="utf-8"))
    disposition["source_facts"]["sha256"] = checker.sha256_file(facts_path)
    disposition["entries"][0]["source"]["content_kind"] = "binary"
    _write_json(disposition_path, disposition)

    with pytest.raises(checker.DispositionValidationError, match="artifact_provenance_allowlist"):
        checker.validate_disposition(facts_path, disposition_path)


def test_binds_facts_parse_and_digest_to_one_protected_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    checker = _load_checker()
    facts_path, disposition_path = _write_minimal_pair(tmp_path)
    original_reader = checker._read_regular_file
    original_facts = facts_path.read_bytes()
    swapped = False

    def read_then_swap(path: Path, label: str) -> bytes:
        nonlocal swapped
        payload = original_reader(path, label)
        if path == facts_path and not swapped:
            swapped = True
            facts_path.write_text('{"unexpected": true}\n', encoding="utf-8")
        return payload

    monkeypatch.setattr(checker, "_read_regular_file", read_then_swap)

    result = checker.validate_disposition(facts_path, disposition_path)

    assert swapped is True
    assert result["record_count"] == 2
    assert original_facts != facts_path.read_bytes()


def test_protected_reader_fails_closed_when_the_platform_lacks_no_follow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checker = _load_checker()
    facts_path, disposition_path = _write_minimal_pair(tmp_path)
    monkeypatch.delattr(checker.os, "O_NOFOLLOW", raising=False)

    with pytest.raises(checker.DispositionValidationError, match="O_NOFOLLOW"):
        checker.validate_disposition(facts_path, disposition_path)


def test_rejects_historical_allowlist_digest_drift(tmp_path: Path) -> None:
    checker = _load_checker()
    facts_path, disposition_path = _write_minimal_pair(tmp_path)
    disposition = json.loads(disposition_path.read_text(encoding="utf-8"))
    disposition["historical_provenance_allowlist"][0]["blob_sha256"] = "e" * 64
    _write_json(disposition_path, disposition)

    with pytest.raises(checker.DispositionValidationError, match="allowlist"):
        checker.validate_disposition(facts_path, disposition_path)


def test_cli_is_fail_closed_for_an_invalid_disposition(tmp_path: Path) -> None:
    _load_checker()
    facts_path, disposition_path = _write_minimal_pair(tmp_path)
    disposition = json.loads(disposition_path.read_text(encoding="utf-8"))
    disposition["source_facts"]["sha256"] = "0" * 64
    _write_json(disposition_path, disposition)

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--facts", str(facts_path), "--disposition", str(disposition_path)],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode != 0
    assert "sha256" in result.stderr
